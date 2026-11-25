import logging
import sys
import os
import time

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from google import genai
from google.genai import types

from config import settings
from core.text_extractor import TextExtractor
from core.table_extractor import TableExtractor
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore


logger = logging.getLogger(__name__)

# One Gemini client for answer generation.
client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


class RAGEngine:
    """
    Main engine for Visual Financial Analyst.

    Supports PDF text extraction, financial table extraction,
    embeddings, Qdrant storage, separate text/table retrieval,
    and source-aware Gemini answer generation.
    """

    def __init__(self):
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()

        logger.info("Multimodal RAGEngine initialized.")

    # ==================================================
    # 1. DOCUMENT PROCESSING
    # ==================================================

    def process_document(
        self,
        pdf_path,
        doc_name,
        table_pages=None,
        coordinate_configs=None
    ):
        """Process PDF text and optional financial tables."""

        try:
            print("\nSTEP 1: Extracting PDF text...")

            pages = self.text_extractor.extract_with_metadata(
                pdf_path
            )

            print(f"Text pages extracted: {len(pages)}")
            print("\nSTEP 2: Preparing text chunks...")

            text_chunks = self.embedder.prepare_chunks_for_storage(
                pages,
                doc_name
            )

            print(f"Text chunks created: {len(text_chunks)}")

            tables = []
            table_chunks = []
            table_status = "not_requested"

            if table_pages:
                print("\nSTEP 3: Extracting financial tables...")

                try:
                    tables = self.table_extractor.extract_tables(
                        pdf_path,
                        pages=table_pages,
                        coordinate_configs=coordinate_configs
                    )

                    print(f"Tables extracted: {len(tables)}")

                    if tables:
                        print("\nPreparing table embeddings...")

                        table_chunks = (
                            self.embedder.prepare_table_chunks_for_storage(
                                tables,
                                doc_name
                            )
                        )

                    table_status = "success"
                    print(f"Table chunks created: {len(table_chunks)}")

                except Exception as table_error:
                    logger.exception("Table processing failed.")
                    table_status = "failed"
                    print(f"Table processing warning: {table_error}")
                    tables = []
                    table_chunks = []
            else:
                print(
                    "\nNo table pages supplied. "
                    "Using text-only processing."
                )

            all_chunks = text_chunks + table_chunks

            print(f"\nTotal chunks prepared: {len(all_chunks)}")
            print("\nSTEP 5: Storing vectors in Qdrant...")

            stored = self.vector_store.upsert_chunks(all_chunks)

            print(f"Vectors stored: {stored}")

            return {
                "status": "success",
                "doc_name": doc_name,
                "pages_processed": len(pages),
                "chunks_created": len(all_chunks),
                "text_chunks_created": len(text_chunks),
                "table_chunks_created": len(table_chunks),
                "tables_extracted": len(tables),
                "table_processing_status": table_status,
                "vectors_stored": stored
            }

        except Exception as e:
            logger.exception(f"Document processing failed: {e}")
            raise

    # ==================================================
    # 2. TABLE-ONLY PROCESSING
    # ==================================================

    def process_tables_for_document(
        self,
        pdf_path,
        doc_name,
        pages,
        coordinate_configs=None
    ):
        """Add financial table vectors without reprocessing text."""

        try:
            print("\nSTEP 1: Extracting financial tables...")

            tables = self.table_extractor.extract_tables(
                pdf_path,
                pages=pages,
                coordinate_configs=coordinate_configs
            )

            print(f"Tables extracted: {len(tables)}")

            if not tables:
                return {
                    "status": "success",
                    "doc_name": doc_name,
                    "tables_extracted": 0,
                    "table_chunks_created": 0,
                    "vectors_stored": 0
                }

            print("\nSTEP 2: Generating table embeddings...")

            table_chunks = (
                self.embedder.prepare_table_chunks_for_storage(
                    tables,
                    doc_name
                )
            )

            print(f"Table chunks prepared: {len(table_chunks)}")
            print("\nSTEP 3: Storing table vectors...")

            stored = self.vector_store.upsert_chunks(table_chunks)

            print(f"Table vectors stored: {stored}")

            return {
                "status": "success",
                "doc_name": doc_name,
                "tables_extracted": len(tables),
                "table_chunks_created": len(table_chunks),
                "vectors_stored": stored
            }

        except Exception as e:
            logger.exception(f"Table processing failed: {e}")
            raise

    # ==================================================
    # 3. SEPARATE TEXT AND TABLE RETRIEVAL
    # ==================================================

    def retrieve_context(self, question, doc_name):
        """
        Retrieve text and table chunks independently so that
        text results cannot occupy all retrieval positions.
        """

        q_embedding = self.embedder.generate_query_embedding(
            question
        )

        text_results = self.vector_store.query(
            embedding=q_embedding,
            top_k=settings.TOP_K,
            filter_dict={
                "doc_name": {"$eq": doc_name},
                "type": {"$ne": "table"}
            }
        )

        table_results = self.vector_store.query(
            embedding=q_embedding,
            top_k=settings.TOP_K,
            filter_dict={
                "doc_name": {"$eq": doc_name},
                "type": {"$eq": "table"}
            }
        )

        combined_results = text_results + table_results

        combined_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        logger.info(
            "Retrieval completed: "
            f"{len(text_results)} text chunks, "
            f"{len(table_results)} table chunks."
        )

        return {
            "text_results": text_results,
            "table_results": table_results,
            "combined_results": combined_results
        }

    # ==================================================
    # 4. QUESTION ANSWERING
    # ==================================================

    def query(self, question, doc_name):
        """Answer a question using retrieved text and table evidence."""

        try:
            print("\nRETRIEVING DOCUMENT CONTEXT...")

            retrieval = self.retrieve_context(
                question,
                doc_name
            )

            text_results = retrieval["text_results"]
            table_results = retrieval["table_results"]
            results = retrieval["combined_results"]

            print("\nRETRIEVAL SUMMARY")
            print("-----------------")
            print(f"Text chunks: {len(text_results)}")
            print(f"Table chunks: {len(table_results)}")
            print(f"Total chunks: {len(results)}")

            if not results:
                return {
                    "answer": "No relevant content found in the document.",
                    "sources": [],
                    "confidence": 0.0,
                    "question": question,
                    "doc_name": doc_name
                }

            prompt = self._build_prompt(question, results)

            answer = self._generate_with_retry(prompt)

            top_scores = [
                result["score"]
                for result in results[:3]
            ]

            confidence = (
                round(sum(top_scores) / len(top_scores), 2)
                if top_scores else 0.0
            )

            sources = []

            for result in results:
                metadata = result["metadata"]
                source_type = metadata.get("type", "text")

                sources.append({
                    "page_num": metadata.get("page_num"),
                    "section": metadata.get("section", "Unknown"),
                    "type": source_type,
                    "table_type": metadata.get("table_type"),
                    "score": result["score"],
                    "text_preview": (
                        result["text"][:200] + "..."
                        if len(result["text"]) > 200
                        else result["text"]
                    )
                })

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "question": question,
                "doc_name": doc_name
            }

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            raise

    # ==================================================
    # 5. GEMINI ANSWER GENERATION
    # ==================================================

    def _generate_with_retry(self, prompt):
        """
        Generate a complete answer with minimal thinking and a
        sufficient output budget. Retry temporary 503 errors.
        Detect MAX_TOKENS instead of returning a truncated answer.
        """

        max_retries = 3
        max_output_tokens = 2048

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=settings.MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=max_output_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.MINIMAL
                        )
                    )
                )

                candidates = response.candidates or []
                finish_reason = (
                    candidates[0].finish_reason
                    if candidates else None
                )

                finish_reason_text = str(finish_reason)

                if "MAX_TOKENS" in finish_reason_text:
                    logger.error(
                        "Gemini reached MAX_TOKENS. "
                        "The response may be incomplete."
                    )
                    raise RuntimeError(
                        "Gemini stopped because it reached the output-token "
                        "limit. Increase max_output_tokens and retry."
                    )

                answer = response.text

                if not answer or not answer.strip():
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return answer.strip()

            except Exception as e:
                if (
                    "503" in str(e)
                    and attempt < max_retries - 1
                ):
                    wait = (attempt + 1) * 10

                    print(
                        f"\nGemini server is busy. "
                        f"Retrying in {wait} seconds..."
                    )

                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            "Gemini could not generate an answer after multiple attempts."
        )

    # ==================================================
    # 6. SOURCE-AWARE RAG PROMPT
    # ==================================================

    def _build_prompt(self, question, chunks):
        """Build a financial RAG prompt from retrieved evidence."""

        context_parts = []
        max_context_chunks = settings.TOP_K * 2

        for index, chunk in enumerate(
            chunks[:max_context_chunks],
            start=1
        ):
            metadata = chunk["metadata"]

            page = metadata.get("page_num", "Unknown")
            source_type = metadata.get("type", "text")
            section = metadata.get("section", "Unknown")
            table_type = metadata.get("table_type", "Not applicable")

            context_parts.append(
                f"""
SOURCE {index}

Page: {page}

Content type: {source_type}

Section: {section}

Table category: {table_type}

Similarity score: {chunk["score"]}

CONTENT:
{chunk["text"]}
"""
            )

        context = "\n\n---\n\n".join(context_parts)

        return f"""
You are a professional financial analyst assistant.

Answer using ONLY the supplied document context.

IMPORTANT RULES:

1. Never invent financial figures.
2. Preserve numerical values exactly as provided.
3. Distinguish between financial years.
4. Carefully interpret financial tables.
5. Never confuse revenue, expenses and profit.
6. Mention the page supporting important claims.
7. If evidence is insufficient, say so clearly.
8. Do not assume missing values.
9. Do not calculate financial changes unless
   the required figures are available.
10. Treat table content as structured financial evidence.
11. If text and table sources disagree,
    identify the discrepancy instead of guessing.
12. Do not claim a figure is consolidated or standalone
    unless the source establishes that distinction.
13. For a direct numerical question, answer the requested
    figure clearly near the beginning of the response.
14. Do not add unrelated figures unless they help clarify
    the requested answer.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

Provide a clear and evidence-based financial answer
with numerical figures and page references.
"""

    # ==================================================
    # 7. DOCUMENT CATEGORIZATION
    # ==================================================

    def categorize_document(self, doc_name):
        """Run standard questions to understand the document."""

        try:
            return {
                "doc_type": self.query(
                    "What type of document is this? "
                    "Annual report, quarterly results, etc?",
                    doc_name
                )["answer"],

                "company_info": self.query(
                    "What is the company name and fiscal year?",
                    doc_name
                )["answer"],

                "sections_found": self.query(
                    "What are the main financial sections "
                    "in this document?",
                    doc_name
                )["answer"],

                "revenue_highlight": self.query(
                    "What is the total revenue or net sales mentioned?",
                    doc_name
                )["answer"]
            }

        except Exception as e:
            logger.exception(
                f"Document categorization failed: {e}"
            )
            return {}

# ==================================================
# 8. COMMAND-LINE DEMO
# ==================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 55)
    print("VISUAL FINANCIAL ANALYST – RAG DEMO")
    print("=" * 55)

    rag = RAGEngine()

    try:
        pdf_path = input("\nEnter PDF path: ").strip()
        doc_name = input("Enter document name: ").strip()

        result = rag.process_document(
            pdf_path,
            doc_name
        )

        print("\nDOCUMENT PROCESSING SUMMARY")
        print("Pages:", result["pages_processed"])
        print("Text chunks:", result["text_chunks_created"])
        print("Table chunks:", result["table_chunks_created"])
        print("Vectors:", result["vectors_stored"])

        while True:
            question = input(
                "\nAsk a question (or type exit): "
            ).strip()

            if question.lower() == "exit":
                break

            answer = rag.query(question, doc_name)

            print("\nANSWER:")
            print(answer["answer"])

            print("\nSOURCES:")
            for source in answer["sources"]:
                print(
                    f"Page {source['page_num']} | "
                    f"Type: {source['type']} | "
                    f"Score: {source['score']}"
                )

    finally:
        if rag.vector_store.client is not None:
            rag.vector_store.client.close()
