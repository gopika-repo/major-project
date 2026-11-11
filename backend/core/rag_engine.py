import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from config import settings
from core.text_extractor import TextExtractor
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Single Gemini client for the whole project
client = genai.Client(api_key=settings.GEMINI_API_KEY)


class RAGEngine:
    """
    The brain of the project.
    Connects text extraction, embeddings, vector store
    and Gemini together into one pipeline.
    """

    def __init__(self):
        self.text_extractor = TextExtractor()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        logger.info("RAGEngine initialized ✅")

    def process_document(self, pdf_path, doc_name):
        """
        Full pipeline for uploading a PDF:
        1. Extract text from every page
        2. Split into chunks and generate embeddings
        3. Store all vectors in Qdrant

        Returns a summary dict with stats.
        """
        try:
            # Step 1 — Extract text
            print(f"\n📄 Step 1: Reading PDF...")
            pages = self.text_extractor.extract_with_metadata(pdf_path)
            print(f"   ✅ Extracted {len(pages)} pages")

            # Step 2 — Chunk and embed
            print(f"\n🔢 Step 2: Creating embeddings...")
            chunks = self.embedder.prepare_chunks_for_storage(
                pages, doc_name
            )
            print(f"   ✅ Created {len(chunks)} chunks")

            # Step 3 — Store in Qdrant
            print(f"\n💾 Step 3: Storing in Qdrant...")
            stored = self.vector_store.upsert_chunks(chunks)
            print(f"   ✅ Stored {stored} vectors")

            return {
                "status": "success",
                "doc_name": doc_name,
                "pages_processed": len(pages),
                "chunks_created": len(chunks),
                "vectors_stored": stored
            }

        except Exception as e:
            logger.error(f"process_document failed: {e}")
            raise

    def query(self, question, doc_name):
        """
        Full pipeline for answering a question:
        1. Embed the question
        2. Search Qdrant for similar chunks
        3. Send chunks + question to Gemini
        4. Return answer with sources and confidence

        Returns a dict with answer, sources, confidence.
        """
        try:
            # Step 1 — Embed the question
            q_embedding = self.embedder.generate_query_embedding(question)

            # Step 2 — Search Qdrant
            results = self.vector_store.query(
                embedding=q_embedding,
                top_k=settings.TOP_K,
                filter_dict={"doc_name": {"$eq": doc_name}}
            )

            # If nothing found
            if not results:
                return {
                    "answer": "No relevant content found in the document.",
                    "sources": [],
                    "confidence": 0.0,
                    "question": question,
                    "doc_name": doc_name
                }

            # Step 3 — Build prompt
            prompt = self._build_prompt(question, results)

            # Step 4 — Ask Gemini with retry on 503
            answer = self._generate_with_retry(prompt)

            # Step 5 — Calculate confidence score
            top_scores = [r["score"] for r in results[:3]]
            confidence = round(
                sum(top_scores) / len(top_scores), 2
            )

            # Step 6 — Build sources list
            sources = [
                {
                    "page_num": r["metadata"].get("page_num"),
                    "section": r["metadata"].get("section", "Unknown"),
                    "score": r["score"],
                    "text_preview": r["text"][:100] + "..."
                }
                for r in results
            ]

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "question": question,
                "doc_name": doc_name
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def _generate_with_retry(self, prompt):
        """
        Calls Gemini to generate an answer.
        Retries up to 3 times if server is busy (503).
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=settings.MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=600
                    )
                )
                return response.text

            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"\n  ⏳ Server busy, retrying in {wait}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

    def _build_prompt(self, question, chunks):
        """
        Builds the prompt we send to Gemini.
        Includes retrieved chunks as context.
        """
        context_parts = []

        for chunk in chunks[:5]:
            page = chunk["metadata"].get("page_num", "?")
            section = chunk["metadata"].get("section", "Unknown")
            context_parts.append(
                f"[Page {page} | Section: {section}]\n{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return f"""You are a professional financial analyst assistant.
Answer the question using ONLY the document context below.
Always mention specific numbers and page numbers in your answer.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Give a clear, specific answer with figures and page references:"""

    def categorize_document(self, doc_name):
        """
        Runs 4 automatic queries to understand
        what kind of document was uploaded.
        """
        try:
            return {
                "doc_type": self.query(
                    "What type of document is this? Annual report, quarterly results, etc?",
                    doc_name
                )["answer"],
                "company_info": self.query(
                    "What is the company name and fiscal year?",
                    doc_name
                )["answer"],
                "sections_found": self.query(
                    "What are the main financial sections in this document?",
                    doc_name
                )["answer"],
                "revenue_highlight": self.query(
                    "What is the total revenue or net sales mentioned?",
                    doc_name
                )["answer"]
            }
        except Exception as e:
            logger.error(f"categorize_document failed: {e}")
            return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("  Visual Financial Analyst — RAG Demo")
    print("=" * 50)

    rag = RAGEngine()

    pdf_path = input("\nEnter PDF path: ").strip()
    doc_name = input("Enter document name (e.g. tcs2023): ").strip()

    # Process the PDF
    result = rag.process_document(pdf_path, doc_name)

    print("\n" + "=" * 50)
    print("  ✅ Document Processed!")
    print(f"  Pages   : {result['pages_processed']}")
    print(f"  Chunks  : {result['chunks_created']}")
    print(f"  Vectors : {result['vectors_stored']}")
    print("=" * 50)

    # Test 3 questions
    questions = [
        "What is the total revenue?",
        "What was the net profit?",
        "What are the key risk factors?"
    ]

    print("\n--- Answering 3 Test Questions ---\n")

    for q in questions:
        print(f"Q: {q}")
        ans = rag.query(q, doc_name)
        print(f"A: {ans['answer'][:200]}")
        print(f"   Confidence : {ans['confidence']}")
        if ans["sources"]:
            print(f"   Source     : Page {ans['sources'][0]['page_num']}")
        print()