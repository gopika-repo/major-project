
import time
import logging
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from google import genai
from config import settings


logger = logging.getLogger(__name__)

# Create one Gemini client — used for everything.
client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


class EmbeddingGenerator:
    """
    Converts text and financial tables into vectors
    using Google Gemini Embeddings.

    Supports:
        1. Ordinary text chunks
        2. Financial table chunks
        3. User query embeddings
    """

    def chunk_text(self, text):
        """
        Splits a long text into smaller overlapping chunks.

        Each chunk = 400 words with 50 word overlap.
        """

        words = text.split()

        # If text is too short, return as one chunk.
        if len(words) < 15:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(words):

            end = start + settings.CHUNK_SIZE

            chunk = " ".join(words[start:end])

            # Only keep chunks with at least 15 words.
            if len(chunk.split()) >= 15:
                chunks.append(chunk)

            # Move forward with overlap.
            start += (
                settings.CHUNK_SIZE -
                settings.CHUNK_OVERLAP
            )

        return chunks

    def generate_embedding(self, text):
        """
        Converts one piece of text into a vector.
        """

        try:

            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )

            return result.embeddings[0].values

        except Exception as e:

            logger.warning(
                f"Embedding failed, retrying in 5s: {e}"
            )

            time.sleep(5)

            # Try one more time.
            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )

            return result.embeddings[0].values

    def generate_query_embedding(self, text):
        """
        Generates an embedding for a user question.
        """

        try:

            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )

            return result.embeddings[0].values

        except Exception as e:

            logger.error(
                f"Query embedding failed: {e}"
            )

            raise

    def batch_embed(self, texts):
        """
        Embeds multiple texts one by one.

        Handles Gemini rate limits.
        """

        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):

            print(
                f"  Embedding {i + 1}/{total}...",
                end="\r"
            )

            max_retries = 5

            for attempt in range(max_retries):

                try:

                    embedding = self.generate_embedding(text)

                    embeddings.append(embedding)

                    break

                except Exception as e:

                    if (
                        "429" in str(e)
                        or "RESOURCE_EXHAUSTED" in str(e)
                    ):

                        wait = (attempt + 1) * 30

                        print(
                            f"\n  Rate limit hit. "
                            f"Waiting {wait}s "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )

                        time.sleep(wait)

                    else:
                        raise

            else:

                raise RuntimeError(
                    f"Could not generate embedding "
                    f"for item {i + 1} after "
                    f"{max_retries} attempts."
                )

            # Pause every 5 embeddings.
            if (i + 1) % 5 == 0:
                time.sleep(2)

        print(
            f"  Embedded {total}/{total} chunks."
        )

        return embeddings

    def prepare_chunks_for_storage(self, pages, doc_name):
        """
        Existing Phase 1 method.

        Takes extracted PDF pages, creates text chunks,
        generates embeddings and returns Qdrant-ready records.
        """

        all_chunks = []

        # Step 1: Chunk every page.
        for page in pages:

            text = page.get("text", "")

            page_num = page.get("page_num", 0)

            section = page.get("section", "UNKNOWN")

            chunks = self.chunk_text(text)

            for idx, chunk in enumerate(chunks):

                all_chunks.append({
                    "text": chunk,
                    "page_num": page_num,
                    "section": section,
                    "chunk_index": idx,
                    "doc_name": doc_name
                })

        logger.info(
            f"Total text chunks created: {len(all_chunks)}"
        )

        # Step 2: Generate embeddings.
        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = self.batch_embed(texts)

        if len(embeddings) != len(all_chunks):
            raise RuntimeError(
                "Text embedding count does not match chunk count."
            )

        # Step 3: Combine everything.
        result = []

        for i, chunk in enumerate(all_chunks):

            result.append({
                "id": (
                    f"{doc_name}_p{chunk['page_num']}"
                    f"_c{chunk['chunk_index']}"
                ),

                "text": chunk["text"],

                "embedding": embeddings[i],

                "metadata": {
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "word_count": len(
                        chunk["text"].split()
                    )
                }
            })

        return result

    def prepare_table_chunks_for_storage(
        self,
        tables,
        doc_name
    ):
        """
        Phase 2 method.

        Converts extracted financial tables into
        embedding-ready chunks for Qdrant.

        Each table becomes one searchable chunk.

        Required table fields:
            page_num
            table_index
            table_type
            text

        Returns:
            List of Qdrant-ready table chunk dictionaries.
        """

        if not doc_name or not doc_name.strip():

            raise ValueError(
                "doc_name cannot be empty."
            )

        if not tables:

            logger.warning(
                "No tables received for embedding."
            )

            return []

        all_tables = []

        # Step 1: Prepare extracted table content.
        for idx, table in enumerate(tables):

            if not isinstance(table, dict):

                raise ValueError(
                    f"Invalid table record at index {idx}."
                )

            table_text = (
                table.get("text") or ""
            ).strip()

            if not table_text:

                logger.warning(
                    f"Skipping empty table at index {idx}."
                )

                continue

            page_num = table.get("page_num")

            if page_num is None:

                raise ValueError(
                    f"Table {idx} is missing page_num."
                )

            table_index = table.get(
                "table_index",
                idx
            )

            table_type = table.get(
                "table_type",
                "other"
            )

            all_tables.append({
                "text": table_text,
                "page_num": page_num,
                "table_index": table_index,
                "table_type": table_type,
                "doc_name": doc_name
            })

        if not all_tables:

            logger.warning(
                "No valid tables available for embedding."
            )

            return []

        logger.info(
            f"Total table chunks created: {len(all_tables)}"
        )

        # Step 2: Generate embeddings for table Markdown.
        texts = [
            table["text"]
            for table in all_tables
        ]

        embeddings = self.batch_embed(texts)

        if len(embeddings) != len(all_tables):

            raise RuntimeError(
                "Table embedding count does not "
                "match extracted table count."
            )

        # Step 3: Build Qdrant-ready table chunks.
        result = []

        for idx, table in enumerate(all_tables):

            result.append({

                "id": (
                    f"{doc_name}_p{table['page_num']}"
                    f"_t{table['table_index']}"
                ),

                "text": table["text"],

                "embedding": embeddings[idx],

                "metadata": {
                    "doc_name": table["doc_name"],

                    "page_num": table["page_num"],

                    "type": "table",

                    "table_type": table["table_type"],

                    "table_index": table["table_index"],

                    "word_count": len(
                        table["text"].split()
                    )
                }
            })

        logger.info(
            f"Prepared {len(result)} table chunks for Qdrant."
        )

        return result


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    gen = EmbeddingGenerator()

    # Test ordinary text chunking.
    sample = (
        "TCS reported revenue of 2,25,458 crore rupees. "
        * 30
    )

    chunks = gen.chunk_text(sample)

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        f"First chunk: {chunks[0][:80]}..."
    )

    # This standalone test makes a real Gemini API call.
    print("\nGenerating embedding for first chunk...")

    emb = gen.generate_embedding(chunks[0])

    print(f"Dimensions: {len(emb)}")

    print("Embedding test successful.")
