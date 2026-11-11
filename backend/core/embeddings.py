import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from config import settings

logger = logging.getLogger(__name__)

# Create one Gemini client — used for everything
client = genai.Client(api_key=settings.GEMINI_API_KEY)


class EmbeddingGenerator:
    """
    Converts text into vectors (numbers) using Gemini.
    These vectors are what gets stored in Qdrant.
    """

    def chunk_text(self, text):
        """
        Splits a long text into smaller overlapping chunks.
        Why? Because AI models can only handle limited text at once.
        Each chunk = 400 words with 50 word overlap.
        """
        words = text.split()

        # If text is too short, return as single chunk
        if len(words) < 15:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(words):
            end = start + settings.CHUNK_SIZE
            chunk = " ".join(words[start:end])

            # Only keep chunks with at least 15 words
            if len(chunk.split()) >= 15:
                chunks.append(chunk)

            # Move forward with overlap
            start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP

        return chunks

    def generate_embedding(self, text):
        """
        Converts one piece of text into a vector.
        Returns a list of 3072 numbers.
        """
        try:
            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )
            return result.embeddings[0].values

        except Exception as e:
            logger.warning(f"Embedding failed, retrying in 5s: {e}")
            time.sleep(5)

            # Try one more time
            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )
            return result.embeddings[0].values

    def generate_query_embedding(self, text):
        """
        Same as generate_embedding but used for
        questions/queries from the user.
        """
        try:
            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=text
            )
            return result.embeddings[0].values

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise

    def batch_embed(self, texts):
        """
        Embeds a list of texts one by one.
        Handles rate limits with smart waiting.
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            print(f"  Embedding {i+1}/{total}...", end="\r")

            # Try with retry on rate limit
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
                    break

                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait = (attempt + 1) * 30
                        print(f"\n  ⏳ Rate limit hit. Waiting {wait}s "
                              f"(attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        raise

            # Pause every 5 embeddings to avoid rate limit
            if (i + 1) % 5 == 0:
                time.sleep(2)

        print(f"  Embedded {total}/{total} chunks ✅")
        return embeddings

    def prepare_chunks_for_storage(self, pages, doc_name):
        """
        Takes list of pages from TextExtractor.
        Chunks each page, embeds all chunks.
        Returns list of dicts ready to store in Qdrant.
        """
        all_chunks = []

        # Step 1 — chunk every page
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

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # Step 2 — embed all chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = self.batch_embed(texts)

        # Step 3 — combine chunk info with embedding
        result = []
        for i, chunk in enumerate(all_chunks):
            result.append({
                "id": f"{doc_name}_p{chunk['page_num']}_c{chunk['chunk_index']}",
                "text": chunk["text"],
                "embedding": embeddings[i],
                "metadata": {
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "word_count": len(chunk["text"].split())
                }
            })

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = EmbeddingGenerator()

    # Test chunking
    sample = "TCS reported revenue of 2,25,458 crore rupees. " * 30
    chunks = gen.chunk_text(sample)
    print(f"Chunks created : {len(chunks)}")
    print(f"First chunk    : {chunks[0][:80]}...")

    # Test embedding
    print("\nGenerating embedding for first chunk...")
    emb = gen.generate_embedding(chunks[0])
    print(f"Dimensions : {len(emb)}")
    print(f"First value: {emb[0]}")
    print("Embedding test ✅")