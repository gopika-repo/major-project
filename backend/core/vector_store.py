import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Handles storing and searching vectors in Qdrant.
    Qdrant runs locally on your machine — no cloud needed.
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION
        self.client = None
        self.connect()

    def connect(self):
        """
        Connects to local Qdrant database.
        Creates the storage folder if it doesn't exist.
        """
        try:
            os.makedirs(settings.QDRANT_PATH, exist_ok=True)

            self.client = QdrantClient(path=settings.QDRANT_PATH)
            logger.info(f"Connected to Qdrant at {settings.QDRANT_PATH} ✅")

            self._ensure_collection()

        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise

    def _ensure_collection(self):
        """
        Creates the vector collection if it doesn't exist.
        A collection is like a table in a database.
        """
        existing = [
            c.name
            for c in self.client.get_collections().collections
        ]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBED_DIMENSIONS,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def upsert_chunks(self, chunks):
        """
        Stores a list of chunks into Qdrant.
        Each chunk has: id, text, embedding, metadata.
        Returns number of vectors stored.
        """
        try:
            points = []

            for i, chunk in enumerate(chunks):
                point = PointStruct(
                    id=i,
                    vector=chunk["embedding"],
                    payload={
                        **chunk["metadata"],
                        "text": chunk["text"],
                        "chunk_id": chunk["id"]
                    }
                )
                points.append(point)

            # Store in batches of 50
            batch_size = 50
            total = 0

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total += len(batch)
                logger.info(f"Stored {total}/{len(points)} vectors")

            return total

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    def query(self, embedding, top_k=5, filter_dict=None):
        """
        Searches Qdrant for the most similar vectors.
        Returns top_k most relevant chunks.
        """
        try:
            # Build filter if doc_name provided
            qdrant_filter = None

            if filter_dict and "doc_name" in filter_dict:
                val = filter_dict["doc_name"]
                if isinstance(val, dict):
                    doc_name = val.get("$eq")
                else:
                    doc_name = val

                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="doc_name",
                            match=MatchValue(value=doc_name)
                        )
                    ]
                )

            # query_points is the new method in latest qdrant-client
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )

            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": round(hit.score, 4),
                    "metadata": {
                        k: v
                        for k, v in hit.payload.items()
                        if k != "text"
                    }
                }
                for hit in results.points
            ]

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def delete_document(self, doc_name):
        """
        Deletes all vectors belonging to one document.
        Used when re-uploading the same document.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_name",
                            match=MatchValue(value=doc_name)
                        )
                    ]
                )
            )
            logger.info(f"Deleted vectors for: {doc_name}")

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise

    def get_stats(self):
        """Returns basic stats about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": info.points_count,
                "collection_name": self.collection_name,
                "dimensions": settings.EMBED_DIMENSIONS
            }
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {
                "total_vectors": 0,
                "collection_name": self.collection_name,
                "dimensions": settings.EMBED_DIMENSIONS
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random

    print("=== Qdrant VectorStore Test ===\n")
    vs = VectorStore()

    stats = vs.get_stats()
    print(f"Collection : {stats['collection_name']}")
    print(f"Dimensions : {stats['dimensions']}")
    print(f"Vectors    : {stats['total_vectors']}")

    # Test with fake vector of correct dimension
    fake_vector = [random.uniform(-1, 1) for _ in range(3072)]

    fake_chunk = [{
        "id": "test_p1_c0",
        "text": "TCS revenue was 2,25,458 crore in FY2023.",
        "embedding": fake_vector,
        "metadata": {
            "doc_name": "test_doc",
            "page_num": 1,
            "section": "FINANCIAL HIGHLIGHTS",
            "chunk_index": 0,
            "word_count": 9
        }
    }]

    count = vs.upsert_chunks(fake_chunk)
    print(f"\nUpserted {count} test vector ✅")

    results = vs.query(
        fake_vector,
        top_k=1,
        filter_dict={"doc_name": {"$eq": "test_doc"}}
    )

    print(f"Query returned {len(results)} result")
    print(f"Score : {results[0]['score']}")
    print(f"Text  : {results[0]['text']}")

    vs.delete_document("test_doc")
    print("\nTest vector deleted ✅")
    print("\nQdrant VectorStore working ✅")