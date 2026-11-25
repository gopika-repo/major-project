
import logging
import sys
import os

from uuid import NAMESPACE_URL, uuid5

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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

    Supports text and financial table chunks.
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION
        self.client = None
        self.connect()

    def connect(self):
        """Connect to the local Qdrant database."""

        try:
            os.makedirs(settings.QDRANT_PATH, exist_ok=True)

            self.client = QdrantClient(
                path=settings.QDRANT_PATH
            )

            logger.info(
                f"Connected to Qdrant at {settings.QDRANT_PATH}"
            )

            self._ensure_collection()

        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise

    def _ensure_collection(self):
        """Create the collection if it does not exist."""

        existing = [
            collection.name
            for collection in self.client.get_collections().collections
        ]

        if self.collection_name not in existing:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBED_DIMENSIONS,
                    distance=Distance.COSINE
                )
            )

            logger.info(
                f"Created collection: {self.collection_name}"
            )

        else:
            logger.info(
                f"Using existing collection: {self.collection_name}"
            )

    @staticmethod
    def _make_point_id(chunk):
        """
        Generate a stable, unique UUID for each chunk.

        Different documents and chunk types get different IDs.
        """

        metadata = chunk.get("metadata", {})

        doc_name = metadata.get("doc_name")
        chunk_key = chunk.get("id")
        chunk_type = metadata.get("type") or "text"

        if not doc_name:
            raise ValueError(
                "Chunk metadata must contain doc_name"
            )

        if not chunk_key:
            raise ValueError(
                "Every chunk must contain a unique id"
            )

        unique_key = (
            f"visual-financial-analyst:"
            f"{doc_name}:"
            f"{chunk_type}:"
            f"{chunk_key}"
        )

        return str(
            uuid5(NAMESPACE_URL, unique_key)
        )

    def upsert_chunks(self, chunks):
        """Store chunks and their embeddings in Qdrant."""

        try:
            if not chunks:
                logger.warning(
                    "No chunks received for storage."
                )
                return 0

            points = []

            for chunk in chunks:

                point_id = self._make_point_id(chunk)

                point = PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        **chunk["metadata"],
                        "text": chunk["text"],
                        "chunk_id": chunk["id"]
                    }
                )

                points.append(point)

            batch_size = 50
            total = 0

            for i in range(0, len(points), batch_size):

                batch = points[i:i + batch_size]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

                total += len(batch)

                logger.info(
                    f"Stored {total}/{len(points)} vectors"
                )

            return total

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    @staticmethod
    def _build_filter(filter_dict=None):
        """
        Build a Qdrant filter from equality and inequality conditions.

        Supported examples:

        {"doc_name": {"$eq": "infosys-ar-24"}}

        {"type": {"$eq": "table"}}

        {"type": {"$ne": "table"}}

        Multiple conditions are combined.
        """

        if not filter_dict:
            return None

        must_conditions = []
        must_not_conditions = []

        for field, condition in filter_dict.items():

            if isinstance(condition, dict):

                if "$eq" in condition:

                    value = condition["$eq"]

                    must_conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )

                elif "$ne" in condition:

                    value = condition["$ne"]

                    must_not_conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )

                else:
                    raise ValueError(
                        f"Unsupported filter operator for {field}. "
                        "Use $eq or $ne."
                    )

            else:

                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=condition)
                    )
                )

        if not must_conditions and not must_not_conditions:
            return None

        return Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None
        )

    def query(self, embedding, top_k=5, filter_dict=None):
        """
        Search Qdrant for the most similar vectors.

        Supports filtering by document name, chunk type,
        and other payload fields.
        """

        try:

            qdrant_filter = self._build_filter(
                filter_dict
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )

            return [
                {
                    "text": (hit.payload or {}).get("text", ""),
                    "score": round(hit.score, 4),
                    "metadata": {
                        key: value
                        for key, value in (hit.payload or {}).items()
                        if key != "text"
                    }
                }
                for hit in results.points
            ]

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def delete_document(self, doc_name):
        """Delete all vectors belonging to one document."""

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

            logger.info(
                f"Deleted vectors for: {doc_name}"
            )

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise

    def get_stats(self):
        """Return basic Qdrant collection statistics."""

        try:

            info = self.client.get_collection(
                self.collection_name
            )

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

    try:

        stats = vs.get_stats()

        print(f"Collection : {stats['collection_name']}")
        print(f"Dimensions : {stats['dimensions']}")
        print(f"Vectors    : {stats['total_vectors']}")

        fake_vector = [
            random.uniform(-1, 1)
            for _ in range(settings.EMBED_DIMENSIONS)
        ]

        fake_chunk = [{
            "id": "test_p1_c0",
            "text": "Test financial document content.",
            "embedding": fake_vector,
            "metadata": {
                "doc_name": "test_doc",
                "page_num": 1,
                "section": "TEST",
                "chunk_index": 0,
                "word_count": 5
            }
        }]

        count = vs.upsert_chunks(fake_chunk)

        print(f"\nUpserted {count} test vector")

        results = vs.query(
            fake_vector,
            top_k=1,
            filter_dict={
                "doc_name": {
                    "$eq": "test_doc"
                }
            }
        )

        print(f"Query returned {len(results)} result")

        if results:
            print(f"Score : {results[0]['score']}")
            print(f"Text  : {results[0]['text']}")

        vs.delete_document("test_doc")

        print("\nTest vector deleted")
        print("VectorStore test completed")

    finally:
        if vs.client is not None:
            vs.client.close()
