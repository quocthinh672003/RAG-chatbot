"""
Qdrant helpers using Haystack QdrantDocumentStore for storage only.
Embeddings are computed via OpenAI in utils.embeddings.
"""

from haystack.document_stores import QdrantDocumentStore
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


def get_qdrant_document_store(recreate: bool = False) -> QdrantDocumentStore:
    """Create or connect to QdrantDocumentStore."""
    ds = QdrantDocumentStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=QDRANT_COLLECTION_NAME,
        prefer_grpc=False,
        embedding_dim=EMBEDDING_DIMENSION,
        similarity="cosine",
        index="hnsw",
    )
    if recreate:
        try:
            ds.delete_index(QDRANT_COLLECTION_NAME)
        except Exception:
            pass
    return ds

    