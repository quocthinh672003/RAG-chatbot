"""
Database service
"""

from haystack.document_stores import QdrantDocumentStore
from config import config


def get_document_store():
    """Get Qdrant document store"""
    try:
        return QdrantDocumentStore(
            host=config.database.host,
            port=config.database.port,
            collection_name=config.database.collection_name,
            embedding_dim=config.models.embedding_dimension,
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        # Fallback to in-memory store
        from haystack.document_stores import InMemoryDocumentStore

        return InMemoryDocumentStore(embedding_dim=config.models.embedding_dimension)
