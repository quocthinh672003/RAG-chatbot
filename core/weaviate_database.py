"""
Weaviate Cloud Document Store
Thay thế Qdrant cho tương thích tốt hơn với Haystack
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_weaviate_document_store():
    """Get Weaviate Cloud Document Store"""
    try:
        from haystack.document_stores import WeaviateDocumentStore

        # Weaviate Cloud credentials
        WEAVIATE_URL = os.getenv(
            "WEAVIATE_URL", "https://your-cluster.weaviate.network"
        )
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "your-api-key")

        # Create Weaviate Document Store
        document_store = WeaviateDocumentStore(
            host=WEAVIATE_URL,
            api_key=WEAVIATE_API_KEY,
            index="RAG-chatbot-docs",
            similarity="cosine",
            embedding_dim=768,
            timeout_config=(30, 60),  # (connect_timeout, read_timeout)
        )

        logger.info("✅ Weaviate Cloud Document Store initialized")
        return document_store

    except Exception as e:
        logger.error(f"❌ Failed to initialize Weaviate: {e}")
        raise


def get_document_store():
    """Get document store (Weaviate preferred)"""
    try:
        return get_weaviate_document_store()
    except Exception as e:
        logger.error(f"❌ Weaviate failed: {e}")
        # Fallback to InMemory if needed
        from haystack.document_stores import InMemoryDocumentStore

        logger.warning("⚠️ Falling back to InMemoryDocumentStore")
        return InMemoryDocumentStore(
            embedding_dim=768, similarity="cosine", use_bm25=True
        )
