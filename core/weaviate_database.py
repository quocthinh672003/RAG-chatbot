"""
Weaviate Cloud Document Store
Thay thế Qdrant cho tương thích tốt hơn với Haystack
"""

import os
import logging
from dotenv import load_dotenv
from config import config

# Load environment variables
load_dotenv()

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

        # Remove any trailing slashes and ensure proper URL format
        if WEAVIATE_URL.endswith("/"):
            WEAVIATE_URL = WEAVIATE_URL[:-1]

        # For Weaviate Cloud, we need to use the full URL without port
        # Extract host from URL
        from urllib.parse import urlparse

        parsed_url = urlparse(WEAVIATE_URL)
        host = parsed_url.netloc

        # Create Weaviate Document Store without port (let Haystack handle it)
        document_store = WeaviateDocumentStore(
            host=host,
            api_key=WEAVIATE_API_KEY,
            index="RAG-chatbot-docs",
            similarity="cosine",
            embedding_dim=config.models.embedding_dimension,
            timeout_config=(60, 120),  # (connect_timeout, read_timeout)
            additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
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
            embedding_dim=config.models.embedding_dimension,
            similarity="cosine",
            use_bm25=True,
        )
