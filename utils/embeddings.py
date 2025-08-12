"""
Utilities to compute OpenAI embeddings.
"""

from typing import List
import openai
from config import OPENAI_API_KEY, EMBEDDING_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using OpenAI embeddings API.

    Uses the legacy openai>=0.27 client interface to keep compatibility with the
    current project dependencies.
    """
    if not texts:
        return []

    openai.api_key = OPENAI_API_KEY

    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # Returned order matches inputs
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings


def embed_query(text: str) -> List[float]:
    """Return a single embedding vector for a query string."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


