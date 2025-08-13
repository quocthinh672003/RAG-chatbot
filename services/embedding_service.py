"""
Embedding service
"""

from typing import List
from openai import OpenAI
from config import config


class EmbeddingService:
    """OpenAI embedding service"""

    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        try:
            response = self.client.embeddings.create(
                model=config.models.embedding_model,
                input=texts,
                encoding_format="float",
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[0.0] * 1536] * len(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generate query embedding"""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else [0.0] * 1536
