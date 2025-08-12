"""
Document ingestion
"""

import uuid
from datetime import datetime
from typing import List

from config import config
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from core.database import get_document_store


class IngestionService:
    """Document ingestion service"""

    def __init__(self):
        self.document_service = DocumentService()
        self.embedding_service = EmbeddingService()
        self.document_store = get_document_store()

    def ingest_document(self, file_path: str) -> str:
        """Ingest a single document"""
        # Generate document ID
        doc_id = str(uuid.uuid4())

        # Convert and process document
        documents = self.document_service.convert_file(file_path)

        # Add document ID and timestamp
        for doc in documents:
            doc.meta.update(
                {
                    "document_id": doc_id,
                    "ingestion_timestamp": datetime.now().isoformat(),
                }
            )

        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_service.embed_texts(contents)

        # Assign embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        # Store in document store
        self.document_store.write_documents(documents)

        return doc_id


# Global service instance
ingestion_service = IngestionService()


def ingest_document(file_path: str) -> str:
    """Convenience function"""
    return ingestion_service.ingest_document(file_path)
