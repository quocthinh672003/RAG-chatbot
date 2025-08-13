"""
Hybrid Document Ingestion Service: Haystack + LangChain fallback
"""

import uuid
from datetime import datetime
from typing import List, Union

from services.document_service import DocumentService
from services.hybrid_rag_pipeline import rag_pipeline


class IngestionService:
    """Document ingestion service with hybrid approach"""

    def __init__(self):
        self.document_service = DocumentService()

    def ingest_document(self, file_path: str) -> str:
        """Ingest a single document"""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Convert and process document using hybrid service
            documents = self.document_service.convert_file(file_path)

            if not documents:
                print(f"⚠️ No documents extracted from {file_path}")
                return None

            # Add document ID and timestamp
            for doc in documents:
                if hasattr(doc, 'meta'):  # Haystack Document
                    doc.meta.update({
                        "document_id": doc_id,
                        "ingestion_timestamp": datetime.now().isoformat(),
                    })
                else:  # LangChain Document
                    doc.metadata.update({
                        "document_id": doc_id,
                        "ingestion_timestamp": datetime.now().isoformat(),
                    })

            # Add to hybrid RAG pipeline
            rag_pipeline.add_documents(documents)

            print(f"✅ Successfully ingested document: {file_path}")
            return doc_id

        except Exception as e:
            print(f"❌ Error ingesting document {file_path}: {e}")
            return None

    def ingest_documents(self, file_paths: List[str]) -> List[str]:
        """Ingest multiple documents"""
        doc_ids = []
        for file_path in file_paths:
            doc_id = self.ingest_document(file_path)
            if doc_id:
                doc_ids.append(doc_id)
        return doc_ids

    def get_document_count(self) -> int:
        """Get total number of documents in pipeline"""
        return rag_pipeline.get_document_count()

    def get_service_info(self) -> dict:
        """Get service information"""
        return {
            "document_service": self.document_service.get_processor_info(),
            "rag_pipeline": rag_pipeline.get_pipeline_info(),
            "document_count": self.get_document_count()
        }


# Global ingestion service instance
ingestion_service = IngestionService()
