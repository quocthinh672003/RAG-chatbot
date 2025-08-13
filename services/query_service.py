"""
Hybrid Query Service: Haystack + LangChain fallback
"""

from typing import List, Dict, Any
from services.hybrid_rag_pipeline import rag_pipeline


class QueryService:
    """Query service with hybrid approach"""

    def __init__(self):
        self.pipeline = rag_pipeline

    def query(self, question: str) -> Dict[str, Any]:
        """Query the hybrid RAG pipeline"""
        try:
            # Use hybrid RAG pipeline
            result = self.pipeline.query(question)

            return {
                "answer": result["answer"],
                "source_documents": result["documents"],
                "sources": result["sources"],
                "pipeline_used": result.get("pipeline", "Unknown")
            }

        except Exception as e:
            print(f"❌ Error in query service: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "source_documents": [],
                "sources": [],
                "pipeline_used": "Error"
            }

    def get_document_count(self) -> int:
        """Get number of documents in pipeline"""
        return self.pipeline.get_document_count()

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return self.pipeline.get_pipeline_info()


# Global query service instance
query_service = QueryService()
