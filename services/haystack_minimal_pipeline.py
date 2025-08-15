"""
Minimal Haystack RAG Pipeline
Chỉ dùng components cơ bản, không transformers
"""

import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)

# Try to import only basic Haystack components
HAYSTACK_AVAILABLE = False
try:
    # Import only the most basic components
    from haystack.schema import Document
    from haystack.document_stores import InMemoryDocumentStore
    
    HAYSTACK_AVAILABLE = True
    logger.info("✅ Minimal Haystack pipeline enabled (basic components only)")
except Exception as e:
    logger.error(f"❌ Minimal Haystack pipeline failed: {e}")

from config import config


class MinimalHaystackPipeline:
    """
    Minimal Haystack RAG Pipeline
    Chỉ dùng components cơ bản, không transformers
    """
    
    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack not available")
        
        self._init_haystack()
        logger.info("🎯 Minimal Haystack pipeline initialized")
    
    def _init_haystack(self):
        """Initialize Haystack với components tối thiểu"""
        try:
            # Document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension, 
                similarity="cosine"
            )
            
            logger.info("✅ Haystack components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Haystack: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents đơn giản"""
        if not documents:
            logger.warning("⚠️ No documents to add")
            return
            
        haystack_docs = []
        
        for doc in documents:
            try:
                # Validate document structure
                if not isinstance(doc, dict) or "page_content" not in doc:
                    logger.warning(f"⚠️ Invalid document structure: {doc}")
                    continue
                
                # Tạo Document đơn giản
                haystack_doc = Document(
                    content=doc["page_content"],
                    meta={
                        "source_name": doc.get("metadata", {}).get("source_name", "Unknown"),
                        "page": doc.get("metadata", {}).get("page", 0),
                        "file_type": doc.get("metadata", {}).get("file_type", "unknown"),
                        "language": doc.get("metadata", {}).get("language", "vi")
                    }
                )
                haystack_docs.append(haystack_doc)
            except Exception as e:
                logger.error(f"❌ Error processing document: {e}")
                continue
        
        if haystack_docs:
            try:
                # Add to document store
                self.document_store.write_documents(haystack_docs)
                logger.info(f"✅ Added {len(haystack_docs)} documents to Minimal Haystack")
            except Exception as e:
                logger.error(f"❌ Error writing documents to store: {e}")
                raise
        else:
            logger.warning("⚠️ No valid documents to add")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query đơn giản - chỉ trả về documents"""
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Minimal Haystack (Basic Search)",
            }
        
        try:
            # Lấy tất cả documents
            docs = self.document_store.get_all_documents()
            
            if not docs:
                return {
                    "answer": "Chưa có tài liệu nào được tải lên.",
                    "documents": [],
                    "sources": [],
                    "pipeline": "Minimal Haystack (Basic Search)",
                }
            
            # Tìm documents có chứa query (case-insensitive)
            query_lower = query.lower()
            relevant_docs = []
            
            for doc in docs:
                if query_lower in doc.content.lower():
                    relevant_docs.append(doc)
            
            # Nếu không tìm thấy exact match, tìm partial matches
            if not relevant_docs:
                query_words = query_lower.split()
                for doc in docs:
                    doc_lower = doc.content.lower()
                    # Kiểm tra xem có ít nhất 2 từ trong query xuất hiện trong document
                    matches = sum(1 for word in query_words if word in doc_lower)
                    if matches >= max(1, len(query_words) // 2):
                        relevant_docs.append(doc)
            
            # Trả về kết quả đơn giản
            if relevant_docs:
                # Tạo answer từ document đầu tiên
                first_doc = relevant_docs[0]
                content_preview = first_doc.content[:300] + "..." if len(first_doc.content) > 300 else first_doc.content
                
                answer = f"Tìm thấy {len(relevant_docs)} tài liệu liên quan. Nội dung: {content_preview}"
            else:
                answer = "Không tìm thấy thông tin trong tài liệu được cung cấp."
            
            return {
                "answer": answer,
                "documents": relevant_docs,
                "sources": [doc.meta.get("source_name", "Unknown") for doc in relevant_docs],
                "pipeline": "Minimal Haystack (Basic Search)",
            }
            
        except Exception as e:
            logger.error(f"❌ Error in query: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Minimal Haystack (Error)",
            }
    
    def get_document_count(self) -> int:
        """Get document count"""
        try:
            return self.document_store.get_document_count()
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Minimal Haystack RAG Pipeline",
            "active_pipeline": "Haystack",
            "components": ["InMemoryDocumentStore"],
            "features": ["Basic Text Search", "No Transformers", "No Complex Dependencies"],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available"
        }
    
    def clear_documents(self) -> None:
        """Clear all documents from the store"""
        try:
            # Reinitialize document store to clear all documents
            self._init_haystack()
            logger.info("✅ Cleared all documents from Minimal Haystack")
        except Exception as e:
            logger.error(f"❌ Error clearing documents: {e}")


def get_minimal_haystack_pipeline():
    """Get Minimal Haystack pipeline instance"""
    if HAYSTACK_AVAILABLE:
        return MinimalHaystackPipeline()
    else:
        raise ImportError("Haystack not available for Minimal Haystack pipeline")
