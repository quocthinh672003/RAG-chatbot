"""
Simple Haystack RAG Pipeline
Theo documentation chính thức của Haystack
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    # Import Haystack components theo documentation
    from haystack import Pipeline
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate
    from haystack.schema import Document
    
    HAYSTACK_AVAILABLE = True
    logger.info("✅ Simple Haystack pipeline enabled")
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"❌ Simple Haystack pipeline failed: {e}")

from config import config


class SimpleHaystackPipeline:
    """
    Simple Haystack RAG Pipeline
    Theo documentation chính thức
    """
    
    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack not available")
        
        self._init_haystack()
        logger.info("🎯 Simple Haystack pipeline initialized")
    
    def _init_haystack(self):
        """Initialize Haystack theo documentation"""
        try:
            # Document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension, 
                similarity="cosine"
            )

            # Retriever
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=config.models.embedding_model,
                model_format="openai",
                api_key=config.openai_api_key,
                top_k=config.processing.top_k,
            )

            # Prompt template - Fixed syntax
            self.prompt_template = PromptTemplate(
                prompt="""
                Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
                
                Ngữ cảnh: {join(documents)}
                Câu hỏi: {query}
                
                Trả lời bằng tiếng Việt, mạch lạc và chính xác. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trong tài liệu được cung cấp."
                """,
                output_parser=None
            )

            # LLM
            self.llm = PromptNode(
                model_name_or_path=config.models.llm_model,
                api_key=config.openai_api_key,
                default_prompt_template=self.prompt_template,
                model_kwargs={"temperature": 0.1}
            )

            # Pipeline đơn giản
            self.pipeline = Pipeline()
            self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            self.pipeline.add_node(component=self.llm, name="LLM", inputs=["Retriever"])
            
            logger.info("✅ Simple Haystack components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Simple Haystack: {e}")
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
                
                # Tạo Document đơn giản, không dùng pandas metadata
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
                
                # Update embeddings
                self.retriever.update_embeddings(haystack_docs)
                
                logger.info(f"✅ Added {len(haystack_docs)} documents to Simple Haystack")
            except Exception as e:
                logger.error(f"❌ Error writing documents to store: {e}")
                raise
        else:
            logger.warning("⚠️ No valid documents to add")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query đơn giản"""
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Simple Haystack",
            }
        
        try:
            result = self.pipeline.run(query=query)
            
            return {
                "answer": result["answers"][0].answer if result["answers"] else "Không tìm thấy câu trả lời.",
                "documents": result["documents"],
                "sources": [doc.meta.get("source_name", "Unknown") for doc in result["documents"]],
                "pipeline": "Simple Haystack",
            }
        except Exception as e:
            logger.error(f"❌ Error in query: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Simple Haystack (Error)",
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
            "pipeline_type": "Simple Haystack RAG Pipeline",
            "active_pipeline": "Haystack",
            "components": ["InMemoryDocumentStore", "EmbeddingRetriever", "PromptNode"],
            "features": ["Simple RAG", "OpenAI Embeddings", "OpenAI LLM"],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available"
        }
    
    def clear_documents(self) -> None:
        """Clear all documents from the store"""
        try:
            # Reinitialize document store to clear all documents
            self._init_haystack()
            logger.info("✅ Cleared all documents from Simple Haystack")
        except Exception as e:
            logger.error(f"❌ Error clearing documents: {e}")


def get_simple_haystack_pipeline():
    """Get Simple Haystack pipeline instance"""
    if HAYSTACK_AVAILABLE:
        return SimpleHaystackPipeline()
    else:
        raise ImportError("Haystack not available for Simple Haystack pipeline")
