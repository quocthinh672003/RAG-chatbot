"""
Haystack-only RAG Pipeline
Sử dụng Haystack components thay thế pandas
"""

import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Set environment variables for Haystack
os.environ['PYDANTIC_ARBITRARY_TYPES_ALLOWED'] = 'true'
os.environ['PYDANTIC_IGNORE_UNKNOWN'] = 'true'

try:
    # Import Haystack components
    from haystack import Pipeline
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate
    from haystack.schema import Document
    from haystack.nodes.file_converter import TextConverter
    from haystack.nodes.preprocessor import PreProcessor
    from haystack.nodes.retriever import BM25Retriever
    
    HAYSTACK_AVAILABLE = True
    logger.info("✅ Haystack-only pipeline enabled")
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"❌ Haystack-only pipeline failed: {e}")

from config import config


class HaystackOnlyPipeline:
    """
    Haystack-only RAG Pipeline
    Sử dụng Haystack components thay thế pandas
    """
    
    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack not available")
        
        self._init_haystack()
        logger.info("🎯 Haystack-only pipeline initialized")
    
    def _init_haystack(self):
        """Initialize Haystack components"""
        try:
            # Document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension, 
                similarity="cosine"
            )

            # Text converter (thay thế pandas cho file processing)
            self.text_converter = TextConverter(
                remove_numeric_tables=True,
                valid_languages=["vi", "en"]
            )

            # Preprocessor (thay thế pandas cho data cleaning)
            self.preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                clean_header_footer=True,
                split_by="word",
                split_length=config.processing.chunk_size,
                split_overlap=config.processing.chunk_overlap
            )

            # Embedding Retriever
            self.embedding_retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=config.models.embedding_model,
                model_format="openai",
                api_key=config.openai_api_key,
                top_k=config.processing.top_k,
            )

            # BM25 Retriever (hybrid search)
            self.bm25_retriever = BM25Retriever(
                document_store=self.document_store,
                top_k=config.processing.top_k
            )

            # Prompt template
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

            # Pipeline - Fixed structure
            self.pipeline = Pipeline()
            self.pipeline.add_node(component=self.embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
            self.pipeline.add_node(component=self.bm25_retriever, name="BM25Retriever", inputs=["Query"])
            self.pipeline.add_node(component=self.llm, name="LLM", inputs=["EmbeddingRetriever", "BM25Retriever"])
            
            logger.info("✅ Haystack-only components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Haystack-only: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to Haystack using Haystack components"""
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
                
                # Tạo Haystack Document không dùng pandas
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
                # Preprocess documents
                processed_docs = self.preprocessor.process(haystack_docs)
                
                # Add to document store
                self.document_store.write_documents(processed_docs)
                
                # Update embeddings
                self.embedding_retriever.update_embeddings(processed_docs)
                
                logger.info(f"✅ Added {len(documents)} documents to Haystack (processed: {len(processed_docs)})")
            except Exception as e:
                logger.error(f"❌ Error writing documents to store: {e}")
                raise
        else:
            logger.warning("⚠️ No valid documents to add")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query Haystack pipeline with hybrid search"""
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack-only (Hybrid Search)",
            }
        
        try:
            result = self.pipeline.run(query=query)
            
            return {
                "answer": result["answers"][0].answer if result["answers"] else "Không tìm thấy câu trả lời.",
                "documents": result["documents"],
                "sources": [doc.meta.get("source_name", "Unknown") for doc in result["documents"]],
                "pipeline": "Haystack-only (Hybrid Search)",
            }
        except Exception as e:
            logger.error(f"❌ Error in query: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack-only (Error)",
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
            "pipeline_type": "Haystack-only RAG Pipeline",
            "active_pipeline": "Haystack",
            "components": [
                "InMemoryDocumentStore", 
                "TextConverter", 
                "PreProcessor",
                "EmbeddingRetriever", 
                "BM25Retriever",
                "PromptNode"
            ],
            "features": [
                "Hybrid Search (Embedding + BM25)",
                "Text Preprocessing",
                "No Pandas Dependency"
            ],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available"
        }
    
    def clear_documents(self) -> None:
        """Clear all documents from the store"""
        try:
            # Reinitialize document store to clear all documents
            self._init_haystack()
            logger.info("✅ Cleared all documents from Haystack-only")
        except Exception as e:
            logger.error(f"❌ Error clearing documents: {e}")


def get_haystack_only_pipeline():
    """Get Haystack-only pipeline instance"""
    if HAYSTACK_AVAILABLE:
        return HaystackOnlyPipeline()
    else:
        raise ImportError("Haystack not available for Haystack-only pipeline")
