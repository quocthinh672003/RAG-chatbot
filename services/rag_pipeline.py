"""
Haystack RAG Pipeline Service
"""

import logging
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

# Set environment variables for Haystack
os.environ['PYDANTIC_ARBITRARY_TYPES_ALLOWED'] = 'true'
os.environ['PYDANTIC_IGNORE_UNKNOWN'] = 'true'

try:
    from haystack import Document, Pipeline
    from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, LostInTheMiddleRanker
    from haystack.nodes import PromptNode, PromptTemplate
    from haystack.document_stores import InMemoryDocumentStore
    
    HAYSTACK_AVAILABLE = True
    logger.info("✅ Haystack RAG pipeline enabled")
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"❌ Haystack RAG pipeline failed: {e}")

from config import config


class HaystackRAGPipeline:
    """Haystack-based RAG Pipeline"""
    
    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack not available")
        
        self._init_haystack()
        logger.info("🎯 Haystack RAG pipeline initialized")
    
    def _init_haystack(self):
        """Initialize Haystack components"""
        try:
            # Initialize document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension,
                similarity="cosine"
            )
            
            # Initialize retriever
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=config.models.embedding_model,
                model_format="openai",
                api_key=config.openai_api_key,
                top_k=config.processing.top_k
            )
            
            # Rankers for better document selection
            self.similarity_ranker = SentenceTransformersRanker(
                model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            
            self.diversity_ranker = LostInTheMiddleRanker()
            
            # Prompt template
            self.prompt_template = PromptTemplate(
                prompt="""
                Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
                YÊU CẦU NGHIÊM NGẶT: Không suy đoán, không dùng kiến thức ngoài ngữ cảnh.
                
                Hướng dẫn trả lời (bằng tiếng Việt):
                - Nếu có số liệu/bảng: trích đúng số, kèm đơn vị (ví dụ: TWh, %, tỷ lệ). Trả lời đầy đủ, KHÔNG tóm tắt.
                - Ưu tiên số liệu đúng NĂM/ĐỊA ĐIỂM được hỏi; nếu nhiều mục (ví dụ theo ngành), liệt kê rõ ràng.
                - Nếu có bảng phù hợp: xuất lại bảng Markdown đầy đủ từ dữ liệu trong ngữ cảnh (không lược bớt cột chính).
                - Nếu ngữ cảnh không đủ thông tin: trả lời đúng câu sau: 'Không tìm thấy thông tin trong tài liệu đã cung cấp.'
                
                [Ngữ cảnh]:
                {join(documents)}
                
                [Câu hỏi]: {query}
                
                Xuất trả lời ở dạng Markdown, có thể bao gồm bảng, bullet. ĐẦY ĐỦ theo ngữ cảnh.
                """,
                output_parser=None
            )
            
            # LLM for generation
            self.prompt_node = PromptNode(
                model_name_or_path=config.models.llm_model,
                api_key=config.openai_api_key,
                default_prompt_template=self.prompt_template,
                model_kwargs={
                    "temperature": 0.2,
                    "max_tokens": 1200
                }
            )
            
            # Build pipeline
            self.pipeline = self._build_pipeline()
            
            logger.info("✅ Haystack RAG components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Haystack RAG: {e}")
            raise
    
    def _build_pipeline(self) -> Pipeline:
        """Build Haystack RAG pipeline"""
        try:
            pipeline = Pipeline()
            
            # Add components
            pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            pipeline.add_node(component=self.similarity_ranker, name="SimilarityRanker", inputs=["Retriever"])
            pipeline.add_node(component=self.diversity_ranker, name="DiversityRanker", inputs=["SimilarityRanker"])
            pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["DiversityRanker"])
            
            return pipeline
        except Exception as e:
            logger.error(f"❌ Failed to build pipeline: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the pipeline"""
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
                
                # Convert to Haystack Document
                if isinstance(doc, dict):
                    haystack_doc = Document(
                        content=doc["page_content"],
                        meta={
                            "source_name": doc.get("metadata", {}).get("source_name", "Unknown"),
                            "page": doc.get("metadata", {}).get("page", 0),
                            "file_type": doc.get("metadata", {}).get("file_type", "unknown"),
                            "language": doc.get("metadata", {}).get("language", "vi")
                        }
                    )
                else:
                    # Assume it's already a Haystack Document
                    haystack_doc = doc
                
                haystack_docs.append(haystack_doc)
            except Exception as e:
                logger.error(f"❌ Error processing document: {e}")
                continue
        
        if haystack_docs:
            try:
                # Add documents to document store
                self.document_store.write_documents(haystack_docs)
                
                # Update embeddings
                self.retriever.update_embeddings(haystack_docs)
                
                logger.info(f"✅ Added {len(haystack_docs)} documents to Haystack RAG pipeline")
            except Exception as e:
                logger.error(f"❌ Error writing documents to store: {e}")
                raise
        else:
            logger.warning("⚠️ No valid documents to add")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack RAG",
            }
        
        try:
            # Run pipeline
            result = self.pipeline.run(query=query)
            
            return {
                "answer": result["answers"][0].answer if result["answers"] else "Không tìm thấy câu trả lời.",
                "documents": result["documents"],
                "sources": [doc.meta.get("source_name", "Unknown") for doc in result["documents"]],
                "pipeline": "Haystack RAG",
            }
            
        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack RAG (Error)",
            }
    
    def get_document_count(self) -> int:
        """Get number of documents in store"""
        try:
            return self.document_store.get_document_count()
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Haystack RAG Pipeline",
            "active_pipeline": "Haystack",
            "components": [
                "InMemoryDocumentStore", 
                "EmbeddingRetriever", 
                "SentenceTransformersRanker",
                "LostInTheMiddleRanker",
                "PromptNode"
            ],
            "features": [
                "Advanced RAG",
                "Document Ranking",
                "Diversity Ranking",
                "OpenAI Integration"
            ],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available"
        }
    
    def clear_documents(self) -> None:
        """Clear all documents from the store"""
        try:
            # Reinitialize document store to clear all documents
            self._init_haystack()
            logger.info("✅ Cleared all documents from Haystack RAG")
        except Exception as e:
            logger.error(f"❌ Error clearing documents: {e}")


# Global pipeline instance
rag_pipeline = HaystackRAGPipeline() if HAYSTACK_AVAILABLE else None
