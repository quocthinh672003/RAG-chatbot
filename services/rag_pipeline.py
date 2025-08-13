"""
Haystack RAG Pipeline Service
"""

from typing import List, Dict, Any, Optional
from haystack import Document, Pipeline
from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, LostInTheMiddleRanker
from haystack.nodes import PromptNode, PromptTemplate
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
import os
from config import config


class HaystackRAGPipeline:
    """Haystack-based RAG Pipeline"""
    
    def __init__(self):
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
    
    def _build_pipeline(self) -> Pipeline:
        """Build Haystack RAG pipeline"""
        pipeline = Pipeline()
        
        # Add components
        pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        pipeline.add_node(component=self.similarity_ranker, name="SimilarityRanker", inputs=["Retriever"])
        pipeline.add_node(component=self.diversity_ranker, name="DiversityRanker", inputs=["SimilarityRanker"])
        pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["DiversityRanker"])
        
        return pipeline
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the pipeline"""
        # Add documents to document store
        self.document_store.write_documents(documents)
        
        # Update embeddings
        self.retriever.update_embeddings(documents)
        
        print(f"✅ Added {len(documents)} documents to Haystack pipeline")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Run pipeline
            result = self.pipeline.run(query=query)
            
            return {
                "answer": result["answers"][0].answer if result["answers"] else "Không tìm thấy câu trả lời.",
                "documents": result["documents"],
                "sources": [doc.meta.get("source_name", "Unknown") for doc in result["documents"]]
            }
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": []
            }
    
    def get_document_count(self) -> int:
        """Get number of documents in store"""
        return self.document_store.get_document_count()


# Global pipeline instance
rag_pipeline = HaystackRAGPipeline()
