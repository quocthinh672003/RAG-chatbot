"""
Query service
"""
from typing import List, Dict, Any, Optional
from haystack import Document
from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, LostInTheMiddleRanker
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import config
from services.embedding_service import EmbeddingService
from core.database import get_document_store

class QueryService:
    """Query service"""
    
    def __init__(self):
        self.document_store = get_document_store()
        self.embedding_service = EmbeddingService()
        
        # Initialize Haystack components
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=config.models.embedding_model,
            model_format="openai",
            top_k=config.processing.top_k
        )
        
        # Initialize rankers for better document ranking
        self.similarity_ranker = SentenceTransformersRanker(
            model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.diversity_ranker = LostInTheMiddleRanker()
        
        self.llm = ChatOpenAI(
            model_name=config.models.llm_model,
            openai_api_key=config.openai_api_key,
            temperature=0.1,
        )
        self.prompt_template = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Bạn là một trợ lý AI thông minh. Dựa trên các tài liệu sau đây, hãy trả lời câu hỏi một cách chính xác và đầy đủ.
            
            **Hướng dẫn:**
            - Chỉ trả lời dựa trên thông tin có trong tài liệu được cung cấp
            - Nếu không thể trả lời từ tài liệu, hãy nói rõ "Tôi không có đủ thông tin để trả lời câu hỏi này"
            - Trả lời bằng tiếng Việt, mạch lạc và có cấu trúc
            
            **Tài liệu tham khảo:**
            {context}
            
            **Câu hỏi:** {question}
            
            **Trả lời:**"""
        )
    
    def retrieve_documents(self, query: str, top_k: int = None) -> List[Document]:
        """Retrieve relevant documents using Haystack retriever and rankers"""
        if top_k is None:
            top_k = config.processing.top_k
        
        # Use Haystack retriever
        documents = self.retriever.retrieve(query=query, top_k=top_k)
        
        # Apply similarity ranking
        if documents:
            documents = self.similarity_ranker.predict(query=query, documents=documents)
        
        # Apply diversity ranking to avoid similar documents
        if documents:
            documents = self.diversity_ranker.predict(query=query, documents=documents)
        
        return documents
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LLM"""
        if not documents:
            return "Không tìm thấy tài liệu liên quan."
        
        # Create context from documents
        context = "\n\n".join([doc.content for doc in documents])
        
        # Generate answer
        result = self.chain.run(context=context, question=query)
        return result.strip()
    
    def prepare_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Prepare source information"""
        sources = []
        for i, doc in enumerate(documents, 1):
            source_info = {
                "rank": i,
                "content": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                "source": doc.meta.get("source_name", "Unknown"),
                "file_type": doc.meta.get("file_type", "Unknown"),
            }
            sources.append(source_info)
        
        return sources
    
    def query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Main query method"""
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Prepare sources
        sources = self.prepare_sources(documents)
        
        return {
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources)
        }

# Global service instance
query_service = QueryService()

def retrieve_and_answer(query_text: str, top_k: int = None) -> Dict[str, Any]:
    """Convenience function"""
    return query_service.query(query_text, top_k)