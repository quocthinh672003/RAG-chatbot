"""
Hybrid RAG Pipeline - Haystack + LangChain Fallback

Mục đích:
- Tạo RAG pipeline sử dụng Haystack framework (ưu tiên)
- Fallback sang LangChain nếu Haystack không khả dụng
- Xử lý document ingestion và query processing
- Quản lý vector store và embedding

Công nghệ sử dụng:
- Haystack: RAG framework chính (nếu khả dụng)
- LangChain: Fallback RAG framework
- OpenAI: LLM cho text generation
- Qdrant: Vector store cho persistence

Kiến trúc:
1. Document Processing: Chunking và embedding
2. Vector Storage: Qdrant Cloud
3. Retrieval: Semantic search với top-k results
4. Generation: LLM response với context
"""

from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import uuid
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try Haystack first
HAYSTACK_AVAILABLE = False
try:
    # Create a clean environment for Haystack
    import os
    import sys
    
    # Temporarily remove pandas from sys.modules
    original_modules = {}
    problematic_modules = ['pandas', 'numpy', 'pandas.core.frame']
    
    for module_name in problematic_modules:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
            del sys.modules[module_name]
    
    # Set environment variables for Haystack
    os.environ['PYDANTIC_ARBITRARY_TYPES_ALLOWED'] = 'true'
    os.environ['PYDANTIC_IGNORE_UNKNOWN'] = 'true'
    
    # Import Haystack components
    from haystack import Pipeline
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate
    
    # Restore original modules
    for module_name, module in original_modules.items():
        sys.modules[module_name] = module
    
    HAYSTACK_AVAILABLE = True
    logger.info("✅ Haystack enabled successfully")
except Exception as e:
    logger.warning(f"⚠️ Haystack not available: {e}")
    # Restore original modules if failed
    if 'original_modules' in locals():
        for module_name, module in original_modules.items():
            sys.modules[module_name] = module

# Try LangChain as fallback
LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate as LangChainPromptTemplate
    from langchain.chains import LLMChain
    
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain enabled as fallback")
except Exception as e:
    logger.warning(f"⚠️ LangChain not available: {e}")

if not HAYSTACK_AVAILABLE and not LANGCHAIN_AVAILABLE:
    logger.error("❌ Neither Haystack nor LangChain available!")

from config import config


class HybridRAGPipeline:
    """
    Hybrid RAG Pipeline
    
    Chức năng chính:
    1. Khởi tạo pipeline với Haystack framework (ưu tiên)
    2. Fallback sang LangChain nếu Haystack không khả dụng
    3. Xử lý document ingestion
    4. Thực hiện query processing
    5. Quản lý vector store
    """

    def __init__(self):
        """
        Khởi tạo Hybrid RAG Pipeline
        - Kiểm tra availability của Haystack và LangChain
        - Khởi tạo pipeline phù hợp
        """
        if HAYSTACK_AVAILABLE:
            self._init_haystack()
            self.active_pipeline = "Haystack"
            logger.info("🎯 Using Haystack as primary RAG pipeline")
        elif LANGCHAIN_AVAILABLE:
            self._init_langchain()
            self.active_pipeline = "LangChain"
            logger.info("🔄 Using LangChain as fallback RAG pipeline")
        else:
            raise ImportError("Neither Haystack nor LangChain available")

    def _init_haystack(self):
        """
        Khởi tạo Haystack pipeline
        
        Components:
        - Document Store: InMemoryDocumentStore với cosine similarity
        - Retriever: EmbeddingRetriever với OpenAI embeddings
        - Prompt Template: Template tùy chỉnh cho tiếng Việt
        - LLM: PromptNode với OpenAI model
        """
        try:
            # Document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension, similarity="cosine"
            )

            # Retriever với top_k cao hơn để có nhiều context
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=config.models.embedding_model,
                model_format="openai",
                api_key=config.openai_api_key,
                top_k=min(config.processing.top_k * 2, 10),  # Lấy nhiều documents hơn
            )

            # Prompt template
            self.prompt_template = PromptTemplate(
                prompt="""
                Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
                YÊU CẦU: 
                - Trả lời dựa trên thông tin có trong ngữ cảnh
                - Không tự thêm thông tin không có trong ngữ cảnh
                - Nếu có bảng: hiển thị bảng chính xác
                - Nếu có bullet points: liệt kê chính xác

                Hướng dẫn trả lời (bằng tiếng Việt):
                - Nếu có bảng: xuất lại bảng Markdown đầy đủ từ dữ liệu trong ngữ cảnh
                - Nếu có bullet points: liệt kê chính xác theo ngữ cảnh
                - Nếu có số liệu: trích đúng số, kèm đơn vị
                - Nếu ngữ cảnh không có thông tin liên quan: trả lời: 'Không tìm thấy thông tin trong tài liệu đã cung cấp.'

                [Ngữ cảnh]:
                {join(documents)}

                [Câu hỏi]: {query}

                Trả lời dựa trên ngữ cảnh được cung cấp.
                """,
                output_parser=None,
            )

            # LLM
            self.prompt_node = PromptNode(
                model_name_or_path=config.models.llm_model,
                api_key=config.openai_api_key,
                default_prompt_template=self.prompt_template,
                model_kwargs={"temperature": 0.2, "max_tokens": 1200},
            )

            # Build pipeline
            self.pipeline = self._build_haystack_pipeline()
            logger.info("✅ Haystack pipeline initialized successfully")

        except Exception as e:
            logger.error(f"❌ Haystack initialization failed: {e}")
            raise e

    def _init_langchain(self):
        """
        Khởi tạo LangChain pipeline
        
        Components:
        - Document Store: FAISS với OpenAI embeddings
        - Retriever: FAISS similarity search
        - Prompt Template: LangChain PromptTemplate
        - LLM: LangChain LLMChain
        """
        try:
            # Embeddings
            self.embeddings = OpenAIEmbeddings(
                model=config.models.embedding_model,
                api_key=config.openai_api_key,
            )

            # Document store - start with empty texts
            self.document_store = FAISS.from_texts(
                texts=[""],
                embedding=self.embeddings,
            )

            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.processing.chunk_size,
                chunk_overlap=config.processing.chunk_overlap,
            )

            # Prompt template
            self.prompt_template = LangChainPromptTemplate(
                input_variables=["context", "query"],
                template="""
                Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
                YÊU CẦU: 
                - Trả lời dựa trên thông tin có trong ngữ cảnh
                - Không tự thêm thông tin không có trong ngữ cảnh
                - Nếu có bảng: hiển thị bảng chính xác
                - Nếu có bullet points: liệt kê chính xác

                Hướng dẫn trả lời (bằng tiếng Việt):
                - Nếu có bảng: xuất lại bảng Markdown đầy đủ từ dữ liệu trong ngữ cảnh
                - Nếu có bullet points: liệt kê chính xác theo ngữ cảnh
                - Nếu có số liệu: trích đúng số, kèm đơn vị
                - Nếu ngữ cảnh không có thông tin liên quan: trả lời: 'Không tìm thấy thông tin trong tài liệu đã cung cấp.'

                [Ngữ cảnh]:
                {context}

                [Câu hỏi]: {query}

                Trả lời dựa trên ngữ cảnh được cung cấp.
                """,
            )

            # LLM
            self.llm = ChatOpenAI(
                model=config.models.llm_model,
                api_key=config.openai_api_key,
                temperature=0.2,
                max_tokens=1200,
            )

            # Chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
            )

            logger.info("✅ LangChain pipeline initialized successfully")

        except Exception as e:
            logger.error(f"❌ LangChain initialization failed: {e}")
            raise e

    def _build_haystack_pipeline(self):
        """Build Haystack RAG pipeline"""
        pipeline = Pipeline()
        pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        pipeline.add_node(
            component=self.prompt_node, name="PromptNode", inputs=["Retriever"]
        )
        return pipeline

    def add_documents(self, documents):
        """Add documents to the pipeline"""
        try:
            if self.active_pipeline == "Haystack":
                # Add to Haystack document store
                self.document_store.write_documents(documents)
                self.retriever.update_embeddings(documents)
            else:  # LangChain
                # Process documents for LangChain
                texts = []
                metadatas = []
                for doc in documents:
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        texts.append(chunk)
                        metadatas.append({
                            "source_name": doc.metadata.get("source_name", "Unknown"),
                            "page": doc.metadata.get("page", 0)
                        })
                
                # Add to FAISS
                self.document_store.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info(f"✅ Added {len(documents)} documents to {self.active_pipeline} pipeline")
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise e

    def query(self, query: str) -> Dict[str, Any]:
        """Query the pipeline"""
        try:
            if self.active_pipeline == "Haystack":
                # Use Haystack pipeline
                result = self.pipeline.run(query=query)
                return {
                    "answer": (
                        result["answers"][0].answer
                        if result["answers"]
                        else "Không tìm thấy câu trả lời."
                    ),
                    "documents": result["documents"],
                    "sources": [
                        doc.meta.get("source_name", "Unknown")
                        for doc in result["documents"]
                    ],
                    "pipeline": self.active_pipeline,
                }
            else:  # LangChain
                # Use LangChain pipeline
                # Search for relevant documents
                docs = self.document_store.similarity_search(
                    query, k=config.processing.top_k
                )
                
                if not docs:
                    return {
                        "answer": "Không tìm thấy thông tin trong tài liệu đã cung cấp.",
                        "documents": [],
                        "sources": [],
                        "pipeline": self.active_pipeline,
                    }
                
                # Combine context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate answer
                result = self.chain.run(context=context, query=query)
                
                return {
                    "answer": result,
                    "documents": docs,
                    "sources": [
                        doc.metadata.get("source_name", "Unknown")
                        for doc in docs
                    ],
                    "pipeline": self.active_pipeline,
                }
        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Error",
            }

    def get_document_count(self) -> int:
        """Get number of documents in store"""
        try:
            if self.active_pipeline == "Haystack":
                return self.document_store.get_document_count()
            else:  # LangChain
                return len(self.document_store.index_to_docstore_id)
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Hybrid RAG Pipeline",
            "active_pipeline": self.active_pipeline,
            "components": [
                "OpenAIEmbedder",
                "InMemoryEmbeddingRetriever", 
                "OpenAIGenerator",
            ],
        }


# Global pipeline instance
rag_pipeline = None

def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        try:
            rag_pipeline = HybridRAGPipeline()
            logger.info(
                f"✅ Hybrid RAG Pipeline initialized successfully"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hybrid RAG Pipeline: {e}")
            rag_pipeline = None
    return rag_pipeline
