"""
Hybrid RAG Pipeline: Haystack Core + LangChain Fallback
"""

from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import uuid

# Try Haystack first
try:
    from haystack import Document, Pipeline
    from haystack.nodes import (
        EmbeddingRetriever,
        SentenceTransformersRanker,
        LostInTheMiddleRanker,
    )
    from haystack.nodes import PromptNode, PromptTemplate
    from haystack.document_stores import InMemoryDocumentStore

    HAYSTACK_AVAILABLE = True
    print("✅ Haystack loaded successfully")
except ImportError as e:
    print(f"⚠️ Haystack not available: {e}")
    HAYSTACK_AVAILABLE = False

# LangChain fallback
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate as LangChainPromptTemplate
    from langchain.chains import LLMChain

    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain fallback loaded successfully")
except ImportError as e:
    print(f"⚠️ LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

from config import config


class HybridRAGPipeline:
    """Hybrid RAG Pipeline with Haystack core + LangChain fallback"""

    def __init__(self):
        self.use_haystack = HAYSTACK_AVAILABLE
        self.use_langchain = not HAYSTACK_AVAILABLE and LANGCHAIN_AVAILABLE

        if self.use_haystack:
            self._init_haystack()
        elif self.use_langchain:
            self._init_langchain()
        else:
            raise ImportError("Neither Haystack nor LangChain available")

    def _init_haystack(self):
        """Initialize Haystack pipeline"""
        try:
            # Document store
            self.document_store = InMemoryDocumentStore(
                embedding_dim=config.models.embedding_dimension, similarity="cosine"
            )

            # Retriever
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=config.models.embedding_model,
                model_format="openai",
                api_key=config.openai_api_key,
                top_k=config.processing.top_k,
            )

            # Rankers
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
            print("✅ Haystack pipeline initialized successfully")

        except Exception as e:
            print(f"❌ Haystack initialization failed: {e}")
            self.use_haystack = False
            if LANGCHAIN_AVAILABLE:
                self._init_langchain()
            else:
                raise e

    def _init_langchain(self):
        """Initialize LangChain fallback"""
        try:
            # Embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=config.openai_api_key
            )

            # Vector store
            self.vector_store = FAISS.from_texts(["Initial document"], self.embeddings)

            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.processing.chunk_size,
                chunk_overlap=config.processing.chunk_overlap,
                length_function=len,
            )

            # LLM
            self.llm = ChatOpenAI(
                model_name=config.models.llm_model,
                temperature=0.2,
                openai_api_key=config.openai_api_key,
                max_tokens=1200,
            )

            # Prompt template
            self.prompt_template = LangChainPromptTemplate(
                input_variables=["context", "question"],
                template="""
                Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
                YÊU CẦU NGHIÊM NGẶT: Không suy đoán, không dùng kiến thức ngoài ngữ cảnh.

                Hướng dẫn trả lời (bằng tiếng Việt):
                - Nếu có số liệu/bảng: trích đúng số, kèm đơn vị (ví dụ: TWh, %, tỷ lệ). Trả lời đầy đủ, KHÔNG tóm tắt.
                - Ưu tiên số liệu đúng NĂM/ĐỊA ĐIỂM được hỏi; nếu nhiều mục (ví dụ theo ngành), liệt kê rõ ràng.
                - Nếu có bảng phù hợp: xuất lại bảng Markdown đầy đủ từ dữ liệu trong ngữ cảnh (không lược bớt cột chính).
                - Nếu ngữ cảnh không đủ thông tin: trả lời đúng câu sau: 'Không tìm thấy thông tin trong tài liệu đã cung cấp.'

                [Ngữ cảnh]:
                {context}

                [Câu hỏi]: {question}

                Xuất trả lời ở dạng Markdown, có thể bao gồm bảng, bullet. ĐẦY ĐỦ theo ngữ cảnh.
                """,
            )

            # Modern chain using RunnableSequence
            self.chain = self.prompt_template | self.llm
            print("✅ LangChain fallback initialized successfully")

        except Exception as e:
            print(f"❌ LangChain initialization failed: {e}")
            raise e

    def _build_haystack_pipeline(self):
        """Build Haystack RAG pipeline"""
        pipeline = Pipeline()
        pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        pipeline.add_node(
            component=self.similarity_ranker,
            name="SimilarityRanker",
            inputs=["Retriever"],
        )
        pipeline.add_node(
            component=self.diversity_ranker,
            name="DiversityRanker",
            inputs=["SimilarityRanker"],
        )
        pipeline.add_node(
            component=self.prompt_node, name="PromptNode", inputs=["DiversityRanker"]
        )
        return pipeline

    def add_documents(self, documents):
        """Add documents to the pipeline"""
        try:
            if self.use_haystack:
                # Add to Haystack document store
                self.document_store.write_documents(documents)
                self.retriever.update_embeddings(documents)
                print(f"✅ Added {len(documents)} documents to Haystack pipeline")
            elif self.use_langchain:
                # Add to LangChain vector store
                for doc in documents:
                    # Handle both Haystack and LangChain document formats
                    if hasattr(doc, "content"):
                        content = doc.content
                        meta = doc.meta if hasattr(doc, "meta") else {}
                    else:
                        content = doc.page_content
                        meta = doc.metadata

                    texts = self.text_splitter.split_text(content)
                    self.vector_store.add_texts(texts, metadatas=[meta] * len(texts))
                print(f"✅ Added {len(documents)} documents to LangChain pipeline")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise e

    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            if self.use_haystack:
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
                    "pipeline": "Haystack",
                }
            elif self.use_langchain:
                # Use LangChain fallback
                docs = self.vector_store.similarity_search(
                    query, k=config.processing.top_k
                )
                context = "\n\n".join([doc.page_content for doc in docs])

                result = self.chain.invoke({"context": context, "question": query})

                return {
                    "answer": (
                        result.content if hasattr(result, "content") else str(result)
                    ),
                    "documents": docs,
                    "sources": [
                        doc.metadata.get("source_name", "Unknown") for doc in docs
                    ],
                    "pipeline": "LangChain",
                }
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return {
                "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Error",
            }

    def get_document_count(self) -> int:
        """Get number of documents in store"""
        try:
            if self.use_haystack:
                return self.document_store.get_document_count()
            elif self.use_langchain:
                return len(self.vector_store.index_to_docstore_id)
        except:
            return 0

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": (
                "Haystack Core + LangChain Fallback"
                if self.use_haystack
                else "LangChain Fallback"
            ),
            "active_pipeline": "Haystack" if self.use_haystack else "LangChain",
            "components": [
                "OpenAIEmbedder",
                (
                    "InMemoryEmbeddingRetriever"
                    if self.use_haystack
                    else "FAISS VectorStore"
                ),
                (
                    "SentenceTransformersRanker"
                    if self.use_haystack
                    else "SimilaritySearch"
                ),
                "LostInTheMiddleRanker" if self.use_haystack else "N/A",
                "OpenAIGenerator",
            ],
        }


# Global pipeline instance
try:
    rag_pipeline = HybridRAGPipeline()
    print(
        f"✅ Hybrid RAG Pipeline initialized with {rag_pipeline.get_pipeline_info()['active_pipeline']}"
    )
except Exception as e:
    print(f"❌ Failed to initialize Hybrid RAG Pipeline: {e}")
    rag_pipeline = None
