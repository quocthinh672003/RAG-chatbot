"""
Hybrid RAG Pipeline: Haystack Core + LangChain Fallback
"""

from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import uuid

# Skip Haystack due to Pydantic DataFrame conflict
HAYSTACK_AVAILABLE = False

# LangChain fallback
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate as LangChainPromptTemplate
    from langchain.chains import LLMChain

    LANGCHAIN_AVAILABLE = True
    print("LangChain fallback loaded successfully")
except ImportError as e:
    print(f"LangChain not available: {e}")
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
            print("Haystack pipeline initialized successfully")

        except Exception as e:
            print(f"Haystack initialization failed: {e}")
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

            # Vector store (lazy init on first add)
            self.vector_store = None
            self._simple_store_texts = []
            self._simple_store_vectors = []
            self._simple_store_metas = []
            
            # Try to load existing FAISS index
            try:
                if os.path.exists("faiss_index"):
                    from langchain_community.vectorstores import FAISS
                    self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
                    print("Loaded existing FAISS index")
            except Exception as e:
                print(f"Could not load existing FAISS index: {e}")
                self.vector_store = None

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
                Bạn là trợ lý dữ liệu, trả lời dựa trên NGỮ CẢNH được cung cấp.
                
                Hướng dẫn trả lời (bằng tiếng Việt):
                - Trả lời dựa trên thông tin có trong ngữ cảnh
                - Nếu có bảng: hiển thị bảng chính xác
                - Nếu có bullet points: liệt kê chính xác
                - Nếu có số liệu: trích đúng số, kèm đơn vị
                - Nếu có link/URL: hiển thị link đầy đủ
                - Nếu có image/video/audio: hiển thị đầy đủ
                - Nếu có reference name/identifier: hiển thị đầy đủ
                - Khi được hỏi về link/reference: tìm và liệt kê tất cả reference name có trong ngữ cảnh
                - Khi được hỏi "Có link nào": tìm tất cả text có dạng reference name, identifier, tên báo cáo, hoặc bất kỳ text nào có thể là link/reference
                - Nếu thấy text có dạng "Market Insights", "AHA", "WEF", "PHTI", hoặc tên báo cáo khác: liệt kê tất cả
                - Nếu có thông tin liên quan (dù ít): hãy trả lời dựa trên thông tin đó
                - Chỉ trả lời 'Không tìm thấy thông tin trong tài liệu đã cung cấp.' khi hoàn toàn không có thông tin liên quan

                [Ngữ cảnh]:
                {context}

                [Câu hỏi]: {question}

                Trả lời dựa trên ngữ cảnh được cung cấp.
                """,
            )

            # Modern chain using RunnableSequence
            self.chain = self.prompt_template | self.llm
            print("LangChain fallback initialized successfully")

        except Exception as e:
            print(f"LangChain initialization failed: {e}")
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
                print(f"Added {len(documents)} documents to Haystack pipeline")
            elif self.use_langchain:
                # Add to LangChain vector store
                # Prepare all texts and metas
                all_texts = []
                all_metas = []
                for doc in documents:
                    # Handle both Haystack and LangChain document formats
                    if hasattr(doc, "content"):
                        content = doc.content
                        meta = doc.meta if hasattr(doc, "meta") else {}
                    else:
                        content = doc.page_content
                        meta = doc.metadata

                    chunk_texts = self.text_splitter.split_text(content)
                    all_texts.extend(chunk_texts)
                    all_metas.extend([meta] * len(chunk_texts))

                # Try FAISS first if available
                if self.vector_store is None:
                    try:
                        from langchain_community.vectorstores import FAISS  # local import to delay faiss
                        self.vector_store = FAISS.from_texts(all_texts, self.embeddings, metadatas=all_metas)
                        # Save FAISS index to disk
                        self.vector_store.save_local("faiss_index")
                        print(f"Added {len(documents)} documents to LangChain pipeline (FAISS) and saved to disk")
                        return
                    except Exception as e:
                        print(f"FAISS unavailable, using simple in-memory store: {e}")
                        self.vector_store = None

                if self.vector_store is not None:
                    self.vector_store.add_texts(all_texts, metadatas=all_metas)
                    # Save updated FAISS index to disk
                    self.vector_store.save_local("faiss_index")
                    print(f"Added {len(documents)} documents to LangChain pipeline (FAISS) and saved to disk")
                else:
                    # Simple in-memory store with precomputed embeddings
                    embeddings = self.embeddings.embed_documents(all_texts)
                    self._simple_store_texts.extend(all_texts)
                    self._simple_store_metas.extend(all_metas)
                    self._simple_store_vectors.extend(embeddings)
                    print(f"Added {len(documents)} documents to LangChain pipeline (simple store)")
        except Exception as e:
            print(f"Error adding documents: {e}")
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
                if self.vector_store is not None:
                    docs = self.vector_store.similarity_search(
                        query, k=config.processing.top_k
                    )
                else:
                    # Simple cosine similarity over in-memory vectors
                    import math
                    if not self._simple_store_vectors:
                        docs = []
                    else:
                        q_vec = self.embeddings.embed_query(query)
                        # Normalize
                        def norm(v):
                            return math.sqrt(sum(x*x for x in v)) or 1.0
                        qn = norm(q_vec)
                        scores = []
                        for idx, vec in enumerate(self._simple_store_vectors):
                            vn = norm(vec)
                            dot = sum(a*b for a,b in zip(q_vec, vec))
                            scores.append((idx, dot/(qn*vn)))
                        scores.sort(key=lambda x: x[1], reverse=True)
                        top = scores[: config.processing.top_k]
                        # Wrap in LangChain-like Document objects
                        from types import SimpleNamespace
                        docs = [
                            SimpleNamespace(page_content=self._simple_store_texts[i], metadata=self._simple_store_metas[i])
                            for i,_ in top
                        ]
                context = "\n\n".join([getattr(doc, 'page_content', '') for doc in docs])
                
                # Debug: Print context for troubleshooting
                print(f"DEBUG: Context length = {len(context)}")
                print(f"DEBUG: Context preview = {context[:500]}...")
                print(f"DEBUG: Query = {query}")

                result = self.chain.invoke({"context": context, "question": query})

                return {
                    "answer": (
                        result.content if hasattr(result, "content") else str(result)
                    ),
                    "documents": docs,
                    "sources": [
                        (getattr(doc, 'metadata', {}) or {}).get("source_name", "Unknown") for doc in docs
                    ],
                    "pipeline": "LangChain",
                }
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
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
                if self.vector_store is not None:
                    return len(self.vector_store.index_to_docstore_id)
                else:
                    return len(self._simple_store_texts)
        except Exception as e:
            print(f"Error getting document count: {e}")
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
        f"Hybrid RAG Pipeline initialized with {rag_pipeline.get_pipeline_info()['active_pipeline']}"
    )
    
    # Reload documents if processed_files.txt exists
    if os.path.exists("processed_files.txt"):
                try:
                    from services.ingest_service import ingestion_service
                    from services.image_database import image_db
            
            with open("processed_files.txt", "r", encoding="utf-8") as f:
                processed_files = [line.strip() for line in f.readlines() if line.strip()]
            
            if processed_files:
                print(f"Reloading {len(processed_files)} processed files...")
                for file_name in processed_files:
                    # Fix double extensions (e.g., .pdf.pdf -> .pdf)
                    clean_name = file_name
                    if clean_name.endswith('.pdf.pdf'):
                        clean_name = clean_name.replace('.pdf.pdf', '.pdf')
                    elif clean_name.endswith('.docx.docx'):
                        clean_name = clean_name.replace('.docx.docx', '.docx')
                    elif clean_name.endswith('.xlsx.xlsx'):
                        clean_name = clean_name.replace('.xlsx.xlsx', '.xlsx')
                    elif clean_name.endswith('.md.md'):
                        clean_name = clean_name.replace('.md.md', '.md')
                    
                    # Try to find file in uploads folder
                    file_path = os.path.join("uploads", clean_name)
                    if not os.path.exists(file_path):
                        # Try with original name
                        file_path = os.path.join("uploads", file_name)
                    if not os.path.exists(file_path):
                        # Try original path
                        file_path = clean_name
                    
                    if os.path.exists(file_path):
                        try:
                            # Ingest document via ingestion service (handles splitting and adding to pipeline)
                            doc_id = ingestion_service.ingest_document(file_path)
                            if doc_id:
                                # Also extract images into the image database so they are queryable after restart
                                image_db.extract_images_from_any_file(file_path, clean_name)
                                print(f"Reloaded: {clean_name}")
                        except Exception as e:
                            print(f"Failed to reload {clean_name}: {e}")
                    else:
                        print(f"File not found: {clean_name} (tried: {file_path})")
                        
        except Exception as e:
            print(f"Error reloading documents: {e}")
            
except Exception as e:
    print(f"Failed to initialize Hybrid RAG Pipeline: {e}")
    rag_pipeline = None
