"""
Haystack 2.x RAG Pipeline Service (Weaviate only)
"""

import logging
from typing import List, Dict, Any
import os
import re
import unicodedata
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Environment knobs for pydantic noise
os.environ["PYDANTIC_ARBITRARY_TYPES_ALLOWED"] = "true"
os.environ["PYDANTIC_IGNORE_UNKNOWN"] = "true"

# Haystack imports
try:
    from haystack import Document, Pipeline, component
    from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
    from haystack.components.preprocessors import DocumentSplitter
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator

    HAYSTACK_AVAILABLE = True
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"Haystack import failed: {e}")

# Weaviate client
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except Exception:
    WEAVIATE_AVAILABLE = False

from config import config


class WeaviateDocumentStore:
    """Minimal Weaviate v4 wrapper for Haystack integration."""

    def __init__(self, url: str, api_key: str, index: str):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client not installed")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=api_key),
        )
        self.index = index
        self.collection = None
        # OpenAI embeddings client (client-side embedding)
        try:
            from openai import OpenAI  # type: ignore
            self._embed_client = OpenAI(api_key=config.openai_api_key)
            self._embed_model = config.models.embedding_model
        except Exception:
            self._embed_client = None
            self._embed_model = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from weaviate.classes.config import Property, DataType, Configure

        try:
            self.collection = self.client.collections.get(self.index)
        except Exception:
            # Create collection configured for client-side vectors (no server vectorizer)
            self.collection = self.client.collections.create(
                name=self.index,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="file_type", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )

        # Ensure chat collection exists as well
        try:
            self.chat_collection = self.client.collections.get("Chats")
        except Exception:
            try:
                self.chat_collection = self.client.collections.create(
                    name="Chats",
                    properties=[
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="sources", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),
                )
            except Exception:
                self.chat_collection = None

    def write_documents(self, documents: List[Document]) -> None:
        if self.collection is None:
            self._ensure_collection()
        for doc in documents:
            props = {
                "content": doc.content,
                "source": (doc.meta or {}).get("source", "unknown"),
                "page": (doc.meta or {}).get("page", 0),
                "file_type": (doc.meta or {}).get("file_type", "unknown"),
            }
            # Compute client-side embedding if available
            vector = None
            if self._embed_client and self._embed_model and doc.content:
                try:
                    emb = self._embed_client.embeddings.create(model=self._embed_model, input=doc.content)
                    vector = emb.data[0].embedding  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Embedding failed, inserting without vector: {e}")
            if vector is not None:
                self.collection.data.insert(properties=props, vector=vector)
            else:
                self.collection.data.insert(properties=props)

    def query_documents(self, query: str, top_k: int = 5) -> List[Document]:
        if self.collection is None:
            self._ensure_collection()
        # Vector search only (collection uses vectorizer none)
        if not (self._embed_client and self._embed_model):
            logger.error("Embedding client/model not initialized for vector search")
            return []
        try:
            emb = self._embed_client.embeddings.create(model=self._embed_model, input=query)
            vector = emb.data[0].embedding  # type: ignore[attr-defined]
            # Weaviate v4: near_vector(near_vector=..., limit=...)
            res = self.collection.query.near_vector(near_vector=vector, limit=top_k)
        except Exception as e:
            logger.error(f"near_vector failed: {e}")
            return []
        docs: List[Document] = []
        for obj in getattr(res, "objects", []) or []:
            props = getattr(obj, "properties", {}) or {}
            content = props.get("content")
            if not content:
                continue
            meta = {
                "source": props.get("source", "unknown"),
                "page": props.get("page", 0),
                "file_type": props.get("file_type", "unknown"),
            }
            docs.append(Document(content=content, meta=meta))
        return docs

    def delete_documents(self) -> None:
        try:
            self.client.collections.delete(self.index)
        except Exception:
            pass
        self.collection = None
        self._ensure_collection()

    def count_documents(self) -> int:
        try:
            if self.collection is None:
                self._ensure_collection()
            # Fallback: iterate – safe on small data
            count = 0
            for _ in self.collection.iterator():
                count += 1
            return count
        except Exception:
            return 0

    # --- Chat persistence helpers ---
    def write_chat_interaction(self, question: str, answer: str, sources: list, timestamp: str) -> None:
        """Persist one chat Q/A with optional vector over question for retrieval."""
        try:
            if self.chat_collection is None:
                self._ensure_collection()
            props = {
                "question": question,
                "answer": answer,
                "sources": ", ".join(sources or []),
                "timestamp": timestamp,
            }
            vector = None
            if self._embed_client and self._embed_model and question:
                try:
                    emb = self._embed_client.embeddings.create(model=self._embed_model, input=question)
                    vector = emb.data[0].embedding  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Embedding chat failed: {e}")
            if vector is not None:
                self.chat_collection.data.insert(properties=props, vector=vector)
            else:
                self.chat_collection.data.insert(properties=props)
        except Exception as e:
            logger.error(f"Failed to write chat interaction: {e}")


@component
class WeaviateRetriever:
    def __init__(self, store: WeaviateDocumentStore, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, query: str):
        return {"documents": self.store.query_documents(query=query, top_k=self.top_k)}


class TextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class HaystackRAGPipeline:
    """Haystack 2.x pipeline wired to Weaviate only (no InMemory)."""

    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack 2.x not available")
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client not available")
        self._init()

    def _init(self) -> None:
        # Document store (Weaviate only)
        self.document_store = WeaviateDocumentStore(
            url=config.database.weaviate_url,
            api_key=config.database.weaviate_api_key,
            index=config.database.weaviate_class_name,
        )

        # Splitter
        self.document_splitter = DocumentSplitter(
            split_by="word", split_length=config.processing.chunk_size, split_overlap=config.processing.chunk_overlap
        )

        # Retriever component
        self.retriever = WeaviateRetriever(self.document_store, top_k=config.processing.top_k)

        # Ranker and LLM
        self.diversity_ranker = LostInTheMiddleRanker(top_k=config.processing.top_k)
        from haystack.utils.auth import Secret
        self.generator = OpenAIGenerator(
            api_key=Secret.from_token(config.openai_api_key),
            model=config.models.llm_model,
            generation_kwargs={
                "temperature": 0,
                "max_tokens": config.models.max_tokens,
                "response_format": {"type": "json_object"},
            },
        )

        # Prompt
        self.prompt_builder = PromptBuilder(
            template=(
                """
Bạn là trợ lý AI thông minh, nhiệm vụ: trả lời dựa duy nhất vào NGỮ CẢNH dưới đây. Tuyệt đối không bịa.

<<<<<<< Updated upstream
NGUYÊN TẮC:
- Chỉ dùng thông tin trong tài liệu được cung cấp; nếu không có thông tin liên quan, nói rõ là không có trong tài liệu.
- Mọi luận điểm quan trọng phải kèm trích nguồn cụ thể (file, trang, loại).
- Nếu có bảng liên quan, tái tạo bảng bằng Markdown và đưa vào mảng tables.
- Đầu ra phải là JSON hợp lệ duy nhất, không có text ngoài JSON.
=======
            # Choose retriever: embeddings for Weaviate, BM25 for fallback
            store_type_name = type(self.document_store).__name__
            self.retriever = None
            if (
                hasattr(self.document_store, "query_by_embedding")
                and "Weaviate" in store_type_name
            ):
                # Lightweight embedding-based retriever for Weaviate
                class WeaviateEmbeddingRetriever:
                    def __init__(self, document_store, openai_client, top_k: int):
                        self.document_store = document_store
                        self.openai_client = openai_client
                        self.top_k = top_k

                    def retrieve(self, query: str):
                        try:
                            emb = self.openai_client.embeddings.create(
                                model=config.models.embedding_model, input=query
                            )
                            q_vec = emb.data[0].embedding
                            results = self.document_store.query_by_embedding(
                                q_vec, top_k=self.top_k
                            )
                            return results or []
                        except Exception as e:
                            logger.error(f"❌ Weaviate embedding retrieve failed: {e}")
                            return []

                self.retriever = WeaviateEmbeddingRetriever(
                    self.document_store, None, config.processing.top_k
                )
                logger.info("✅ Using embedding-based retriever with Weaviate")
            else:
                # Fallback: BM25 for in-memory store
                self.retriever = BM25Retriever(
                    document_store=self.document_store,
                    top_k=config.processing.top_k,
                )
                logger.info("✅ Using BM25Retriever (fallback)")
>>>>>>> Stashed changes

NGỮ CẢNH (có trích nguồn):
{% for doc in documents %}
=== DOC {{ loop.index }} ===
SOURCE: {{ doc.meta.source | default('unknown') }} | PAGE: {{ doc.meta.page | default(0) }} | TYPE: {{ doc.meta.file_type | default('unknown') }}
CONTENT:
{{ doc.content }}
{% endfor %}

CÂU HỎI: {{ query }}

YÊU CẦU ĐẦU RA (JSON):
{
  "answer": "Câu trả lời đầy đủ trực tiếp cho câu hỏi, ưu tiên chính xác và trung thực. Nếu không có thông tin trong tài liệu, ghi rõ.",
  "details": "Giải thích chi tiết dựa trên các đoạn trích phù hợp trong NGỮ CẢNH. Không thêm thông tin ngoài tài liệu.",
  "tables": [
    "(Tùy chọn) Bảng ở dạng Markdown nếu có bảng liên quan trong tài liệu, bảo toàn đầy đủ dữ liệu và format."
  ],
  "sources": [
    {
      "file": "Tên file từ doc.meta.source",
      "page": "Số trang từ doc.meta.page (nếu có)",
      "type": "doc.meta.file_type",
      "snippet": "Trích đoạn ngắn minh họa (nguyên văn)"
    }
  ],
  "limitations": "Nêu rõ giới hạn dữ liệu, phần thiếu, hoặc mâu thuẫn nếu có.",
  "follow_up_questions": ["(Tùy chọn) Gợi ý các câu hỏi tiếp theo nếu cần"]
}

QUY TẮC BỔ SUNG:
- Nếu không tìm thấy thông tin liên quan: đặt short_answer nêu rõ không có dữ liệu trong tài liệu; details rỗng; sources là mảng rỗng; tables rỗng.
- Tuyệt đối không chèn Markdown hay văn bản ngoài JSON. Trả về đúng một JSON duy nhất.
"""
            ),
            required_variables=["query", "documents"],
        )

<<<<<<< Updated upstream
        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("diversity_ranker", self.diversity_ranker)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)
=======
            # Inject client into custom retriever if applicable
            try:
                if self.retriever and hasattr(self.retriever, "openai_client"):
                    self.retriever.openai_client = self.openai_client
            except Exception:
                pass

            # Build pipeline
            self.pipeline = self._build_pipeline()
>>>>>>> Stashed changes

        self.pipeline.connect("retriever.documents", "diversity_ranker")
        self.pipeline.connect("diversity_ranker.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.prompt")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        hs_docs: List[Document] = []
        for d in documents:
            content = d.get("page_content")
            if not content:
                continue
            meta = (d.get("metadata") or {})
            source = meta.get("source") or meta.get("source_name") or meta.get("source_filename") or "unknown"
            page = meta.get("page") or meta.get("page_number") or meta.get("paragraph_index") or 0
            file_type = meta.get("file_type")
            if not file_type and isinstance(source, str):
                try:
                    _, ext = os.path.splitext(source)
                    file_type = ext.lstrip(".") or "unknown"
                except Exception:
                    file_type = "unknown"
            hs_docs.append(Document(content=content, meta={
                "source": source,
                "page": page,
                "file_type": file_type or "unknown",
            }))
        if not hs_docs:
            return
<<<<<<< Updated upstream
        split_out = self.document_splitter.run(documents=hs_docs)
        chunks = split_out.get("documents", []) if isinstance(split_out, dict) else []
        self.document_store.write_documents(chunks)
=======

        try:
            # Suppress ALL preprocessing logs
            import logging

            haystack_logger = logging.getLogger(
                "haystack.nodes.preprocessor.preprocessor"
            )
            original_level = haystack_logger.level
            haystack_logger.setLevel(logging.CRITICAL)

            # Also suppress other noisy loggers
            logging.getLogger("haystack.nodes.retriever").setLevel(logging.CRITICAL)
            logging.getLogger("haystack.document_stores").setLevel(logging.CRITICAL)
            logging.getLogger("unstructured").setLevel(logging.CRITICAL)
            logging.getLogger("pdfminer").setLevel(logging.CRITICAL)
            logging.getLogger("PIL").setLevel(logging.CRITICAL)

            # Debug: Log before preprocessing
            logger.info(f"🔍 Before preprocessing: {len(haystack_docs)} documents")
            if haystack_docs:
                logger.info(
                    f"🔍 First doc content: {haystack_docs[0].content[:100]}..."
                )

            # Use preprocessing to create proper chunks
            preprocessed_docs = self.preprocessor.run(haystack_docs)
            if isinstance(preprocessed_docs, dict) and "documents" in preprocessed_docs:
                preprocessed_docs = preprocessed_docs["documents"]
            logger.info(f"🔍 Preprocessed {len(preprocessed_docs)} documents")

            # If using Weaviate, create embeddings before writing
            store_type_name = type(self.document_store).__name__
            if (
                hasattr(self.document_store, "query_by_embedding")
                and "Weaviate" in store_type_name
            ):
                try:
                    contents = [doc.content for doc in preprocessed_docs]
                    if contents:
                        emb_resp = self.openai_client.embeddings.create(
                            model=config.models.embedding_model, input=contents
                        )
                        vectors = [item.embedding for item in emb_resp.data]
                        for doc, vec in zip(preprocessed_docs, vectors):
                            try:
                                doc.embedding = vec
                            except Exception:
                                pass
                except Exception as e:
                    logger.error(f"❌ Failed to embed documents for Weaviate: {e}")

            # Add preprocessed documents to document store
            self.document_store.write_documents(preprocessed_docs)

            # Verify documents were added (minimal logging)
            try:
                all_docs = self.document_store.get_all_documents()
                logger.info(
                    f"✅ Added {len(preprocessed_docs)} documents to store (total: {len(all_docs)})"
                )
            except Exception:
                pass  # Silently ignore verification errors

        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        finally:
            # Restore logging levels
            haystack_logger.setLevel(original_level)
>>>>>>> Stashed changes

    def query(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"answer": "Vui lòng nhập câu hỏi.", "documents": [], "sources": [], "pipeline": "Haystack 2.x RAG"}
        # Provide `query` to both retriever and prompt_builder
        out = self.pipeline.run({
            "retriever": {"query": query},
            "prompt_builder": {"query": query}
        })
        if not isinstance(out, dict):
            return {"answer": str(out), "documents": [], "sources": [], "pipeline": "Haystack 2.x RAG (Weaviate)"}
        gen = out.get("generator", {}) if isinstance(out.get("generator", {}), dict) else {}
        replies = gen.get("replies", []) if isinstance(gen, dict) else []
        if replies:
            first = replies[0]
            if isinstance(first, str):
                answer = first
            elif isinstance(first, dict):
                answer = first.get("content", str(first))
            else:
                answer = str(first)
        else:
            answer = "Không tìm thấy thông tin liên quan."
        docs = out.get("diversity_ranker", {}).get("documents", [])
        sources = []
        for d in docs:
            meta = getattr(d, "meta", {}) or {}
            src = meta.get("source", "Unknown")
            try:
                src = os.path.basename(src) if isinstance(src, str) else src
            except Exception:
                pass
            if src:
                sources.append(src)
        return {"answer": answer, "documents": docs, "sources": sources, "pipeline": "Haystack 2.x RAG (Weaviate)"}

    def get_document_count(self) -> int:
        return self.document_store.count_documents()

    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Haystack 2.x RAG Pipeline",
            "active_pipeline": "Haystack 2.x",
            "components": [
                "WeaviateDocumentStore",
                "DocumentSplitter",
                "WeaviateRetriever",
                "LostInTheMiddleRanker",
                "PromptBuilder",
                "OpenAIGenerator",
            ],
            "features": ["RAG", "Weaviate", "Haystack 2.x"],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available",
        }

    def clear_documents(self) -> None:
        self.document_store.delete_documents()


class FallbackRAGPipeline:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents.extend(documents or [])

    def query(self, query: str) -> Dict[str, Any]:
        return {"answer": "RAG fallback active.", "documents": [], "sources": [], "pipeline": "Fallback"}

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_pipeline_info(self) -> Dict[str, Any]:
        return {"document_count": self.get_document_count(), "pipeline_type": "Fallback", "active_pipeline": "Fallback", "components": ["Fallback"], "features": [], "status": "Available"}


# Global instance
try:
    rag_pipeline = HaystackRAGPipeline() if HAYSTACK_AVAILABLE else FallbackRAGPipeline()
    logger.info("RAG pipeline ready")
except Exception as e:
    logger.error(f"Failed to init Haystack pipeline: {e}")
    rag_pipeline = FallbackRAGPipeline()
    logger.info("Using fallback pipeline")


# --- Optional utilities for diagnostics ---
def debug_vector_status() -> Dict[str, Any]:
    """Return basic diagnostic info about collection and vector availability."""
    info: Dict[str, Any] = {}
    try:
        if isinstance(rag_pipeline, HaystackRAGPipeline):
            store = rag_pipeline.document_store
            # count
            info["document_count"] = store.count_documents()
            # try one vector query with a dummy token to verify near_vector works
            try:
                _ = store.query_documents("test", top_k=1)
                info["near_vector_ok"] = True
            except Exception as e:
                info["near_vector_ok"] = False
                info["near_vector_error"] = str(e)
        else:
            info["status"] = "fallback"
    except Exception as e:
        info["error"] = str(e)
    return info


