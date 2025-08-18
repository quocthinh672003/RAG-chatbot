"""
Haystack 2.x RAG Pipeline Service - Chỉ dùng components chính thức
"""

import logging
from typing import List, Dict, Any
import os
import re
import unicodedata
from datetime import datetime

logger = logging.getLogger(__name__)

# Environment knobs for pydantic noise
os.environ["PYDANTIC_ARBITRARY_TYPES_ALLOWED"] = "true"
os.environ["PYDANTIC_IGNORE_UNKNOWN"] = "true"

# Haystack imports - Dùng components có sẵn
try:
    from haystack import Document, Pipeline
    from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
    from haystack.components.preprocessors import DocumentSplitter
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
    from haystack.utils.auth import Secret

    HAYSTACK_AVAILABLE = True
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"Haystack import failed: {e}")

# Weaviate client trực tiếp
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except Exception:
    WEAVIATE_AVAILABLE = False

from config import config


class WeaviateStore:
    """Simple Weaviate wrapper for Haystack integration"""

    def __init__(self, url: str, api_key: str, index: str):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client not installed")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=api_key),
        )
        self.index = index
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self):
        from weaviate.classes.config import Property, DataType, Configure

        try:
            self.collection = self.client.collections.get(self.index)
        except Exception:
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

            # Get embedding from document
            vector = getattr(doc, "embedding", None)
            if vector is not None:
                self.collection.data.insert(properties=props, vector=vector)
            else:
                self.collection.data.insert(properties=props)

    def query_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        if self.collection is None:
            self._ensure_collection()

        try:
            res = self.collection.query.near_vector(vector=query_embedding, limit=top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        docs = []
        for obj in getattr(res, "objects", []) or []:
            props = getattr(obj, "properties", {}) or {}
            doc = Document(
                content=props.get("content", ""),
                meta={
                    "source": props.get("source", "unknown"),
                    "page": props.get("page", 0),
                    "file_type": props.get("file_type", "unknown"),
                },
            )
            docs.append(doc)
        return docs

    def count_documents(self) -> int:
        if self.collection is None:
            self._ensure_collection()
        try:
            return self.collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            return 0

    def delete_documents(self) -> None:
        if self.collection is None:
            self._ensure_collection()
        try:
            self.collection.data.delete_many()
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")


class HaystackRAGPipeline:
    """Haystack 2.x pipeline - Chỉ dùng components chính thức"""

    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack 2.x not available")
        self._init()

    def _init(self) -> None:
        # Weaviate store (vector store)
        self.weaviate_store = WeaviateStore(
            url=config.database.weaviate_url,
            api_key=config.database.weaviate_api_key,
            index=config.database.weaviate_class_name,
        )

        # Splitter
        self.document_splitter = DocumentSplitter(
            split_by="word",
            split_length=config.processing.chunk_size,
            split_overlap=config.processing.chunk_overlap,
        )

        # Embedders (Haystack chính thức)
        self.document_embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token(config.openai_api_key), model=config.models.embedding_model
        )

        self.text_embedder = OpenAITextEmbedder(
            api_key=Secret.from_token(config.openai_api_key), model=config.models.embedding_model
        )

        # Retriever (Haystack chính thức) - dùng InMemory nhưng query Weaviate
        # Note: InMemoryEmbeddingRetriever requires document_store, but we're using Weaviate directly
        # So we'll skip the retriever for now and use Weaviate directly in query method
        self.retriever = None

        # Ranker and LLM
        self.diversity_ranker = LostInTheMiddleRanker(top_k=config.processing.top_k)

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

NGUYÊN TẮC:
- Chỉ dùng thông tin trong tài liệu được cung cấp; nếu không có thông tin liên quan, nói rõ là không có trong tài liệu.
- Mọi luận điểm quan trọng phải kèm trích nguồn cụ thể (file, trang, loại).
- Nếu có bảng liên quan, tái tạo bảng bằng Markdown và đưa vào mảng tables.
- Đầu ra phải là JSON hợp lệ duy nhất, không có text ngoài JSON.

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

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("diversity_ranker", self.diversity_ranker)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)

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
            meta = d.get("metadata") or {}
            source = (
                meta.get("source")
                or meta.get("source_name")
                or meta.get("source_filename")
                or "unknown"
            )
            page = meta.get("page") or meta.get("page_number") or meta.get("paragraph_index") or 0
            file_type = meta.get("file_type")
            if not file_type and isinstance(source, str):
                try:
                    _, ext = os.path.splitext(source)
                    file_type = ext.lstrip(".") or "unknown"
                except Exception:
                    file_type = "unknown"
            hs_docs.append(
                Document(
                    content=content,
                    meta={
                        "source": source,
                        "page": page,
                        "file_type": file_type or "unknown",
                    },
                )
            )
        if not hs_docs:
            return

        # Split documents
        split_out = self.document_splitter.run(documents=hs_docs)
        chunks = split_out.get("documents", []) if isinstance(split_out, dict) else []

        # Embed documents với Haystack embedder
        embedded_docs = self.document_embedder.run(documents=chunks)
        embedded_chunks = embedded_docs.get("documents", [])

        # Write to Weaviate Cloud
        self.weaviate_store.write_documents(embedded_chunks)

    def query(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack 2.x RAG",
            }

        # Generate query embedding
        query_embedding = self.text_embedder.run(text=query)
        query_vector = query_embedding.get("embedding", [])

        # Query Weaviate directly
        docs = self.weaviate_store.query_documents(query_vector, top_k=config.processing.top_k)

        # Use Haystack pipeline for ranking and generation
        out = self.pipeline.run(
            {"diversity_ranker": {"documents": docs}, "prompt_builder": {"query": query}}
        )
        if not isinstance(out, dict):
            return {
                "answer": str(out),
                "documents": [],
                "sources": [],
                "pipeline": "Haystack 2.x RAG (Weaviate)",
            }
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
        return {
            "answer": answer,
            "documents": docs,
            "sources": sources,
            "pipeline": "Haystack 2.x RAG (Weaviate)",
        }

    def get_document_count(self) -> int:
        return self.weaviate_store.count_documents()

    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Haystack 2.x RAG Pipeline",
            "active_pipeline": "Haystack 2.x",
            "components": [
                "WeaviateStore",
                "DocumentSplitter",
                "OpenAIDocumentEmbedder",
                "OpenAITextEmbedder",
                "LostInTheMiddleRanker",
                "PromptBuilder",
                "OpenAIGenerator",
            ],
            "features": ["RAG", "Weaviate", "Haystack 2.x"],
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available",
        }

    def clear_documents(self) -> None:
        self.weaviate_store.delete_documents()


# Initialize pipeline
try:
    if HAYSTACK_AVAILABLE:
        rag_pipeline = HaystackRAGPipeline()
        logger.info("✅ Haystack 2.x RAG Pipeline initialized successfully")
    else:
        rag_pipeline = None
        logger.warning("⚠️ Haystack not available, using fallback")
except Exception as e:
    logger.error(f"❌ Failed to initialize Haystack pipeline: {e}")
    rag_pipeline = None


def get_pipeline_info() -> Dict[str, Any]:
    """Get pipeline information for debugging"""
    info = {
        "haystack_available": HAYSTACK_AVAILABLE,
        "pipeline_available": rag_pipeline is not None,
        "status": "unknown",
    }

    if rag_pipeline:
        try:
            pipeline_info = rag_pipeline.get_pipeline_info()
            info.update(pipeline_info)
            info["status"] = "active"
        except Exception as e:
            info["error"] = str(e)
            info["status"] = "error"
    else:
        info["status"] = "fallback"
    return info
