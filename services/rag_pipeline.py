"""
Haystack RAG Pipeline Service
Enhanced with PreProcessor and better pipeline structure
"""

import logging
from typing import List, Dict, Any, Optional
import os
import re
import unicodedata
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Set environment variables for Haystack
os.environ["PYDANTIC_ARBITRARY_TYPES_ALLOWED"] = "true"
os.environ["PYDANTIC_IGNORE_UNKNOWN"] = "true"

try:
    from haystack import Document
    from haystack.pipelines import Pipeline
    from haystack.nodes import (
        BM25Retriever,
        LostInTheMiddleRanker,
    )
    from haystack.nodes import PromptTemplate, PreProcessor
    from haystack.document_stores import InMemoryDocumentStore
    from core.weaviate_database import get_document_store

    HAYSTACK_AVAILABLE = True
    logger.info("✅ Haystack RAG pipeline enabled")
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logger.error(f"❌ Haystack RAG pipeline failed: {e}")

from config import config


class TextProcessor:
    """Generic text processing utilities for multilingual support"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better matching across languages"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove diacritics (accent marks) for better matching
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def create_search_variations(query: str) -> List[str]:
        """Create comprehensive search variations generically"""
        if not query:
            return []

        variations = set()

        # Original query
        variations.add(query)

        # Normalized version
        normalized = TextProcessor.normalize_text(query)
        variations.add(normalized)

        # Split into words and add individual words
        words = normalized.split()
        variations.update(words)

        # Add word combinations (bigrams, trigrams)
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                variations.add(" ".join(words[i:j]))

        # Add partial matches (prefixes)
        for word in words:
            if len(word) > 2:
                for i in range(2, len(word)):
                    variations.add(word[:i])

        return list(variations)

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts using fuzzy matching"""
        if not text1 or not text2:
            return 0.0

        # Normalize both texts
        norm1 = TextProcessor.normalize_text(text1)
        norm2 = TextProcessor.normalize_text(text2)

        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()


class HaystackRAGPipeline:
    """Enhanced Haystack-based RAG Pipeline with PreProcessor"""

    def __init__(self):
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack not available")

        self._init_haystack()
        logger.info("🎯 Enhanced Haystack RAG pipeline initialized")

    def _init_haystack(self):
        """Initialize Haystack components with PreProcessor"""
        try:
            # Initialize document store (Weaviate Cloud or fallback to InMemory)
            try:
                self.document_store = get_document_store()
                logger.info("✅ Using Weaviate Cloud document store")

                # Test if Weaviate is working properly
                try:
                    test_docs = self.document_store.get_all_documents()
                    logger.info(
                        f"✅ Weaviate test successful: {len(test_docs)} documents"
                    )
                except Exception as e:
                    logger.error(f"❌ Weaviate test failed: {e}")
                    raise Exception("Weaviate not working properly")

            except Exception as e:
                logger.warning(f"⚠️ Weaviate Cloud not available, using InMemory: {e}")
                self.document_store = InMemoryDocumentStore(
                    embedding_dim=config.models.embedding_dimension,
                    similarity="cosine",
                    use_bm25=True,
                )
                logger.info("✅ Using InMemoryDocumentStore as fallback")

            # Initialize PreProcessor for better document cleaning
            self.preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                clean_header_footer=True,
                split_by="word",
                split_length=config.processing.chunk_size,
                split_overlap=config.processing.chunk_overlap,
                split_respect_sentence_boundary=True,
                language="vi",
            )

            # Use BM25Retriever with Weaviate (fully compatible)
            self.retriever = BM25Retriever(
                document_store=self.document_store,
                top_k=config.processing.top_k,
            )
            logger.info("✅ Using BM25Retriever with Weaviate Document Store")

            # Use only LostInTheMiddleRanker to avoid transformers conflict
            self.diversity_ranker = LostInTheMiddleRanker(
                top_k=5  # Final top 5 documents
            )

            # Universal prompt template for general document Q&A
            self.prompt_template = """
            Bạn là trợ lý AI thông minh chuyên về tìm kiếm và trả lời thông tin từ tài liệu. Trả lời dựa trên thông tin trong ngữ cảnh được cung cấp.
            
            HƯỚNG DẪN TỔNG QUÁT:
            1. PHÂN TÍCH CÂU HỎI:
                - Hiểu rõ câu hỏi của người dùng
                - Xác định từ khóa chính cần tìm kiếm
                - Không giả định loại tài liệu cụ thể
            
            2. TÌM KIẾM THÔNG TIN:
                - Tìm kiếm trong tất cả nội dung được cung cấp
                - Không phân biệt loại tài liệu (JD, CV, technical, general, etc.)
                - Tập trung vào thông tin liên quan đến câu hỏi
            
            3. QUY TẮC TRẢ LỜI:
                - Trả lời chính xác dựa trên thông tin có sẵn
                - Nếu có nhiều thông tin liên quan: tổng hợp và sắp xếp logic
                - Nếu thông tin không đầy đủ: nêu rõ những gì tìm thấy
                - Nếu không tìm thấy: thông báo rõ ràng
            
            4. ĐỊNH DẠNG TRẢ LỜI:
                - Rõ ràng, có cấu trúc
                - Sử dụng bullet points khi cần
                - Trả lời bằng tiếng Việt
                - Nếu không tìm thấy: "Không tìm thấy thông tin liên quan trong tài liệu"
            
            [NGỮ CẢNH]:
            {context}
            
            [CÂU HỎI]: {query}
            
            TRẢ LỜI:
            """

            # Use OpenAI directly instead of PromptNode
            import openai

            self.openai_client = openai.OpenAI(api_key=config.openai_api_key)

            # Build pipeline
            self.pipeline = self._build_pipeline()

            logger.info("✅ Enhanced Haystack RAG components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Haystack RAG: {e}")
            raise

    def _build_pipeline(self):
        """Build enhanced Haystack RAG pipeline"""
        try:
            # Check if retriever is compatible with Haystack Pipeline
            if hasattr(self.retriever, "_component_config"):
                # Standard Haystack retriever - use Pipeline
                pipeline = Pipeline()
                pipeline.add_node(
                    component=self.retriever, name="Retriever", inputs=["Query"]
                )
                pipeline.add_node(
                    component=self.diversity_ranker,
                    name="DiversityRanker",
                    inputs=["Retriever"],
                )
                return pipeline
            else:
                # Custom retriever - return None to use direct retrieval
                logger.info("✅ Using custom retriever (no Haystack Pipeline)")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to build pipeline: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the pipeline with preprocessing"""
        if not documents:
            return

        haystack_docs = []

        for doc in documents:
            # Validate document structure
            if not isinstance(doc, dict) or "page_content" not in doc:
                continue

            # Convert to Haystack Document
            haystack_doc = Document(
                content=doc["page_content"],
                meta={
                    "source_name": doc.get("metadata", {}).get("source", "unknown"),
                    "page": doc.get("metadata", {}).get("page", 0),
                    "file_type": doc.get("metadata", {}).get("file_type", "unknown"),
                    "language": doc.get("metadata", {}).get("language", "vi"),
                },
            )
            haystack_docs.append(haystack_doc)

        if not haystack_docs:
            return

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

    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG pipeline with enhanced retrieval"""
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack RAG",
            }

        try:
            # Enhanced retrieval with debugging
            logger.info(f"🔍 Querying: '{query}'")

            # Debug: Check documents in store first
            self.debug_documents()

            # Simple query processing
            query_lower = query.lower()

            # Simple query enhancement for better retrieval
            enhanced_query = query

            # Add common terms for better matching (without bias)
            if len(query.split()) <= 3:  # Short queries need enhancement
                enhanced_query = f"{query} OR {query.lower()}"
                logger.info(f"🔍 Query enhanced: {enhanced_query}")
            else:
                logger.info(f"🔍 Using original query: {enhanced_query}")

            # Run pipeline to get documents
            if self.pipeline is None:
                # Custom retriever case (SmartRetriever)
                documents = self.retriever.retrieve(query)
                logger.info(
                    f"📄 Retrieved {len(documents)} documents with SmartRetriever"
                )
            else:
                # Standard Haystack pipeline case
                result = self.pipeline.run(query=enhanced_query)
                documents = result.get("documents", [])
                logger.info(
                    f"📄 Retrieved {len(documents)} documents with Haystack Pipeline"
                )

            # Debug: Log document contents
            for i, doc in enumerate(documents[:3]):  # Log first 3 documents
                logger.info(f"📄 Document {i+1}: {doc.content[:200]}...")
                logger.info(f"📄 Document {i+1} meta: {doc.meta}")

            # Simple document processing - keep original order
            logger.info(f"🔍 Using {len(documents)} documents in original order")

            if not documents:
                return {
                    "answer": "Không tìm thấy thông tin trong tài liệu đã cung cấp.",
                    "documents": [],
                    "sources": [],
                    "pipeline": "Haystack RAG",
                }

            # Simple context preparation - use all retrieved documents
            context = "\n\n".join([doc.content for doc in documents])
            logger.info(f"📝 Context length: {len(context)} characters")

            # Simple context truncation
            max_context_length = 50000
            if len(context) > max_context_length:
                context = context[:max_context_length]
                logger.info(f"📝 Context truncated to {len(context)} characters")

            # Use OpenAI directly for generation
            logger.info(f"🤖 Sending request to OpenAI...")
            logger.info(f"🤖 Context length: {len(context)} characters")
            logger.info(f"🤖 Query: {query}")

            try:
                formatted_prompt = self.prompt_template.format(
                    context=context, query=query
                )

                # Debug: Log the full context being sent to OpenAI
                logger.info(f"🔍 Full context being sent to OpenAI:")
                logger.info(f"🔍 {context[:1000]}...")
                logger.info(f"🔍 Context contains 'OKVIP': {'OKVIP' in context}")
                logger.info(
                    f"🔍 Context contains 'tuyển dụng': {'tuyển dụng' in context}"
                )
                logger.info(f"🔍 Context contains 'IT DEV': {'IT DEV' in context}")

                response = self.openai_client.chat.completions.create(
                    model=config.models.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": formatted_prompt,
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.2,
                    max_tokens=1200,
                )

                answer = response.choices[0].message.content
                logger.info(f"🤖 OpenAI response: {answer[:200]}...")
            except Exception as e:
                logger.error(f"❌ Error calling OpenAI: {e}")
                raise

            return {
                "answer": answer,
                "documents": documents,
                "sources": [
                    doc.meta.get("source_name", "Unknown") for doc in documents
                ],
                "pipeline": "Haystack RAG",
            }

        except Exception as e:
            logger.error(f"❌ Error in RAG query: {e}")
            return {
                "answer": f"❌ Lỗi xử lý câu hỏi: {str(e)}",
                "documents": [],
                "sources": [],
                "pipeline": "Haystack RAG",
            }

    def get_document_count(self) -> int:
        """Get number of documents in store"""
        try:
            return self.document_store.get_document_count()
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get enhanced pipeline information"""
        return {
            "document_count": self.get_document_count(),
            "pipeline_type": "Enhanced Haystack RAG Pipeline",
            "active_pipeline": "Haystack",
            "components": [
                "PreProcessor",
                "WeaviateDocumentStore",
                "BM25Retriever",
                "LostInTheMiddleRanker",
                "OpenAI Direct",
            ],
            "features": [
                "Document Preprocessing",
                "Advanced RAG",
                "Multi-stage Ranking",
                "Diversity Ranking",
                "Vietnamese Language Support",
                "OpenAI Integration",
            ],
            "ranking_stages": {
                "retriever": f"Top {config.processing.top_k} documents",
                "diversity": "Top 5 final documents",
            },
            "status": "Available" if HAYSTACK_AVAILABLE else "Not Available",
        }

    def clear_documents(self) -> None:
        """Clear all documents from the store"""
        try:
            if hasattr(self.document_store, "clear_collection"):
                # Use WeaviateDocumentStore clear_collection method
                self.document_store.clear_collection()
                logger.info("✅ Cleared all documents from Weaviate collection")
            else:
                # Fallback: Reinitialize document store to clear all documents
                self._init_haystack()
                logger.info("✅ Cleared all documents from Haystack RAG")
        except Exception as e:
            logger.error(f"❌ Error clearing documents: {e}")
            # Fallback to reinitialization
            try:
                self._init_haystack()
                logger.info("✅ Fallback: Reinitialized document store")
            except Exception as e2:
                logger.error(f"❌ Error in fallback reinitialization: {e2}")

    def debug_documents(self) -> None:
        """Debug function to check documents in store"""
        try:
            all_docs = self.document_store.get_all_documents()
            logger.info(f"📊 Total documents in store: {len(all_docs)}")

            # Show document store type
            store_type = type(self.document_store).__name__
            logger.info(f"📊 Document store type: {store_type}")

            # If it's WeaviateDocumentStore, show collection info
            if hasattr(self.document_store, "get_document_count"):
                try:
                    count = self.document_store.get_document_count()
                    logger.info(f"📊 Weaviate collection document count: {count}")
                except Exception as e:
                    logger.error(f"❌ Error getting Weaviate count: {e}")

            for i, doc in enumerate(all_docs[:5]):  # Show first 5 documents
                logger.info(f"📄 Document {i+1}:")
                logger.info(f"   Content: {doc.content[:200]}...")
                logger.info(f"   Meta: {doc.meta}")
                logger.info(f"   Length: {len(doc.content)} characters")
                logger.info("---")

        except Exception as e:
            logger.error(f"❌ Error debugging documents: {e}")


# Global pipeline instance
rag_pipeline = HaystackRAGPipeline() if HAYSTACK_AVAILABLE else None


def get_rag_pipeline():
    """Get global RAG pipeline instance"""
    global rag_pipeline
    if rag_pipeline is None and HAYSTACK_AVAILABLE:
        try:
            rag_pipeline = HaystackRAGPipeline()
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
            return None
    return rag_pipeline
