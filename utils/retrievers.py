"""
Specialized retrievers using Haystack advanced components
"""
from typing import List, Dict, Any
from haystack import Document
from haystack.nodes import (
    # Retrievers
    SentenceWindowRetriever,
    AutoMergingRetriever,
    FilterRetriever,
    # Rankers
    TransformersSimilarityRanker,
    LostInTheMiddleRanker,
    MetaFieldRanker,
    # Document Store
    QdrantDocumentStore
)
from utils.qdrant_store import get_qdrant_document_store
from utils.embeddings import embed_query
from config import TOP_K

def get_specialized_retriever(file_type: str = None):
    """Get specialized retriever based on file type or use general retriever"""
    
    ds = get_qdrant_document_store()
    
    # Base retriever with sentence window for better context
    base_retriever = SentenceWindowRetriever(
        document_store=ds,
        embedding_model="text-embedding-3-small",  # Will be overridden by our custom embedding
        top_k=TOP_K * 2,  # Get more for re-ranking
        window_size=3,  # Include 3 sentences before and after
        window_stride=1,  # Move 1 sentence at a time
    )
    
    # Specialized retrievers for different file types
    if file_type == "csv" or file_type == "xlsx" or file_type == "xls":
        # For spreadsheets, use filter retriever to get related rows
        return FilterRetriever(
            document_store=ds,
            top_k=TOP_K,
            filters={"file_type": {"$in": [file_type]}}
        )
    
    elif file_type == "json":
        # For JSON, use auto-merging to combine related fields
        return AutoMergingRetriever(
            document_store=ds,
            embedding_model="text-embedding-3-small",
            top_k=TOP_K,
            similarity_threshold=0.7
        )
    
    elif file_type == "pdf" or file_type == "docx":
        # For structured documents, use sentence window with larger window
        return SentenceWindowRetriever(
            document_store=ds,
            embedding_model="text-embedding-3-small",
            top_k=TOP_K * 2,
            window_size=5,  # Larger window for structured docs
            window_stride=2,
        )
    
    elif file_type == "html":
        # For HTML, use filter to get content from same page
        return FilterRetriever(
            document_store=ds,
            top_k=TOP_K,
            filters={"file_type": {"$in": [file_type]}}
        )
    
    else:
        # Default: sentence window retriever
        return base_retriever

def get_advanced_ranker():
    """Get advanced ranker pipeline"""
    
    # 1. Transformers similarity ranker for re-ranking
    similarity_ranker = TransformersSimilarityRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=TOP_K,
        batch_size=32
    )
    
    # 2. Lost in the middle ranker for context optimization
    lim_ranker = LostInTheMiddleRanker(
        top_k=TOP_K,
        sort_by_score=True
    )
    
    # 3. Meta field ranker for boosting by file type relevance
    meta_ranker = MetaFieldRanker(
        top_k=TOP_K,
        field="file_type",
        weight=0.1  # Small weight to boost relevant file types
    )
    
    return [similarity_ranker, lim_ranker, meta_ranker]

def retrieve_documents(query_text: str, file_type: str = None, top_k: int = TOP_K) -> List[Document]:
    """
    Retrieve documents using specialized pipeline
    """
    
    # 1. Get specialized retriever
    retriever = get_specialized_retriever(file_type)
    
    # 2. Retrieve initial documents
    if hasattr(retriever, 'retrieve'):
        # For standard retrievers
        docs = retriever.retrieve(query=query_text, top_k=top_k * 2)
    else:
        # For custom retrievers, use query_by_embedding
        query_vec = embed_query(query_text)
        docs = retriever.document_store.query_by_embedding(
            query_embedding=query_vec,
            top_k=top_k * 2,
            return_embedding=False,
        )
    
    if not docs:
        return []
    
    # 3. Apply advanced ranking pipeline
    rankers = get_advanced_ranker()
    
    for ranker in rankers:
        try:
            ranked_result = ranker.predict(
                query=query_text,
                documents=docs
            )
            docs = ranked_result.get("documents", docs)
        except Exception as e:
            # Continue with original docs if ranker fails
            continue
    
    # 4. Return top_k documents
    return docs[:top_k]
