"""
queryt.py
 def retrieve_and_answer(query_text, user_permission_group)
 flow:
 - retriever (haystack embeddingRetriever) -> initial candidates
 - ranker (TransformersRanker / cross-encoder) -> rerank 
 - langchain (LLM) -> answer bằng cách using top-k reranked documents as context
"""
from utils.qdrant_store import get_qdrant_document_store, get_embedding_retriever
from haystack import Document
from haystack.nodes import EmbeddingRetriever, TransformersRanker
from config import META_PERMISSION_KEY, RANKER_MODEL, TOP_K
from langchain.llms import OpenAI
from haystack.pipelines import Pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import OpenAI as LangchainOpenAI

def retrieve_and_answer(query_text, user_permission_group, top_k=TOP_K):
    """
    user_permission_group: list of groups( e.g.[finance_team])
    returns dict: { "answer": str, "sources": List[Document] }
    """
    # setup store + retriever
    ds = get_qdrant_document_store()
    retriever = get_embedding_retriever(ds)

    # haystack retriever with permission filter
    # haystack filter uses dict with meta key -> values. We will use simple matching (any of groups match)
    # filter = {META_PERMISSION_KEY: {"$in": user_permission_group}}
    # build filters: permission group must match any of the user's groups

    filter = {META_PERMISSION_KEY: {"$in": user_permission_group}}

    candidate_docs = retriever.retrieve(
        query=query_text,
        top_k=top_k,
        filters=filter
    )
    if not candidate_docs: 
        return {"answer": "No relevant documents found.", "sources": []}
    
    #ranker using cross-encoder
    ranker = TransformersRanker(model_name=RANKER_MODEL)
    reranked = ranker.predict(
        query=query_text,
        documents=candidate_docs
        top_k=min(top_k, len(candidate_docs))
    )
    top_docs = reranked["documents"] if "documents" in reranked else candidate_docs
    # if top doc are images( metadata has image_path), and used ask image, return images
    image_results = [d for d in top_docs if d.meta.get]