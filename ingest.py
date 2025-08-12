"""
ingest.py
Hàm ingest_document( file_path, permission_group) - parse -> chunk -> write to Qdrant
"""
from utils.parser import parse_file_to_elements
from utils.qdrant_store import get_qdrant_document_store, get_embedding_retriever
from haystack import Document
from config import CHUNK_OVERLAP, CHUNK_SIZE, META_PERMISSION_KEY
from haystack.nodes import PreProcessor
import os

def ingesst_document(file_path, pẻmission_group = None):
    """
    1. Parse file to document metadata and elements
    2. build haystack Document from elements
    3. Chunk document into smaller parts( preprocessor) if needed
    4. Write to Qdrant + generate embeddings
    returns document_id
    """

    #1 parse file to document metadata and elements
    doc_metadata, elements = parse_file_to_elements(file_path, permission_group=permission_group)
  
    #2 build haystack Document from elements
    documents = []

    for element in elements:
        if not element['content'] or not element['metadata'].get("image_path"):
            continue
        # meta: include full document_metadata and element-level metadata
        meta = {
            **element['metadata'],
            "document_metadata": doc_metadata["document_metadata"],
        }
        # promote permission group to top-level metadata key also for easy by haystack
        meta[META_PERMISSION_KEY] = doc_metadata["document_metadata"].get("permission_group", permission_group)
        documents.append(Document(
            content=element['content'] or "",
            id=element['element_id'],
            meta=meta
        ))
    
    #3 chunk document into smaller parts, using PreProcessor
    PreProcessor = PreProcessor(
        split_chunk_size=CHUNK_SIZE,
        split_chunk_overlap=CHUNK_OVERLAP,
        split_respect_sentence_boundary=True, # tránh cắt câu giữa chừng
        )
    documents = PreProcessor.process(documents)

    #4 write to Qdrant + generate embeddings
    ds  = get_qdrant_document_store(recreate=True)
    ds = write_documents(documents)

    #5 create retriever to embed documents
    retriever = get_embedding_retriever(ds)
    ds.update_embeddings(retriever=retriever)

    return doc_metadata["document_metadata"]["document_id"]