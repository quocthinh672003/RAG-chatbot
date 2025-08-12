"""
ingest.py - Enhanced document ingestion using specialized Haystack pipeline
"""
from utils.converters import convert_file_to_documents
from utils.qdrant_store import get_qdrant_document_store
from utils.embeddings import embed_texts
from haystack.nodes import DocumentCleaner
from config import CHUNK_SIZE, CHUNK_OVERLAP
import uuid
from datetime import datetime

def ingest_document(file_path: str) -> str:
    """
    Enhanced document ingestion with specialized processing pipeline:
    1. Convert file using specialized Haystack converters
    2. Apply file-type specific processing (already done in converters)
    3. Clean documents
    4. Generate embeddings
    5. Store in Qdrant
    """
    
    # Generate document ID
    doc_id = str(uuid.uuid4())
    
    # 1. Convert file to documents using specialized processing pipeline
    # This already includes file-type specific processing
    documents = convert_file_to_documents(file_path)
    
    # Add document ID and timestamp to metadata
    for doc in documents:
        doc.meta["document_id"] = doc_id
        doc.meta["ingestion_timestamp"] = datetime.now().isoformat()
    
    # 2. Final cleaning step (if needed)
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_substrings=None,
        remove_regex_substrings=None,
    )
    cleaned_docs = cleaner.process(documents)
    
    # 3. Generate embeddings
    contents = [doc.content for doc in cleaned_docs]
    vectors = embed_texts(contents)
    
    # Assign embeddings to documents
    for doc, vec in zip(cleaned_docs, vectors):
        doc.embedding = vec
    
    # 4. Store in Qdrant
    ds = get_qdrant_document_store(recreate=False)
    ds.write_documents(cleaned_docs)
    
    return doc_id