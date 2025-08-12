from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import tempfile
from ingest import ingest_document
from query import retrieve_and_answer
from utils.qdrant_store import get_qdrant_document_store
from qdrant_client import QdrantClient
from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME

app = FastAPI(
    title="RAG Chatbot API",
    description="API cho hệ thống RAG Chatbot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    permission_groups: List[str] = ["public"]
    top_k: int = 10

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentInfo(BaseModel):
    document_id: str
    source_name: str
    source_path: str
    ingestion_timestamp: str
    file_size: Optional[int]
    file_type: str
    permission_group: List[str]
    language: str
    title: Optional[str]
    author: Optional[str]
    keywords: List[str]
    page_count: Optional[int]

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

# API Endpoints

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API", "version": "1.0.0"}

@app.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    permission_groups: str = Form("public")
):
    """
    Upload và xử lý tài liệu
    """
    try:
        # Validate file type
        allowed_types = ['.pdf', '.docx', '.txt', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Parse permission groups
        permission_list = [p.strip() for p in permission_groups.split(',')]
        
        # Ingest document
        doc_id = ingest_document(tmp_file_path, permission_list)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "message": "Document uploaded and processed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Truy vấn tài liệu và trả lời câu hỏi
    """
    try:
        result = retrieve_and_answer(
            request.question, 
            request.permission_groups, 
            request.top_k
        )
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    Lấy danh sách tất cả tài liệu đã upload
    """
    try:
        # Use Haystack store to fetch documents
        ds = get_qdrant_document_store()
        docs = ds.get_all_documents(return_embedding=False)

        documents = []
        seen_docs = set()
        for doc in docs:
            meta = doc.meta or {}
            doc_meta = meta.get('document_metadata', {})
            doc_id = doc_meta.get('document_id')
            if not doc_id or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            # Map to pydantic model fields
            documents.append(DocumentInfo(
                document_id=doc_meta.get('document_id'),
                source_name=doc_meta.get('source_name'),
                source_path=doc_meta.get('source_path'),
                ingestion_timestamp=doc_meta.get('ingestion_timestamp'),
                file_size=doc_meta.get('file_size'),
                file_type=doc_meta.get('file_type'),
                permission_group=doc_meta.get('permission_group', []),
                language=doc_meta.get('language'),
                title=doc_meta.get('title'),
                author=doc_meta.get('author'),
                keywords=doc_meta.get('keywords', []),
                page_count=doc_meta.get('page_count'),
            ))
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Xóa tài liệu theo document_id
    """
    try:
        # Connect to Qdrant
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Delete by filter using typed models
        filt = QFilter(must=[
            FieldCondition(
                key="document_metadata.document_id",
                match=MatchValue(value=document_id)
            )
        ])
        client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=filt)
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex_documents():
    """
    Tái lập chỉ mục cho tất cả tài liệu
    """
    try:
        # Get document store with recreate=True
        ds = get_qdrant_document_store(recreate=True)
        
        return {
            "success": True,
            "message": "Document index recreated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái hệ thống
    """
    try:
        # Test Qdrant connection
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": [col.name for col in collections.collections]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
