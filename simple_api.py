"""
Simple API for admin functions - only list documents and health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from core.database import get_document_store
from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue

app = FastAPI(
    title="RAG Chatbot - Simple Admin API",
    description="API đơn giản để xem danh sách tài liệu và quản lý",
    version="1.0.0",
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
class DocumentInfo(BaseModel):
    document_id: str
    source_name: str
    file_size: Optional[int]
    file_type: str
    ingestion_timestamp: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int


# API Endpoints


@app.get("/")
async def root():
    return {"message": "RAG Chatbot - Simple Admin API", "version": "1.0.0"}


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    Lấy danh sách tất cả tài liệu đã upload
    """
    try:
        # Use Haystack store to fetch documents
        ds = get_document_store()
        docs = ds.get_all_documents(return_embedding=False)

        documents = []
        seen_docs = set()

        for doc in docs:
            meta = doc.meta or {}
            doc_id = meta.get("document_id")

            if not doc_id or doc_id in seen_docs:
                continue

            seen_docs.add(doc_id)

            documents.append(
                DocumentInfo(
                    document_id=doc_id,
                    source_name=meta.get("source_name", "Unknown"),
                    file_size=meta.get("file_size"),
                    file_type=meta.get("file_type", "Unknown"),
                    ingestion_timestamp=meta.get("ingestion_timestamp", "Unknown"),
                )
            )

        return DocumentListResponse(documents=documents, total_count=len(documents))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Xóa tài liệu theo document_id
    """
    try:
        # Connect to Qdrant
        ds = get_document_store()
        client = ds.get_qdrant_client()

        # Delete by filter using typed models
        filt = QFilter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        )

        delete_result = client.delete(
            collection_name=ds.collection_name, points_selector={"filter": filt}
        )

        if delete_result.status == "completed":
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document: {delete_result.status}",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái hệ thống
    """
    try:
        # Test Qdrant connection
        ds = get_document_store()
        docs = ds.get_all_documents(limit=1, return_embedding=False)

        return {
            "status": "healthy",
            "qdrant_connected": True,
            "document_count": len(ds.get_all_documents(return_embedding=False)),
        }

    except Exception as e:
        return {"status": "unhealthy", "qdrant_connected": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
