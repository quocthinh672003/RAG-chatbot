"""
Database service
"""

from typing import List, Dict, Any
from haystack import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from config import config


class QdrantDocumentStore:
    """Qdrant document store with persistence"""
    
    def __init__(self):
        self.client = QdrantClient(
            host=config.database.host,
            port=config.database.port
        )
        self.collection_name = config.database.collection_name
        self.embedding_dim = config.models.embedding_dimension
        
        # Create collection if not exists
        self._create_collection()
    
    def _create_collection(self):
        """Create collection if not exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"✅ Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
    
    def write_documents(self, documents: List[Document]):
        """Add documents to store"""
        points = []
        for doc in documents:
            if hasattr(doc, 'embedding') and doc.embedding is not None:
                points.append(PointStruct(
                    id=hash(doc.id) % (2**63),  # Qdrant requires int64
                    vector=doc.embedding.tolist() if hasattr(doc.embedding, 'tolist') else doc.embedding,
                    payload={
                        'content': doc.content,
                        'meta': doc.meta,
                        'id': doc.id
                    }
                ))
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Added {len(points)} documents to Qdrant")
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust as needed
            )
            
            documents = []
            for point in result[0]:
                doc = Document(
                    content=point.payload['content'],
                    meta=point.payload['meta'],
                    id=point.payload['id']
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        try:
            point_ids = [hash(doc_id) % (2**63) for doc_id in document_ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            print(f"✅ Deleted {len(document_ids)} documents")
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")


def get_document_store():
    """Get document store"""
    try:
        return QdrantDocumentStore()
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("🔄 Falling back to in-memory store...")
        
        # Fallback to simple in-memory store
        class SimpleDocumentStore:
            def __init__(self):
                self.documents: List[Document] = []
            
            def write_documents(self, documents: List[Document]):
                self.documents.extend(documents)
            
            def get_all_documents(self) -> List[Document]:
                return self.documents
            
            def delete_documents(self, document_ids: List[str]):
                self.documents = [doc for doc in self.documents if doc.id not in document_ids]
        
        return SimpleDocumentStore()
