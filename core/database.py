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
        # Handle both URL and host+port formats
        if config.database.host.startswith(("http://", "https://")):
            # Use URL format for Qdrant Cloud
            self.client = QdrantClient(
                url=config.database.host, api_key=config.database.api_key
            )
        else:
            # Use host+port format for local Qdrant
            self.client = QdrantClient(
                host=config.database.host, port=config.database.port
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
                        size=self.embedding_dim, distance=Distance.COSINE
                    ),
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
            # Create a simple embedding if none exists (for text-only search)
            if not hasattr(doc, "embedding") or doc.embedding is None:
                # Create a simple hash-based embedding for text-only documents
                import hashlib

                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                simple_embedding = [
                    float(int(content_hash[i : i + 2], 16)) / 255.0
                    for i in range(0, min(len(content_hash), self.embedding_dim * 2), 2)
                ]
                # Pad or truncate to match embedding dimension
                while len(simple_embedding) < self.embedding_dim:
                    simple_embedding.extend(
                        simple_embedding[: self.embedding_dim - len(simple_embedding)]
                    )
                simple_embedding = simple_embedding[: self.embedding_dim]
                doc.embedding = simple_embedding

            points.append(
                PointStruct(
                    id=hash(doc.id) % (2**63),  # Qdrant requires int64
                    vector=(
                        doc.embedding.tolist()
                        if hasattr(doc.embedding, "tolist")
                        else doc.embedding
                    ),
                    payload={"content": doc.content, "meta": doc.meta, "id": doc.id},
                )
            )

        if points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"✅ Added {len(points)} documents to Qdrant")
            except Exception as e:
                print(f"❌ Error adding documents to Qdrant: {e}")
                raise

    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name, limit=10000  # Adjust as needed
            )

            documents = []
            for point in result[0]:
                try:
                    # Handle missing meta key
                    meta = point.payload.get("meta", {})
                    if meta is None:
                        meta = {}

                    doc = Document(
                        content=point.payload.get("content", ""),
                        meta=meta,
                        id=point.payload.get("id", str(point.id)),
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"❌ Error processing document point: {e}")
                    continue

            print(f"✅ Retrieved {len(documents)} documents from Qdrant")
            return documents
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []

    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        try:
            point_ids = [hash(doc_id) % (2**63) for doc_id in document_ids]
            self.client.delete(
                collection_name=self.collection_name, points_selector=point_ids
            )
            print(f"✅ Deleted {len(document_ids)} documents")
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")

    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            # Get all point IDs
            result = self.client.scroll(
                collection_name=self.collection_name, limit=10000
            )
            point_ids = [point.id for point in result[0]]

            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name, points_selector=point_ids
                )
                print(f"✅ Cleared {len(point_ids)} documents from collection")
            else:
                print("✅ Collection is already empty")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")

    def get_document_count(self) -> int:
        """Get number of documents in collection"""
        try:
            result = self.client.scroll(collection_name=self.collection_name, limit=1)
            # Get total count from collection info
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"❌ Error getting document count: {e}")
            return 0


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
                self.documents = [
                    doc for doc in self.documents if doc.id not in document_ids
                ]

        return SimpleDocumentStore()
