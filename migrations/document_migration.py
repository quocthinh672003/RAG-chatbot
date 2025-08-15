"""
Document Migration to Qdrant Cloud
Migrate documents from local storage to Qdrant Cloud
"""

import os
import json
from typing import List, Dict, Any
from .base_migration import BaseMigration


class DocumentMigration(BaseMigration):
    """
    Migrate documents from local storage to Qdrant Cloud
    
    Features:
    - Load documents from processed_files.txt
    - Generate embeddings using OpenAI
    - Store in Qdrant Cloud with metadata
    - Progress tracking and error handling
    """
    
    def __init__(self):
        super().__init__()
        self.documents = []
    
    def load_documents(self) -> bool:
        """Load documents from local storage"""
        try:
            # Load from processed_files.txt
            if os.path.exists("processed_files.txt"):
                with open("processed_files.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.documents.append({
                                "source": line,
                                "content": f"Document from {line}",
                                "metadata": {"source": line}
                            })
            
            # Load from chat_history.json
            if os.path.exists("chat_history.json"):
                with open("chat_history.json", "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                    for message in chat_data.get("messages", []):
                        if "source_documents" in message:
                            for doc in message["source_documents"]:
                                self.documents.append({
                                    "source": doc.get("source", "unknown"),
                                    "content": doc.get("content", ""),
                                    "metadata": doc.get("metadata", {})
                                })
            
            self.total_count = len(self.documents)
            self.log(f"Loaded {self.total_count} documents")
            return True
            
        except Exception as e:
            self.log_error("Failed to load documents", e)
            return False
    
    def migrate(self) -> bool:
        """Execute document migration"""
        self.log("🚀 Starting Document Migration to Qdrant Cloud")
        
        # Validate configuration
        if not self.validate_config():
            return False
        
        # Load documents
        if not self.load_documents():
            return False
        
        if not self.documents:
            self.log("No documents found to migrate")
            return True
        
        # Initialize Qdrant client
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            client = QdrantClient(
                host=self.config['qdrant_host'],
                port=self.config['qdrant_port'],
                api_key=self.config['qdrant_api_key']
            )
            
            self.log(f"Connected to Qdrant: {self.config['qdrant_host']}")
            
        except Exception as e:
            self.log_error("Failed to initialize Qdrant client", e)
            return False
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=self.config['openai_api_key'])
            self.log("OpenAI client initialized")
        except Exception as e:
            self.log_error("Failed to initialize OpenAI client", e)
            return False
        
        # Migrate documents
        for i, doc in enumerate(self.documents):
            try:
                self.log(f"Migrating document {i+1}/{self.total_count}: {doc['source']}")
                
                # Generate embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=doc["content"]
                )
                embedding = response.data[0].embedding
                
                # Create point for Qdrant
                point = PointStruct(
                    id=hash(doc["source"]) % (2**63),
                    vector=embedding,
                    payload={
                        'type': 'document',
                        'content': doc["content"],
                        'metadata': doc["metadata"],
                        'source': doc["source"]
                    }
                )
                
                # Add to Qdrant
                client.upsert(
                    collection_name=self.config['qdrant_collection'],
                    points=[point]
                )
                
                self.log_success(f"Migrated: {doc['source']}")
                
            except Exception as e:
                self.log_error(f"Failed to migrate {doc['source']}", e)
        
        # Print summary
        summary = self.get_summary()
        self.log(f"Migration completed: {summary['successful']}/{summary['total_items']} successful")
        
        return summary['successful'] > 0
    
    def rollback(self) -> bool:
        """Rollback document migration"""
        self.log("🔄 Document migration rollback not implemented")
        return False
