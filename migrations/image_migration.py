"""
Image Migration to Qdrant Cloud
Migrate images from local storage to Qdrant Cloud
"""

import os
import json
import base64
from typing import List, Dict, Any
from .base_migration import BaseMigration


class ImageMigration(BaseMigration):
    """
    Migrate images from local storage to Qdrant Cloud
    
    Features:
    - Load images from image_database
    - Convert to base64 for storage
    - Generate embeddings from context
    - Store in Qdrant Cloud with metadata
    - Progress tracking and error handling
    """
    
    def __init__(self):
        super().__init__()
        self.images = []
    
    def load_images(self) -> bool:
        """Load images from local database"""
        try:
            metadata_file = "image_database/image_metadata.json"
            
            if not os.path.exists(metadata_file):
                self.log("No image metadata file found")
                return True
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Convert metadata to list format
            for image_id, image_data in metadata.items():
                image_data['id'] = image_id
                self.images.append(image_data)
            
            self.total_count = len(self.images)
            self.log(f"Loaded {self.total_count} images from metadata")
            return True
            
        except Exception as e:
            self.log_error("Failed to load images", e)
            return False
    
    def migrate(self) -> bool:
        """Execute image migration"""
        self.log("🖼️ Starting Image Migration to Qdrant Cloud")
        
        # Validate configuration
        if not self.validate_config():
            return False
        
        # Load images
        if not self.load_images():
            return False
        
        if not self.images:
            self.log("No images found to migrate")
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
        
        # Migrate images
        for i, image_data in enumerate(self.images):
            try:
                image_path = image_data.get('path', '')
                self.log(f"Migrating image {i+1}/{self.total_count}: {os.path.basename(image_path)}")
                
                # Check if image file exists
                if not os.path.exists(image_path):
                    self.log(f"Image file not found: {image_path}")
                    continue
                
                # Read image file and convert to base64
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Get image context for embedding
                context = image_data.get('context', '')
                keywords = image_data.get('keywords', [])
                source_file = image_data.get('source_file', '')
                
                # Combine context and keywords for embedding
                text_for_embedding = f"{context} {' '.join(keywords)} {source_file}"
                
                # Generate embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text_for_embedding
                )
                embedding = response.data[0].embedding
                
                # Create point for Qdrant
                point = PointStruct(
                    id=hash(image_path) % (2**63),
                    vector=embedding,
                    payload={
                        'type': 'image',
                        'content': image_base64,
                        'metadata': {
                            'path': image_path,
                            'context': context,
                            'keywords': keywords,
                            'source_file': source_file,
                            'image_type': image_data.get('type', 'unknown'),
                            'file_size': len(image_bytes)
                        }
                    }
                )
                
                # Add to Qdrant
                client.upsert(
                    collection_name=self.config['qdrant_collection'],
                    points=[point]
                )
                
                self.log_success(f"Migrated: {os.path.basename(image_path)} ({len(image_bytes)} bytes)")
                
            except Exception as e:
                self.log_error(f"Failed to migrate {image_path}", e)
        
        # Print summary
        summary = self.get_summary()
        self.log(f"Migration completed: {summary['successful']}/{summary['total_items']} successful")
        
        return summary['successful'] > 0
    
    def rollback(self) -> bool:
        """Rollback image migration"""
        self.log("🔄 Image migration rollback not implemented")
        return False
    
    def cleanup_local_images(self) -> bool:
        """Clean up local images after successful migration"""
        try:
            import shutil
            if os.path.exists("image_database"):
                shutil.rmtree("image_database")
                self.log("Local images deleted")
                return True
            else:
                self.log("No local image database found")
                return False
        except Exception as e:
            self.log_error("Failed to delete local images", e)
            return False
