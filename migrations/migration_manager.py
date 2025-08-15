"""
Migration Manager
Centralized management for all migration operations
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from .document_migration import DocumentMigration
from .image_migration import ImageMigration


class MigrationManager:
    """
    Centralized migration manager
    
    Features:
    - Execute individual migrations
    - Execute full migration (documents + images)
    - Track migration history
    - Generate reports
    - Rollback capabilities
    """
    
    def __init__(self):
        self.migrations = {
            'documents': DocumentMigration(),
            'images': ImageMigration()
        }
        self.history = []
    
    def migrate_documents(self) -> bool:
        """Migrate documents only"""
        self.log("📄 Starting Document Migration")
        success = self.migrations['documents'].migrate()
        self._record_migration('documents', success)
        return success
    
    def migrate_images(self, cleanup_local: bool = False) -> bool:
        """Migrate images only"""
        self.log("🖼️ Starting Image Migration")
        success = self.migrations['images'].migrate()
        
        if success and cleanup_local:
            self.migrations['images'].cleanup_local_images()
        
        self._record_migration('images', success)
        return success
    
    def migrate_all(self, cleanup_local: bool = False) -> Dict[str, bool]:
        """Migrate both documents and images"""
        self.log("🚀 Starting Full Migration (Documents + Images)")
        
        results = {}
        
        # Migrate documents first
        results['documents'] = self.migrate_documents()
        
        # Migrate images
        results['images'] = self.migrate_images(cleanup_local)
        
        # Overall success
        overall_success = all(results.values())
        self.log(f"Full migration completed: {'✅ Success' if overall_success else '❌ Failed'}")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get migration status"""
        status = {
            'documents': self.migrations['documents'].get_summary(),
            'images': self.migrations['images'].get_summary(),
            'history': self.history
        }
        return status
    
    def generate_report(self) -> str:
        """Generate migration report"""
        status = self.get_status()
        
        report = f"""
# Migration Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Documents Migration
- Total: {status['documents']['total_items']}
- Successful: {status['documents']['successful']}
- Failed: {status['documents']['failed']}

## Images Migration
- Total: {status['images']['total_items']}
- Successful: {status['images']['successful']}
- Failed: {status['images']['failed']}

## Recent History
"""
        
        for entry in self.history[-5:]:  # Last 5 entries
            report += f"- {entry['timestamp']}: {entry['type']} - {'✅' if entry['success'] else '❌'}\n"
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """Save migration report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"migration_report_{timestamp}.txt"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.log(f"Report saved to: {filename}")
        return filename
    
    def _record_migration(self, migration_type: str, success: bool):
        """Record migration in history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': migration_type,
            'success': success
        }
        self.history.append(entry)
    
    def log(self, message: str):
        """Log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def test_connection(self) -> Dict[str, bool]:
        """Test connections to external services"""
        results = {}
        
        # Test Qdrant connection
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=self.migrations['documents'].config['qdrant_host'],
                port=self.migrations['documents'].config['qdrant_port'],
                api_key=self.migrations['documents'].config['qdrant_api_key']
            )
            collections = client.get_collections()
            results['qdrant'] = True
            self.log("✅ Qdrant connection successful")
        except Exception as e:
            results['qdrant'] = False
            self.log(f"❌ Qdrant connection failed: {e}")
        
        # Test OpenAI connection
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.migrations['documents'].config['openai_api_key'])
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            results['openai'] = True
            self.log("✅ OpenAI connection successful")
        except Exception as e:
            results['openai'] = False
            self.log(f"❌ OpenAI connection failed: {e}")
        
        return results
