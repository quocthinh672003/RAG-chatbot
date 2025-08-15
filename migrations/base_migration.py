"""
Base Migration Class
Provides common functionality for all migration operations
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseMigration(ABC):
    """
    Base class for all migration operations
    
    Provides common functionality:
    - Environment configuration
    - Logging and progress tracking
    - Error handling
    - Validation
    """
    
    def __init__(self):
        """Initialize base migration"""
        self.config = self._load_config()
        self.logs = []
        self.errors = []
        self.success_count = 0
        self.total_count = 0
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            'qdrant_host': os.getenv("QDRANT_HOST"),
            'qdrant_port': int(os.getenv("QDRANT_PORT", "6333")),
            'qdrant_api_key': os.getenv("QDRANT_API_KEY"),
            'qdrant_collection': os.getenv("QDRANT_COLLECTION_NAME", "rag_documents"),
            'openai_api_key': os.getenv("OPENAI_API_KEY"),
            'upload_dir': os.getenv("UPLOAD_DIR", "uploads"),
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Add log message"""
        log_entry = {
            'timestamp': self._get_timestamp(),
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")
    
    def log_error(self, message: str, error: Optional[Exception] = None):
        """Log error message"""
        self.errors.append({
            'message': message,
            'error': str(error) if error else None,
            'timestamp': self._get_timestamp()
        })
        self.log(message, "ERROR")
    
    def log_success(self, message: str):
        """Log success message"""
        self.success_count += 1
        self.log(message, "SUCCESS")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_fields = ['qdrant_host', 'qdrant_api_key', 'openai_api_key']
        
        for field in required_fields:
            if not self.config.get(field):
                self.log_error(f"Missing required configuration: {field}")
                return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get migration summary"""
        return {
            'total_items': self.total_count,
            'successful': self.success_count,
            'failed': len(self.errors),
            'errors': self.errors,
            'logs': self.logs
        }
    
    @abstractmethod
    def migrate(self) -> bool:
        """Execute migration - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def rollback(self) -> bool:
        """Rollback migration - must be implemented by subclasses"""
        pass
