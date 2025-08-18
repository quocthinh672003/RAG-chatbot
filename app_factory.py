"""
App Factory for RAG Chatbot
Centralized application initialization and service management
"""

from services.rag_pipeline import rag_pipeline
from services.image_database import ImageDatabase
from config import config


class AppFactory:
    """Application factory for managing services and initialization"""

    def __init__(self):
        self.rag_pipeline = None
        self.image_database = None
        self.config = config

    def initialize_app(self):
        """Initialize the application and all services"""
        print("🔄 Initializing services...")

        # Initialize RAG pipeline - try to get from import or create new
        try:
            from services.rag_pipeline import rag_pipeline
            self.rag_pipeline = rag_pipeline
            if self.rag_pipeline is None:
                print("⚠️ Warning: RAG pipeline not available, using fallback")
            else:
                print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: RAG pipeline initialization failed: {e}")
            self.rag_pipeline = None

        # Initialize image database
        self.image_database = ImageDatabase()

        print("✅ Services initialized")
        print("✅ Application initialized successfully")

    def get_rag_pipeline(self):
        """Get RAG pipeline instance"""
        return self.rag_pipeline

    def get_image_database(self):
        """Get image database instance"""
        return self.image_database

    def get_config(self):
        """Get configuration"""
        return self.config


# Global app factory instance
_app_factory = None


def get_app_factory():
    """Get global app factory instance"""
    global _app_factory
    if _app_factory is None:
        _app_factory = AppFactory()
        _app_factory.initialize_app()
    return _app_factory


def initialize_app():
    """Initialize the application"""
    return get_app_factory()
