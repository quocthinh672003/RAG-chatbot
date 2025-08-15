"""
App Factory for RAG Chatbot
Centralized application initialization and service management
"""

import streamlit as st
from services.hybrid_rag_pipeline import get_rag_pipeline
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
        
        # Initialize RAG pipeline
        self.rag_pipeline = get_rag_pipeline()
        
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
