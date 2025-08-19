"""
App Factory for RAG Chatbot
Centralized application initialization and service management
"""

import streamlit as st
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
        
        # Initialize RAG pipeline
        self.rag_pipeline = rag_pipeline
        if self.rag_pipeline is None:
            pass  # Silent fallback
        
        # Initialize image database
        self.image_database = ImageDatabase()
    
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
