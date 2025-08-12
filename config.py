"""
Configuration management for RAG Chatbot
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from core.constants import *

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = DEFAULT_DB_HOST
    port: int = DEFAULT_DB_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("QDRANT_HOST", DEFAULT_DB_HOST),
            port=int(os.getenv("QDRANT_PORT", str(DEFAULT_DB_PORT))),
            collection_name=os.getenv(
                "QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME
            ),
        )


@dataclass
class ModelConfig:
    """AI Models configuration"""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_dimension: int = OPENAI_EMBEDDING_DIM

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_dimension=int(
                os.getenv("EMBEDDING_DIMENSION", str(OPENAI_EMBEDDING_DIM))
            ),
        )


@dataclass
class ProcessingConfig:
    """Document processing configuration"""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
            top_k=int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
        )


@dataclass
class AppConfig:
    """Application configuration"""

    upload_dir: str = "uploads"
    log_level: str = "INFO"
    permission_key: str = "permission_groups"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            permission_key=os.getenv("META_PERMISSION_KEY", "permission_groups"),
        )


class Config:
    """Main configuration class"""

    def __init__(self):
        # Validate OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"])

        # Load configurations
        self.database = DatabaseConfig.from_env()
        self.models = ModelConfig.from_env()
        self.processing = ProcessingConfig.from_env()
        self.app = AppConfig.from_env()

        # Create upload directory
        os.makedirs(self.app.upload_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "database": self.database.__dict__,
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "app": self.app.__dict__,
            "openai_api_key": "***" if self.openai_api_key else None,
        }


# Global config instance
config = Config()

# Backward compatibility exports
OPENAI_API_KEY = config.openai_api_key
QDRANT_HOST = config.database.host
QDRANT_PORT = config.database.port
QDRANT_COLLECTION_NAME = config.database.collection_name
EMBEDDING_MODEL = config.models.embedding_model
LLM_MODEL = config.models.llm_model
EMBEDDING_DIMENSION = config.models.embedding_dimension
CHUNK_SIZE = config.processing.chunk_size
CHUNK_OVERLAP = config.processing.chunk_overlap
TOP_K = config.processing.top_k
UPLOAD_DIR = config.app.upload_dir
LOG_LEVEL = config.app.log_level
META_PERMISSION_KEY = config.app.permission_key
