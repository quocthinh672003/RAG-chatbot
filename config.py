"""
Configuration management for RAG Chatbot
Enhanced with environment-specific configs and validation
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from core.constants import *

# Set pydantic config to allow arbitrary types
os.environ["PYDANTIC_ARBITRARY_TYPES_ALLOWED"] = "true"
os.environ["PYDANTIC_IGNORE_UNKNOWN"] = "true"

# Disable pandas to avoid pydantic conflicts
os.environ["PYDANTIC_DISABLE_PANDAS"] = "true"

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = DEFAULT_DB_HOST
    port: int = DEFAULT_DB_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME
    api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    # Weaviate configuration
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str = "Documents"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("QDRANT_HOST", DEFAULT_DB_HOST),
            port=int(os.getenv("QDRANT_PORT", str(DEFAULT_DB_PORT))),
            collection_name=os.getenv(
                "QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME
            ),
            api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL")
            or f"https://{os.getenv('QDRANT_HOST', '')}",
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            # Weaviate config
            weaviate_url=os.getenv("WEAVIATE_URL"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_class_name=os.getenv("WEAVIATE_CLASS_NAME", "Documents"),
        )

    def validate(self) -> bool:
        """Validate database configuration"""
        if not self.host:
            raise ValueError("Database host is required")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Invalid database port")
        return True


@dataclass
class ModelConfig:
    """AI Models configuration"""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_dimension: int = OPENAI_EMBEDDING_DIM
    temperature: float = 0.7
    max_tokens: int = 1000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_dimension=int(
                os.getenv("EMBEDDING_DIMENSION", str(OPENAI_EMBEDDING_DIM))
            ),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        )

    def validate(self) -> bool:
        """Validate model configuration"""
        if not self.embedding_model:
            raise ValueError("Embedding model is required")
        if not self.llm_model:
            raise ValueError("LLM model is required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return True


@dataclass
class ProcessingConfig:
    """Document processing configuration"""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list = field(default_factory=lambda: SUPPORTED_FILE_TYPES)

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
            top_k=int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024))),
        )

    def validate(self) -> bool:
        """Validate processing configuration"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")
        return True


@dataclass
class AppConfig:
    """Application configuration"""

    upload_dir: str = "uploads"
    log_level: str = "INFO"
    permission_key: str = "permission_groups"

    session_timeout: int = 3600  # 1 hour

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            permission_key=os.getenv("META_PERMISSION_KEY", "permission_groups"),
    
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600")),
        )

    def validate(self) -> bool:
        """Validate app configuration"""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)
        return True


@dataclass
class ImageConfig:
    """Image processing configuration"""

    max_image_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: list = field(
        default_factory=lambda: ["png", "jpg", "jpeg", "gif", "bmp"]
    )
    quality: int = 85
    max_width: int = 1920
    max_height: int = 1080

    @classmethod
    def from_env(cls) -> "ImageConfig":
        return cls(
            max_image_size=int(os.getenv("MAX_IMAGE_SIZE", str(10 * 1024 * 1024))),
            quality=int(os.getenv("IMAGE_QUALITY", "85")),
            max_width=int(os.getenv("MAX_IMAGE_WIDTH", "1920")),
            max_height=int(os.getenv("MAX_IMAGE_HEIGHT", "1080")),
        )

    def validate(self) -> bool:
        """Validate image configuration"""
        if self.max_image_size <= 0:
            raise ValueError("Max image size must be positive")
        if self.quality < 1 or self.quality > 100:
            raise ValueError("Image quality must be between 1 and 100")
        return True


class Config:
    """Main configuration class with validation"""

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
        self.image = ImageConfig.from_env()

        # Validate all configurations
        self._validate_all()

    def _validate_all(self):
        """Validate all configurations"""
        try:
            self.database.validate()
            self.models.validate()
            self.processing.validate()
            self.app.validate()
            self.image.validate()
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def get_environment(self) -> str:
        """Get current environment"""
        return os.getenv("ENVIRONMENT", "development")

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.get_environment() == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.get_environment() == "development"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "collection_name": self.database.collection_name,
            },
            "models": {
                "embedding_model": self.models.embedding_model,
                "llm_model": self.models.llm_model,
                "embedding_dimension": self.models.embedding_dimension,
                "temperature": self.models.temperature,
            },
            "processing": {
                "chunk_size": self.processing.chunk_size,
                "chunk_overlap": self.processing.chunk_overlap,
                "top_k": self.processing.top_k,
            },
            "app": {
                "upload_dir": self.app.upload_dir,
                "log_level": self.app.log_level,
    
            },
            "image": {
                "max_image_size": self.image.max_image_size,
                "quality": self.image.quality,
            },
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
# Weaviate exports
WEAVIATE_URL = config.database.weaviate_url
WEAVIATE_API_KEY = config.database.weaviate_api_key
WEAVIATE_CLASS_NAME = config.database.weaviate_class_name
