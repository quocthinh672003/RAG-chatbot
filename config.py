import os
from dotenv import load_dotenv

load_dotenv()
# Load environment variables from .env file
# or from the system environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in the .env file or as an environment variable.")
# Ensure that the OPENAI_API_KEY is set before proceeding

#Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_document")

# Ensure that the Qdrant host and port are set correctly

# models - using OpenAI models as specified
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
RANKER_MODEL = os.getenv("RANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Ensure that the models are set correctly

# haystack settings
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))  # Dimension for text-embedding-3-small
TOP_K = int(os.getenv("TOP_K", 10))  # Default number of top documents to retrieve

# chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))  # Default chunk size in characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))  # Default overlap in characters
# Ensure that the chunk size and overlap are set correctly

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Default log level
# Ensure that the log level is set correctly

# metadata key for permission group
META_PERMISSION_KEY = "permission_groups"

# Upload dir
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)