"""
Constants for RAG Chatbot
"""

# File types supported
SUPPORTED_FILE_TYPES = [
    "pdf",
    "docx",
    "txt",
    "md",
    "markdown",
    "xlsx",
    "xls",
    "pptx",
    "html",
    "htm",
    "json",
    "csv",
]

# Default processing settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 10

# Embedding dimensions
OPENAI_EMBEDDING_DIM = 1536
LOCAL_EMBEDDING_DIM = 384

# Database settings
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 6333
DEFAULT_COLLECTION_NAME = "rag_document"

# Model names
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Processing separators
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", " ", ""]

# Error messages
ERROR_MESSAGES = {
    "file_not_found": "Không tìm thấy file",
    "conversion_failed": "Lỗi chuyển đổi file",
    "embedding_failed": "Lỗi tạo embedding",
    "no_documents": "Không tìm thấy tài liệu liên quan",
    "api_key_missing": "Thiếu OpenAI API key",
}
