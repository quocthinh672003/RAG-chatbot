"""
Constants for RAG Chatbot

Mục đích:
- Định nghĩa các hằng số và cấu hình mặc định cho RAG chatbot
- Quản lý file types được hỗ trợ
- Cấu hình processing parameters
- Định nghĩa model names và error messages

Các nhóm constants:
1. File Types: Các định dạng file được hỗ trợ
2. Processing Settings: Chunk size, overlap, top-k
3. Embedding Settings: Dimensions và model names
4. Database Settings: Host, port, collection name
5. Error Messages: Các thông báo lỗi tiếng Việt
"""

# File types supported - Các định dạng file được hỗ trợ
SUPPORTED_FILE_TYPES = [
    "pdf",       # Portable Document Format
    "docx",      # Microsoft Word Document
    "txt",       # Plain Text File
    "md",        # Markdown File
    "markdown",  # Markdown File (alternative extension)
    "xlsx",      # Microsoft Excel Spreadsheet
    "xls",       # Microsoft Excel (legacy)
]

# Default processing settings - Cấu hình xử lý mặc định
DEFAULT_CHUNK_SIZE = (
    5000  # Kích thước mỗi chunk (số ký tự) - tăng để giữ nguyên cấu trúc JD
)
DEFAULT_CHUNK_OVERLAP = 1000  # Số ký tự overlap giữa các chunk
DEFAULT_TOP_K = 10  # Số lượng documents top-k để retrieve - giảm để tập trung

# Embedding dimensions - Kích thước vector embedding
OPENAI_EMBEDDING_DIM = 1536  # OpenAI embedding dimension (text-embedding-3-small)
LOCAL_EMBEDDING_DIM = 384  # Local embedding dimension (sentence-transformers)

# Database settings
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 6333
DEFAULT_COLLECTION_NAME = "rag_document"

# Model names - Tên các model được sử dụng
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # OpenAI LLM model

# Alternative embedding models
LOCAL_EMBEDDING_MODELS = {
    "sentence-transformers": "all-MiniLM-L6-v2",  # Fast, 384 dimensions
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual support
    "bge": "BAAI/bge-small-en-v1.5",  # High quality, English
    "bge-multilingual": "BAAI/bge-small-zh-v1.5",  # Multilingual BGE
}

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
