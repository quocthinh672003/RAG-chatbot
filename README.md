# 🤖 Hybrid RAG Chatbot với Qdrant Cloud

**Hybrid RAG Pipeline** - Giải pháp tối ưu performance và độ tin cậy cao, tích hợp Qdrant Cloud cho data persistence.

## 🚀 Tính năng

- **🔧 Hybrid Architecture**: Haystack làm core, LangChain làm fallback
- **📚 Multi-format Support**: PDF, DOCX, TXT, MD, XLSX, XLS, CSV, HTML, JSON
- **⚡ Auto Fallback**: Tự động chuyển sang LangChain khi Haystack có vấn đề
- **🎯 Smart Retrieval**: Embedding + Ranking + Diversity
- **💬 Chat Interface**: UI thân thiện giống ChatGPT
- **📊 Real-time Stats**: Hiển thị pipeline đang hoạt động
- **🖼️ Image Support**: Trích xuất và hiển thị ảnh từ tài liệu
- **☁️ Cloud Storage**: Qdrant Cloud cho data persistence và portability
- **🔍 Smart File Management**: Search, pagination, file type icons

## 🏗️ Kiến trúc Hybrid

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid RAG Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  🎯 Primary: Haystack Core                                  │
│  ├── UnstructuredFileConverter (Universal)                 │
│  ├── PreProcessor (Cleaning + Splitting)                   │
│  ├── InMemoryDocumentStore                                 │
│  ├── EmbeddingRetriever (OpenAI)                           │
│  ├── SentenceTransformersRanker                            │
│  ├── LostInTheMiddleRanker                                 │
│  └── PromptNode (OpenAI GPT)                               │
├─────────────────────────────────────────────────────────────┤
│  🔄 Fallback: LangChain                                    │
│  ├── Document Loaders (PDF, DOCX, TXT)                     │
│  ├── RecursiveCharacterTextSplitter                        │
│  ├── FAISS Vector Store                                    │
│  ├── OpenAI Embeddings                                     │
│  └── LLMChain (OpenAI GPT)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Logic Code và Luồng Đi Chi Tiết

### **🔄 Main Application Flow (app.py)**

#### **1. Initialization Phase**

```python
def main():
    """
    🚀 Main application function - Khởi tạo và quản lý toàn bộ ứng dụng

    Luồng đi:
    1. Initialize services với caching (@st.cache_resource)
    2. Download NLTK data một lần duy nhất
    3. Load chat history từ file JSON
    4. Auto-reload documents từ uploads folder
    5. Render UI components
    """
```

#### **2. Service Initialization**

```python
@st.cache_resource
def initialize_services():
    """
    🔧 Initialize services với caching cho performance tối ưu

    Luồng đi:
    1. Gọi initialize_app() từ app_factory
    2. Lấy AppFactory instance
    3. Cache result để tránh re-initialization
    4. Return factory để access services
    """
```

#### **3. Chat History Management**

```python
@lru_cache(maxsize=1)
def load_chat_history() -> List[Dict[str, Any]]:
    """
    📚 Load chat history từ JSON file với caching

    Luồng đi:
    1. Check file tồn tại
    2. Load JSON với UTF-8 encoding
    3. Cache result để tránh re-reading
    4. Return empty list nếu file không tồn tại
    """

def save_chat_history(chat_history: List[Dict[str, Any]]) -> None:
    """
    💾 Save chat history vào JSON file

    Luồng đi:
    1. Write JSON với UTF-8 encoding
    2. Clear cache để force reload
    3. Log success/failure
    """
```

#### **4. Document Auto-Reload**

```python
def auto_reload_documents(rag_pipeline, image_database) -> None:
    """
    🔄 Auto-reload documents từ uploads folder

    Luồng đi:
    1. Get files từ uploads directory
    2. Check session state để tránh duplicate processing
    3. Process từng file:
       - Convert với DocumentService
       - Add vào RAG pipeline
       - Extract images
       - Add vào processed_files list
    4. Log results
    """
```

#### **5. File Upload Processing**

```python
def process_uploaded_files_old(uploaded_files, rag_pipeline, image_database) -> None:
    """
    📁 Process uploaded files với error handling

    Luồng đi:
    1. Initialize processed_files và failed_files lists
    2. Process từng file:
       - Save file to disk
       - Convert với DocumentService
       - Add vào RAG pipeline
       - Extract images
       - Update session state
    3. Display results và errors
    """
```

#### **6. Chat Input Processing**

```python
def process_chat_input_old(prompt, rag_pipeline, image_database):
    """
    💬 Process chat input với Hybrid RAG

    Luồng đi:
    1. Display user message
    2. Query RAG pipeline
    3. Display AI answer
    4. Find relevant images:
       - Extract source files từ documents
       - Search images by source file
       - Fallback to query-based search
    5. Display images với download buttons
    6. Show sources
    7. Save to chat history
    """
```

#### **7. Smart File List Management**

```python
# Trong main() function - Sidebar section
"""
📋 Smart File List với Search và Pagination

Luồng đi:
1. Check processed_files trong session state
2. Add search input với placeholder
3. Filter files based on search term
4. Show file count (filtered/total)
5. Display files với pagination:
   - Limit to 10 files initially
   - Show "Xem thêm" button nếu cần
   - Add file type icons
   - Truncate long filenames
6. Handle "Show All" và "Thu gọn" buttons
"""
```

### **🔧 File Processing Logic**

#### **Document Service Flow**

```python
# services/document_service.py
class DocumentService:
    def convert_file(self, file_path: str) -> List[Document]:
        """
        🔄 Convert file thành Documents

        Luồng đi:
        1. Detect file type từ extension
        2. Use appropriate converter:
           - PDF: PyPDF2Loader
           - DOCX: Docx2txtLoader
           - TXT: TextLoader
           - MD: UnstructuredMarkdownLoader
           - XLSX: UnstructuredExcelLoader
        3. Load documents
        4. Add metadata (source, timestamp)
        5. Return list of Documents
        """
```

#### **Image Extraction Flow**

```python
# services/image_database.py
class ImageDatabase:
    def extract_images_from_any_file(self, file_path: str, filename: str) -> List[Dict]:
        """
        🖼️ Extract images từ bất kỳ file type nào

        Luồng đi:
        1. Detect file type
        2. Use appropriate extractor:
           - PDF: PyMuPDF (fitz)
           - DOCX: python-docx
           - XLSX: openpyxl
        3. Extract images với context
        4. Save to local storage
        5. Update metadata
        6. Return image info
        """
```

### **🎯 Hybrid RAG Pipeline Logic**

#### **Primary Flow (Haystack)**

```python
# services/hybrid_rag_pipeline.py
class HybridRAGPipeline:
    def query(self, query: str) -> Dict[str, Any]:
        """
        🎯 Query với Hybrid RAG Pipeline

        Luồng đi:
        1. Try Haystack pipeline first:
           - EmbeddingRetriever → SentenceTransformersRanker → LostInTheMiddleRanker → PromptNode
        2. If Haystack fails, fallback to LangChain:
           - FAISS VectorStore → SimilaritySearch → LLMChain
        3. Return unified result format
        """
```

#### **Fallback Logic**

```python
def query_with_fallback(self, query: str) -> Dict[str, Any]:
    """
    🔄 Fallback logic cho reliability

    Luồng đi:
    1. Try Haystack pipeline
    2. Catch any exception
    3. Log fallback reason
    4. Try LangChain pipeline
    5. Return result hoặc raise error
    """
```

### **📊 Session State Management**

#### **Key Session Variables**

```python
# Session State Structure
st.session_state = {
    "chat_history": [],              # Chat messages
    "processed_files": [],           # Uploaded files
    "auto_reloaded": False,          # Auto-reload flag
    "force_show_upload": False,      # Show upload area
    "show_all_files": False,         # File list pagination
    "last_displayed_images": [],     # Last shown images
    "file_search": ""                # File search term
}
```

#### **State Persistence**

```python
# Chat History Persistence
"""
💾 Chat History được lưu vào file JSON:
- Load khi app khởi động
- Save sau mỗi interaction
- Cache để performance
- UTF-8 encoding cho Vietnamese
"""

# File List Persistence
"""
📁 File List được lưu trong session state:
- Auto-reload từ uploads folder
- Persist qua app restarts
- Search và filter real-time
- Pagination cho performance
"""
```

## 🛠️ Cài đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd RAG-chatbot
```

### 2. Tạo Environment File

```bash
# Tạo file .env
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Cloud Configuration (Optional)
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION_NAME=rag_documents
```

### 3. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Chạy Application

```bash
streamlit run app.py
```

## 📁 Cấu trúc Project

```
RAG-chatbot/
├── app.py                          # 🚀 Main Streamlit UI với logic chi tiết
├── config.py                       # Configuration management
├── app_factory.py                  # 🔧 Service factory pattern
├── services/
│   ├── hybrid_rag_pipeline.py      # 🎯 Hybrid RAG Pipeline
│   ├── document_service.py         # 📄 Document processing
│   ├── ingest_service.py           # 📥 Document ingestion
│   ├── query_service.py            # 🔍 Query processing
│   └── image_database.py           # 🖼️ Image extraction & management
├── core/
│   ├── constants.py                # Constants
│   └── database.py                 # Database utilities
├── migrations/                     # 🚀 Migration System
│   ├── __init__.py                 # Package initialization
│   ├── base_migration.py           # Base migration class
│   ├── document_migration.py       # Document migration
│   ├── image_migration.py          # Image migration
│   └── migration_manager.py        # Migration manager
├── utils/
│   └── helpers.py                  # Utility functions
├── migrate.py                      # Migration script
├── requirements.txt                # Dependencies với comments
├── uploads/                        # 📁 Uploaded files storage
├── image_database/                 # 🖼️ Extracted images storage
├── chat_history.json               # 💬 Chat history persistence
└── README.md                       # This file
```

## 🎯 Ưu điểm của Hybrid Approach

### **1. Performance Tối Ưu**

- **Haystack Core**: Xử lý nhanh với pipeline tối ưu
- **Ranking Layers**: SentenceTransformers + LostInTheMiddle
- **Memory Efficient**: InMemoryDocumentStore
- **Caching**: @st.cache_resource và @lru_cache

### **2. Độ Tin Cậy Cao**

- **Auto Fallback**: Tự động chuyển sang LangChain khi có lỗi
- **Error Handling**: Graceful degradation
- **Dependency Resilience**: Không bị phụ thuộc vào 1 framework
- **Session Persistence**: Chat history và file list được lưu

### **3. Flexibility**

- **Universal Converter**: UnstructuredFileConverter xử lý mọi file type
- **Configurable**: Dễ dàng thay đổi components
- **Extensible**: Dễ thêm features mới
- **Smart UI**: Search, pagination, file type icons

## 🔄 Hybrid Pipeline Logic

### **Primary Flow (Haystack)**

```python
# 1. Document Processing
UnstructuredFileConverter → PreProcessor → InMemoryDocumentStore

# 2. Retrieval Pipeline
Query → EmbeddingRetriever → SentenceTransformersRanker → LostInTheMiddleRanker → PromptNode
```

### **Fallback Flow (LangChain)**

```python
# 1. Document Processing
DocumentLoader → RecursiveCharacterTextSplitter → FAISS VectorStore

# 2. Retrieval Pipeline
Query → SimilaritySearch → LLMChain
```

### **Auto Switch Logic**

```python
try:
    # Try Haystack first
    haystack_result = haystack_pipeline.query(query)
    return haystack_result
except Exception:
    # Fallback to LangChain
    langchain_result = langchain_pipeline.query(query)
    return langchain_result
```

## 📊 Performance Metrics

### **Haystack Core**

- **Speed**: ⚡⚡⚡⚡⚡ (Very Fast)
- **Memory**: 💾💾💾 (Efficient)
- **Features**: 🎯🎯🎯🎯🎯 (Full-featured)

### **LangChain Fallback**

- **Speed**: ⚡⚡⚡⚡ (Fast)
- **Memory**: 💾💾💾💾 (Good)
- **Features**: 🎯🎯🎯🎯 (Good)

### **Qdrant Cloud**

- **Storage**: 💾💾💾💾💾 (Unlimited)
- **Speed**: ⚡⚡⚡⚡⚡ (Very Fast)
- **Reliability**: 🔒🔒🔒🔒🔒 (High)

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py
```

### Production

```bash
# Docker (nếu cần)
docker-compose up -d

# Hoặc direct
streamlit run app.py --server.port 8501
```

## 🔧 Troubleshooting

### **Haystack Import Error**

- Hệ thống tự động chuyển sang LangChain
- Không cần manual intervention

### **API Key Issues**

- Kiểm tra `.env` file
- Đảm bảo `OPENAI_API_KEY` đúng format

### **Memory Issues**

- Giảm `chunk_size` trong config
- Sử dụng ít documents hơn

### **File List Issues**

- Check `uploads/` directory
- Verify file permissions
- Clear session state nếu cần

### **Migration Issues**

```bash
# Test connections first
python migrate.py --test

# Check specific migration
python migrate.py --type documents

# Generate detailed report
python migrate.py --report
```

## 📈 Migration Reports

### Migration Report Format

```
# Migration Report
Generated: 2024-01-15 14:30:25

## Documents Migration
- Total: 15
- Successful: 15
- Failed: 0

## Images Migration
- Total: 20
- Successful: 20
- Failed: 0

## Recent History
- 2024-01-15T14:30:25: documents - ✅
- 2024-01-15T14:35:10: images - ✅
```

## 🎯 Next Steps

1. **Integrate with RAG Pipeline**: Cập nhật RAG pipeline để sử dụng Qdrant Cloud
2. **Add Rollback**: Implement rollback functionality
3. **Incremental Migration**: Support incremental updates
4. **Monitoring**: Add monitoring và alerting
5. **Advanced Search**: Implement semantic search cho file list

## 📝 Best Practices

1. **Test First**: Luôn test connections trước khi migrate
2. **Backup**: Backup dữ liệu local trước khi cleanup
3. **Monitor**: Theo dõi logs và reports
4. **Validate**: Kiểm tra dữ liệu trên Qdrant Cloud sau migration
5. **Cache**: Sử dụng caching cho performance
6. **Error Handling**: Implement comprehensive error handling
7. **Session Management**: Quản lý session state cẩn thận
