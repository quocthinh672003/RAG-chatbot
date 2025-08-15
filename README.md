# �� Hybrid RAG Chatbot với Qdrant Cloud

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

## 🎯 Ưu điểm của Hybrid Approach

### **1. Performance Tối Ưu**

- **Haystack Core**: Xử lý nhanh với pipeline tối ưu
- **Ranking Layers**: SentenceTransformers + LostInTheMiddle
- **Memory Efficient**: InMemoryDocumentStore

### **2. Độ Tin Cậy Cao**

- **Auto Fallback**: Tự động chuyển sang LangChain khi có lỗi
- **Error Handling**: Graceful degradation
- **Dependency Resilience**: Không bị phụ thuộc vào 1 framework

### **3. Flexibility**

- **Universal Converter**: UnstructuredFileConverter xử lý mọi file type
- **Configurable**: Dễ dàng thay đổi components
- **Extensible**: Dễ thêm features mới

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
pip install -r requirement.txt
```

### 4. Chạy Application

```bash
streamlit run app.py
```

## 📁 Cấu trúc Project

```
RAG-chatbot/
├── app.py                          # Main Streamlit UI
├── config.py                       # Configuration management
├── services/
│   ├── hybrid_rag_pipeline.py      # 🎯 Hybrid RAG Pipeline
│   ├── document_service.py         # Document processing
│   ├── ingest_service.py           # Document ingestion
│   ├── query_service.py            # Query processing
│   └── image_database.py           # Image extraction & management
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
├── requirement.txt                 # Dependencies
└── README.md                       # This file
```

## 🚀 Migration System (Qdrant Cloud)

### **Tại sao cần Migration?**

- **Portability**: Dữ liệu có thể truy cập từ mọi nơi
- **Scalability**: Không bị giới hạn bởi local storage
- **Management**: Web dashboard để quản lý dữ liệu
- **Backup**: Tự động backup và recovery

### **Migration Commands**

```bash
# Test connections
python migrate.py --test

# Migrate documents only
python migrate.py --type documents

# Migrate images only
python migrate.py --type images

# Migrate all (documents + images)
python migrate.py --type all

# Cleanup local files after migration
python migrate.py --type all --cleanup

# Generate migration report
python migrate.py --report
```

### **Migration Features**

#### **✅ BaseMigration Class**
- **Configuration Management**: Tự động load từ .env
- **Structured Logging**: Với timestamps và levels
- **Error Handling**: Comprehensive error tracking
- **Progress Tracking**: Success/failure counting
- **Validation**: Config validation trước khi migrate

#### **✅ DocumentMigration**
- **Multi-source Loading**: Từ processed_files.txt và chat_history.json
- **OpenAI Embeddings**: Sử dụng text-embedding-3-small
- **Metadata Preservation**: Giữ nguyên metadata gốc
- **Batch Processing**: Với progress tracking

#### **✅ ImageMigration**
- **Image Loading**: Từ image_database/image_metadata.json
- **Base64 Conversion**: Chuyển ảnh thành base64
- **Context Embedding**: Từ context và keywords
- **Cleanup Option**: Xóa ảnh local sau migration

#### **✅ MigrationManager**
- **Centralized Control**: Quản lý tất cả migrations
- **History Tracking**: Lưu lịch sử migrations
- **Report Generation**: Báo cáo chi tiết
- **Connection Testing**: Test Qdrant và OpenAI

## 🔧 Configuration

### Models Configuration

```python
# config.py
models:
  embedding_model: "text-embedding-3-small"
  llm_model: "gpt-4o-mini"
  embedding_dimension: 1536
```

### Processing Configuration

```python
processing:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 10
```

### Qdrant Cloud Configuration

```env
# .env file
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION_NAME=rag_documents
```

## 🎮 Sử dụng

### 1. Upload Documents

- Chọn file từ nhiều định dạng
- Hệ thống tự động detect và xử lý
- Hiển thị progress và kết quả

### 2. Chat với Documents

- Hỏi câu hỏi về nội dung đã upload
- Hệ thống trả lời dựa trên context
- Hiển thị sources tham khảo
- **🖼️ Hiển thị ảnh liên quan**

### 3. Monitor Pipeline

- Xem pipeline đang hoạt động (Haystack/LangChain)
- Theo dõi số lượng documents
- Kiểm tra performance

### 4. Migration to Cloud

- Migrate dữ liệu lên Qdrant Cloud
- Quản lý dữ liệu qua web dashboard
- Backup và recovery tự động

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

## 📝 Best Practices

1. **Test First**: Luôn test connections trước khi migrate
2. **Backup**: Backup dữ liệu local trước khi cleanup
3. **Monitor**: Theo dõi logs và reports
4. **Validate**: Kiểm tra dữ liệu trên Qdrant Cloud sau migration


