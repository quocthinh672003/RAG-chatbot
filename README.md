# 🤖 RAG Chatbot với Weaviate Vector Store

**RAG Pipeline tối ưu** - Giải pháp chatbot thông minh sử dụng Haystack 2.x và Weaviate Cloud cho vector search.

## 🚀 Tính năng

- **🔧 Haystack 2.x Pipeline**: RAG pipeline hiện đại với vector search
- **📚 Multi-format Support**: PDF, DOCX, TXT, MD, XLSX, CSV, HTML, JSON
- **⚡ Weaviate Vector Store**: Cloud-based vector database cho performance cao
- **🎯 Smart Retrieval**: OpenAI embeddings + LostInTheMiddleRanker
- **💬 Chat Interface**: UI thân thiện giống ChatGPT
- **📊 Real-time Stats**: Hiển thị pipeline đang hoạt động
- **🖼️ Image Support**: Trích xuất và hiển thị ảnh từ tài liệu
- **☁️ Cloud Storage**: Weaviate Cloud cho data persistence
- **🔍 Smart File Management**: Search, pagination, file type icons

## 🏗️ Kiến trúc RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Architecture                │
├─────────────────────────────────────────────────────────────┤
│  📁 Document Processing                                    │
│  ├── DocumentService (Multi-format support)               │
│  ├── DocumentSplitter (Word-based chunking)               │
│  └── Metadata Extraction                                   │
├─────────────────────────────────────────────────────────────┤
│  🗄️ Vector Store (Weaviate Cloud)                         │
│  ├── WeaviateDocumentStore                                 │
│  ├── OpenAI Embeddings (text-embedding-3-small)           │
│  └── Cosine Similarity Search                              │
├─────────────────────────────────────────────────────────────┤
│  🔍 Retrieval & Ranking                                    │
│  ├── WeaviateRetriever (Vector search)                     │
│  ├── LostInTheMiddleRanker (Diversity)                     │
│  └── Top-K Selection                                       │
├─────────────────────────────────────────────────────────────┤
│  🤖 Generation                                             │
│  ├── PromptBuilder (Vietnamese template)                   │
│  ├── OpenAIGenerator (GPT-4o-mini)                        │
│  └── JSON Response Format                                  │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Cài đặt và Chạy

### **Yêu cầu hệ thống**

- Python 3.11 (khuyến nghị) hoặc 3.10
- 4GB RAM trở lên
- Kết nối internet cho OpenAI và Weaviate

### **1. Clone và Setup**

```bash
git clone <repository-url>
cd RAG-chatbot

# Tạo virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate
```

### **2. Cài đặt dependencies**

```bash
# Cài đặt phiên bản cụ thể
pip install -r requirements.txt

# Hoặc cài từng package
pip install streamlit==1.34.0 haystack-ai==2.0.1 weaviate-client==4.6.4 openai==1.40.0
```

### **3. Cấu hình Environment**

Tạo file `.env` trong thư mục gốc:

```env
# OpenAI API
OPENAI_API_KEY=sk-proj-your-openai-key-here

# Weaviate Cloud
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key-here
```

### **4. Chạy ứng dụng**

```bash
# Chạy Streamlit app
python -m streamlit run app.py

# Hoặc với port cụ thể
python -m streamlit run app.py --server.port 8501
```

Truy cập: `http://localhost:8501`

## 📁 Cấu trúc Project

```
RAG-chatbot/
├── app.py                 # Main Streamlit application
├── app_factory.py         # Service factory pattern
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── core/                # Core database modules
│   ├── database.py      # Database abstractions
│   ├── weaviate_database.py  # Weaviate integration
│   └── constants.py     # Constants and defaults
├── services/            # Business logic services
│   ├── rag_pipeline.py  # RAG pipeline implementation
│   ├── document_service.py  # Document processing
│   └── image_database.py    # Image extraction
├── utils/               # Utility functions
│   └── helpers.py       # Helper functions
├── uploads/             # Uploaded documents
├── image_database/      # Extracted images
└── chat_history.json    # Persistent chat data
```

## 🔧 Cấu hình

### **Document Processing**

```python
# config.py
processing = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "supported_formats": [".pdf", ".docx", ".txt", ".md", ".xlsx", ".csv"]
}
```

### **Models**

```python
models = {
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 1536,
    "max_tokens": 2000
}
```

## 💬 Sử dụng

### **1. Upload Documents**

- Kéo thả hoặc chọn file từ sidebar
- Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, CSV
- Auto-processing và chunking

### **2. Chat với AI**

- Nhập câu hỏi vào ô chat
- Bot sẽ tìm kiếm trong documents
- Trả lời với sources và images

### **3. File Management**

- Xem danh sách files đã upload
- Search và filter files
- Xóa files hoặc chat history

## 🚨 Troubleshooting

### **Weaviate Connection Issues**

```bash
# Kiểm tra .env file
cat .env

# Test Weaviate connection
python -c "import weaviate; print('Weaviate OK')"
```

### **Python Version Issues**

```bash
# Kiểm tra Python version
python --version  # Phải là 3.11.x

# Recreate venv nếu cần
rm -rf .venv
python -m venv .venv
```

### **Package Conflicts**

```bash
# Clean install
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📊 Performance

- **Document Processing**: ~2-5s per MB
- **Query Response**: ~1-3s
- **Vector Search**: ~100-500ms
- **Memory Usage**: ~500MB-1GB

## 🔄 Development

### **Code Formatting**

```bash
# Install formatting tools
pip install ruff black isort

# Format code
ruff format .
ruff check . --fix
```

### **Testing**

```bash
# Run basic tests
python -c "from services.rag_pipeline import rag_pipeline; print('Pipeline OK')"
```

## 📝 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Support

- **Issues**: Tạo issue trên GitHub
- **Email**: your-email@example.com
- **Documentation**: Xem thêm docs/ folder
