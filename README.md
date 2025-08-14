# 🤖 Hybrid RAG Chatbot

**Hybrid RAG Pipeline với Haystack Core + LangChain Fallback** - Giải pháp tối ưu performance và độ tin cậy cao.

## 🚀 Tính năng

- **🔧 Hybrid Architecture**: Haystack làm core, LangChain làm fallback
- **📚 Multi-format Support**: PDF, DOCX, TXT, MD, XLSX, XLS, CSV, HTML, JSON
- **⚡ Auto Fallback**: Tự động chuyển sang LangChain khi Haystack có vấn đề
- **🎯 Smart Retrieval**: Embedding + Ranking + Diversity
- **💬 Chat Interface**: UI thân thiện giống ChatGPT
- **📊 Real-time Stats**: Hiển thị pipeline đang hoạt động

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
│   └── query_service.py            # Query processing
├── core/
│   ├── constants.py                # Constants
│   └── database.py                 # Database utilities
├── utils/
│   ├── helpers.py                  # Utility functions
│   ├── parser.py                   # Document parsing
│   ├── qdrant_store.py            # Vector store
│   ├── retrievers.py              # Retrieval components
│   └── schema.py                  # Data schemas
└── requirement.txt                # Dependencies
```

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

## 🎮 Sử dụng

### 1. Upload Documents

- Chọn file từ nhiều định dạng
- Hệ thống tự động detect và xử lý
- Hiển thị progress và kết quả

### 2. Chat với Documents

- Hỏi câu hỏi về nội dung đã upload
- Hệ thống trả lời dựa trên context
- Hiển thị sources tham khảo

### 3. Monitor Pipeline

- Xem pipeline đang hoạt động (Haystack/LangChain)
- Theo dõi số lượng documents
- Kiểm tra performance

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


