# 🤖 RAG Chatbot - Hệ thống Hỏi Đáp Thông Minh

Hệ thống RAG (Retrieval-Augmented Generation) chatbot có khả năng đọc, hiểu và trích xuất thông tin từ tài liệu để trả lời câu hỏi của người dùng bằng ngôn ngữ tự nhiên.

## 🚀 Tính năng chính

### FR1: Module Nạp và Xử lý Tài liệu
- ✅ Tải lên tài liệu qua Streamlit UI (PDF, DOCX, TXT, MD, XLSX, PPTX, HTML, JSON, CSV)
- ✅ Sử dụng Haystack converters cho từng loại file
- ✅ Phân tách văn bản thông minh (chunking)
- ✅ Vector hóa sử dụng OpenAI text-embedding-3-small
- ✅ Lưu trữ vào Qdrant Vector Database

### FR2: Module Tìm kiếm và Trả lời
- ✅ Giao diện chat thân thiện với Streamlit
- ✅ Tìm kiếm tương đồng (similarity search)
- ✅ Tạo câu trả lời bằng GPT-4o-mini
- ✅ Trích dẫn nguồn tài liệu

### FR3: Module Quản lý (API đơn giản)
- ✅ Xem danh sách tài liệu qua REST API
- ✅ Health check hệ thống

## 🛠️ Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirement.txt
```

### 2. Cài đặt Qdrant
```bash
# Sử dụng Docker (khuyến nghị)
docker-compose up -d

# Hoặc chạy trực tiếp
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Cấu hình môi trường
Tạo file `.env` trong thư mục gốc:
```env
OPENAI_API_KEY=your_openai_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_document
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## 🚀 Chạy ứng dụng

### 1. Chạy giao diện Streamlit (chính)
```bash
streamlit run app.py
```
Truy cập: http://localhost:8501

### 2. Chạy API admin (tùy chọn)
```bash
python api.py
```
API docs: http://localhost:8000/docs

## 📖 Hướng dẫn sử dụng

### Giao diện Web (Streamlit)

1. **Upload tài liệu:**
   - Chọn file từ sidebar (hỗ trợ nhiều định dạng)
   - Chọn nhóm quyền truy cập
   - Nhấn "Xử lý và Lưu trữ"

2. **Đặt câu hỏi:**
   - Nhập câu hỏi vào ô chat
   - Chọn nhóm quyền truy vấn
   - Nhấn "Tìm kiếm và Trả lời"

3. **Xem kết quả:**
   - Câu trả lời từ AI
   - Nguồn tham khảo với trích dẫn
   - Lịch sử chat

### API Endpoints (Admin)

#### Xem danh sách tài liệu
```bash
GET /documents
```

#### Kiểm tra trạng thái
```bash
GET /health
```

## 🏗️ Kiến trúc hệ thống

```
RAG Chatbot/
├── app.py              # Giao diện Streamlit chính
├── api.py              # API FastAPI
├── config.py           # Cấu hình hệ thống
├── core/               # Core logic
├── services/           # Services
├── utils/              # Utilities
├── uploads/            # Thư mục upload
├── docker-compose.yml  # Khởi động Qdrant
└── requirement.txt     # Dependencies
├── simple_api.py       # API admin đơn giản
├── ingest.py           # Module xử lý tài liệu
├── query.py            # Module truy vấn và trả lời
├── config.py           # Cấu hình hệ thống
├── utils/
│   ├── converters.py   # Haystack converters cho từng loại file
│   ├── embeddings.py   # OpenAI embeddings
│   └── qdrant_store.py # Qdrant database
└── requirements.txt    # Dependencies
```

## 🔧 Cấu hình

### Models sử dụng:
- **Embedding:** OpenAI text-embedding-3-small (1536 dimensions)
- **LLM:** OpenAI gpt-4o-mini

### Haystack Converters:
- **PDF:** PDFMinerToDocument
- **DOCX:** DOCXToDocument
- **TXT:** TextFileToDocument
- **MD:** MarkdownToDocument
- **XLSX:** XLSXToDocument
- **PPTX:** PPTXToDocument
- **HTML:** HTMLToDocument
- **JSON:** JSONConverter
- **CSV:** CSVToDocument

### Cấu hình chunking:
- **Chunk size:** 1000 ký tự
- **Chunk overlap:** 100 ký tự
- **Top K:** 10 documents

## 📝 Ví dụ sử dụng

### 1. Upload tài liệu qua UI
- Mở http://localhost:8501
- Chọn file từ sidebar
- Chọn nhóm quyền
- Click "Xử lý và Lưu trữ"

### 2. Hỏi đáp
- Nhập câu hỏi vào ô chat
- Chọn nhóm quyền truy vấn
- Click "Tìm kiếm và Trả lời"

### 3. Xem danh sách tài liệu (Admin)
```bash
curl http://localhost:8000/documents
```

## 🐛 Troubleshooting

### Lỗi kết nối Qdrant
```bash
# Kiểm tra Qdrant đang chạy
curl http://localhost:6333/collections

# Khởi động lại Qdrant
docker-compose restart
```

### Lỗi OpenAI API
- Kiểm tra API key trong file `.env`
- Đảm bảo có đủ credit trong tài khoản OpenAI

### Lỗi dependencies
```bash
# Cài đặt lại dependencies
pip install -r requirement.txt --force-reinstall
```

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.
