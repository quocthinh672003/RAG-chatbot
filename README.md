# 🤖 RAG Chatbot với Image Support

Ứng dụng RAG (Retrieval-Augmented Generation) chatbot hybrid với khả năng trích xuất và tìm kiếm ảnh từ tài liệu đa định dạng.

## 🚀 Tính năng chính

- **💬 Chat thông minh**: Tương tác với tài liệu qua RAG pipeline
- **📄 Đa định dạng**: Hỗ trợ PDF, DOCX, TXT, MD
- **🖼️ Trích xuất ảnh**: Tự động trích xuất ảnh từ tài liệu với metadata
- **🔍 Tìm kiếm ảnh**: Tìm kiếm ảnh liên quan dựa trên context và keywords
- **📊 Hiển thị bảng**: Tự động phát hiện và hiển thị bảng dữ liệu
- **💾 Lưu trữ chat**: Lưu lịch sử chat và tài liệu đã upload
- **🎯 Nguồn trích dẫn**: Hiển thị chính xác nguồn thông tin

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                             │
├─────────────────────────────────────────────────────────────┤
│  📁 File Upload & Management                                │
│  💬 Chat Interface                                          │
│  🖼️ Image Display & Search                                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services                            │
├─────────────────────────────────────────────────────────────┤
│  🔧 App Factory                                             │
│  📄 Document Service                                        │
│  🎯 RAG Pipeline (Haystack 2.x)                            │
│  🖼️ Image Database                                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    External Services                        │
├─────────────────────────────────────────────────────────────┤
│  🧠 OpenAI API (GPT-4o-mini)                               │
│  🗄️ Weaviate Cloud (Vector Database)                       │
│  📊 Local Storage (Images & Chat History)                  │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Công nghệ sử dụng

### **Core Framework**
- **Streamlit 1.37.1**: Web UI framework
- **Haystack 2.7.0**: RAG pipeline framework
- **Weaviate Client 4.7.1**: Vector database

### **AI & ML**
- **OpenAI 1.40.8**: GPT-4o-mini cho text generation
- **Pydantic 2.8.2**: Data validation

### **Document Processing**
- **PyPDF 4.3.1**: Xử lý PDF files
- **python-docx 1.1.2**: Xử lý DOCX files
- **openpyxl 3.1.5**: Xử lý Excel files
- **markdown 3.6**: Xử lý Markdown files
- **PyMuPDF 1.24.9**: Trích xuất ảnh từ PDF
- **BeautifulSoup 4.12.3**: Xử lý HTML files

### **Image Processing**
- **Pillow 10.4.0**: Xử lý và lưu ảnh
- **zipfile**: Trích xuất ảnh từ Excel/PowerPoint
- **requests 2.32.3**: Download ảnh từ URL

### **Utilities**
- **python-dotenv 1.0.1**: Environment variables
- **pandas 2.2.2**: Data processing
- **loguru 0.7.2**: Logging
- **orjson 3.10.7**: Fast JSON processing

## 📋 Cài đặt

### 1. Clone Repository
```bash
git clone <repository-url>
cd RAG-chatbot
```

### 2. Tạo Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 3. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình Environment
Tạo file `.env` trong thư mục gốc:
```env
# OpenAI API Key (bắt buộc)
OPENAI_API_KEY=your_openai_api_key_here

# Weaviate Cloud (bắt buộc)
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_CLASS_NAME=RAGDocuments
```

### 5. Chạy ứng dụng
```bash
streamlit run app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:8501`

## 📁 Cấu trúc Project

```
RAG-chatbot/
├── app.py                          # 🚀 Main Streamlit application
├── app_factory.py                  # 🔧 Service factory pattern
├── config.py                       # ⚙️ Configuration management
├── requirements.txt                # 📦 Dependencies
├── README.md                       # 📖 Documentation
├── chat_history.json               # 💬 Chat history persistence
├── services/
│   ├── __init__.py
│   ├── document_service.py         # 📄 Document processing
│   ├── image_database.py           # 🖼️ Image extraction & search
│   └── rag_pipeline.py             # 🎯 RAG pipeline (Haystack 2.x)
├── core/
│   ├── __init__.py
│   ├── constants.py                # 📋 Constants & configurations
│   ├── database.py                 # 🗄️ Database utilities
│   └── weaviate_database.py        # 🌐 Weaviate integration
├── utils/
│   ├── __init__.py
│   └── helpers.py                  # 🛠️ Utility functions
├── uploads/                        # 📁 Uploaded files storage
├── image_database/                 # 🖼️ Extracted images storage
│   ├── extracted/                  # Ảnh trích xuất từ tài liệu
│   ├── screenshots/                # Screenshot của tài liệu
│   ├── documents/                  # Ảnh từ document preview
│   ├── general/                    # Ảnh tổng quát khác
│   └── image_metadata.json         # Metadata của ảnh
└── .gitignore                      # Git ignore rules
```

## 🎯 Tính năng chi tiết

### **📄 Document Processing**
- **Đa định dạng**: PDF, DOCX, TXT, MD
- **Text extraction**: Trích xuất text với metadata
- **Chunking**: Chia nhỏ tài liệu thành chunks tối ưu
- **Metadata**: Lưu trữ thông tin nguồn, trang, loại file

### **🖼️ Image Extraction & Search**
- **Trích xuất ảnh**: Từ PDF, DOCX, Excel, PowerPoint
- **Context analysis**: Phân tích context xung quanh ảnh
- **Keyword extraction**: Trích xuất keywords từ context
- **Smart search**: Tìm kiếm ảnh dựa trên query và context
- **Metadata storage**: Lưu trữ thông tin ảnh với JSON

### **💬 Chat Interface**
- **Real-time chat**: Tương tác trực tiếp với AI
- **Source citation**: Hiển thị nguồn thông tin chính xác
- **Table detection**: Tự động phát hiện và hiển thị bảng
- **Image display**: Hiển thị ảnh liên quan với download
- **Chat history**: Lưu trữ và khôi phục lịch sử chat

### **📁 File Management**
- **Smart file list**: Hiển thị danh sách file với search
- **File type icons**: Icons cho từng loại file
- **Pagination**: Phân trang cho danh sách file dài
- **Auto-reload**: Tự động tải lại tài liệu từ thư mục uploads


## 🚀 Sử dụng

### **1. Upload Tài liệu**
- Kéo thả hoặc click để upload file
- Hỗ trợ: PDF, DOCX, TXT, MD, XLSX
- Tự động trích xuất text và ảnh

### **2. Chat với AI**
- Gõ câu hỏi về tài liệu đã upload
- AI sẽ trả lời dựa trên nội dung tài liệu
- Hiển thị nguồn thông tin chính xác

### **3. Tìm kiếm Ảnh**
- Hỏi về ảnh trong tài liệu
- AI sẽ tìm và hiển thị ảnh liên quan
- Có thể download ảnh về máy

### **4. Quản lý File**
- Xem danh sách file đã upload
- Tìm kiếm file theo tên
- Xóa file không cần thiết

## 🔍 Tìm kiếm Ảnh

### **Logic Tìm kiếm**
1. **Context matching**: +2 điểm cho mỗi từ khớp trong context
2. **Keyword matching**: +3 điểm cho mỗi keyword khớp
3. **Source file relevance**: +1 điểm nếu tên file liên quan
4. **General queries**: +5 điểm nếu query hỏi về ảnh nói chung

### **Logic Tìm kiếm Ảnh**
- **Context matching**: +2 điểm cho mỗi từ khớp trong context xung quanh ảnh
- **Keyword matching**: +3 điểm cho mỗi keyword được trích xuất từ context
- **Source file relevance**: +1 điểm nếu tên file nguồn liên quan
- **General queries**: +5 điểm nếu query hỏi chung về ảnh

### **Cách Keywords được tạo**
Keywords được trích xuất **động** từ context xung quanh ảnh, Hệ thống sẽ:
1. Phân tích text xung quanh ảnh trong tài liệu
2. Tìm các từ khóa liên quan đến chủ đề
3. Lưu keywords vào metadata của ảnh
4. Sử dụng keywords này để tìm kiếm khi user hỏi

