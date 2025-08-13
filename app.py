import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import json
import re
import hashlib
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import qdrant_client
from qdrant_client.models import Distance, VectorParams
import uuid
from datetime import datetime
import nltk
import ssl

# Download NLTK data if not available
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data for markdown processing"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.warning(f"⚠️ Warning: Could not download NLTK data: {e}")
        return False

# Download NLTK data on startup
download_nltk_data()


# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm gọn gàng hơn
st.markdown("""
<style>
    /* Dark theme palette */
    :root {
        --bg: #121212;
        --bg-elev: #1a1a1a;
        --panel: #1f1f1f;
        --panel-2: #262626;
        --text: #e8e8e8;
        --muted: #b5b5b5;
        --border: #3a3a3a;  
        --accent: #8b5cf6;
    }
    
    /* Modern UI Styling */
    

    

    

    

    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Custom styling cho nút xử lý */
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(255, 75, 75, 0.3) !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(45deg, #ff3333, #ff5252) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(255, 75, 75, 0.4) !important;
    }
    
    /* Keep default uploader visible; only lightly style it */
    .stFileUploader > div > div {
        display: block;
    }
    

    

    
    /* Improved sidebar styling without forced scrollbars - theme tối */
    [data-testid="stSidebar"] {
        background: #2d2d2d;
        border-right: 1px solid #404040;
        color: white;
    }
    
    /* Clean file list container */
    .files-container {
        background: var(--panel);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid var(--border);
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
        color: var(--text);
    }
    
    /* Clean uploaded files container */
    .uploaded-files-container {
        background: var(--panel);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid var(--border);
        margin: 10px 0;
        max-height: 200px;
        overflow-y: auto;
        color: var(--text);
    }
    
    /* Custom scrollbar styling - only when needed */
    .files-container::-webkit-scrollbar,
    .uploaded-files-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .files-container::-webkit-scrollbar-track,
    .uploaded-files-container::-webkit-scrollbar-track {
        background: #f0f0f0;
        border-radius: 3px;
    }
    
    .files-container::-webkit-scrollbar-thumb,
    .uploaded-files-container::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 3px;
    }
    
    .files-container::-webkit-scrollbar-thumb:hover,
    .uploaded-files-container::-webkit-scrollbar-thumb:hover {
        background: #666;
    }
    

    
    /* Button row styling */
    .button-row {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 15px;
    }
    
    /* File info styling */
    .file-info {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px;
        background: var(--panel-2);
        border-radius: 6px;
        margin: 5px 0;
        border: 1px solid var(--border);
        color: var(--text);
    }
    
    .file-name {
        font-weight: 500;
        color: var(--text);
    }
    
    .file-size {
        font-size: 0.8rem;
        color: var(--muted);
    }
    
    /* Remove default scrollbars */
    .main .block-container {
        overflow: visible !important;
    }
    
    [data-testid="stSidebar"] > div {
        overflow: visible !important;
    }
    
    /* Hide unnecessary scrollbars */
    .element-container {
        overflow: visible !important;
    }
    
    /* Sidebar files container với thanh cuộn - theme tối */
    .sidebar-files-container {
        max-height: 300px !important;
        overflow-y: auto !important;
        padding: 10px !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background: var(--panel) !important;
        margin: 10px 0 !important;
        color: var(--text) !important;
    }
    
    /* Theme tối cho toàn bộ ứng dụng */
    .main {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    
    .main .block-container {
        background: var(--bg-elev) !important;
        color: var(--text) !important;
    }
    
    /* Sidebar theme tối */
    [data-testid="stSidebar"] {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] strong {
        color: var(--text) !important;
        font-weight: 700 !important;
    }
    
    /* Main content theme tối */
    .stMarkdown {
        color: var(--text) !important;
    }
    
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: var(--text) !important;
    }
    
    .stMarkdown strong {
        color: var(--text) !important;
    }
    
    /* File uploader theme tối */
    .stFileUploader {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    /* Uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel-2) !important;
        border: 1px dashed var(--border) !important;
        color: var(--text) !important;
    }
    
    /* Buttons theme tối */
    .stButton > button {
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    
    .stButton > button:hover {
        background: #333 !important;
    }
    
    .sidebar-files-container::-webkit-scrollbar {
        width: 6px !important;
    }
    
    .sidebar-files-container::-webkit-scrollbar-track {
        background: var(--bg-elev) !important;
        border-radius: 3px !important;
    }
    
    .sidebar-files-container::-webkit-scrollbar-thumb {
        background: #555 !important;
        border-radius: 3px !important;
    }
    
    .sidebar-files-container::-webkit-scrollbar-thumb:hover {
        background: #666 !important;
    }

    /* Chat input dark theme */
    [data-testid="stChatInput"] textarea {
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
    }
    [data-testid="stChatInput"] button {
        background: var(--panel-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    /* Remove empty Streamlit containers and top gaps */
    [data-testid="element-container"]:empty { display: none !important; }
    [data-testid="element-container"] > div:empty { display: none !important; }
    [data-testid="element-container"]:has(> div:empty) { display: none !important; }
    section.main > div.block-container { padding-top: 0.5rem !important; }
    section.main [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    .main h1 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'show_upload' not in st.session_state:
    # Hiển thị khu vực upload lúc khởi động; sẽ ẩn sau khi người dùng gửi prompt
    st.session_state.show_upload = True
if 'corpus_docs' not in st.session_state:
    st.session_state.corpus_docs = []
if 'bm25_retriever' not in st.session_state:
    st.session_state.bm25_retriever = None

# Load processed files from persistent storage
def load_processed_files():
    """Load processed files list from file"""
    try:
        if os.path.exists('processed_files.txt'):
            with open('processed_files.txt', 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f.readlines() if line.strip()]
                st.session_state.processed_files = files
    except Exception as e:
        st.warning(f"⚠️ Could not load processed files: {e}")

def save_processed_files():
    """Save processed files list to file"""
    try:
        with open('processed_files.txt', 'w', encoding='utf-8') as f:
            for file_name in st.session_state.processed_files:
                f.write(f"{file_name}\n")
    except Exception as e:
        st.error(f"❌ Could not save processed files: {e}")

# Load chat history from persistent storage
def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists('chat_history.json'):
            import json
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load chat history: {e}")

def save_chat_history():
    """Save chat history to file"""
    try:
        import json
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ Could not save chat history: {e}")

# Load data on startup
load_processed_files()
load_chat_history()

def init_vectorstore():
    """Initialize Qdrant vector store"""
    try:
        client = qdrant_client.QdrantClient("localhost", port=6333)
        
        # Create collection if not exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "rag_documents" not in collection_names:
            client.create_collection(
                collection_name="rag_documents",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            st.success("✅ Created new vector collection")
            
        # Initialize embeddings with explicit API key
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OPENAI_API_KEY not found in environment variables")
            return None
            
        # Set environment variable for OpenAI
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Create embeddings with simple syntax
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        
        return Qdrant(
            client=client,
            collection_name="rag_documents",
            embeddings=embeddings
        )
    except Exception as e:
        st.error(f"❌ Vector store error: {e}")
        return None

def load_document(file_path, file_type):
    """Load document based on file type"""
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        elif file_type == "md":
            # Try UnstructuredMarkdownLoader first, fallback to TextLoader if it fails
            try:
                loader = UnstructuredMarkdownLoader(file_path)
            except Exception as md_error:
                st.warning(f"⚠️ Markdown loader failed, using text loader instead: {md_error}")
                loader = TextLoader(file_path)
        elif file_type in ["xlsx", "xls"]:
            # Try UnstructuredExcelLoader first, fallback to TextLoader if it fails
            try:
                loader = UnstructuredExcelLoader(file_path)
            except Exception as excel_error:
                st.warning(f"⚠️ Excel loader failed, using text loader instead: {excel_error}")
                loader = TextLoader(file_path)
        else:
            return None
            
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "file_name": os.path.basename(file_path),
                "file_type": file_type,
                "upload_time": datetime.now().isoformat(),
                "document_id": str(uuid.uuid4()),
                "source_path": file_path,
                "permission_groups": ["public"]
            })
            
        return documents
    except Exception as e:
        st.error(f"❌ Error loading {file_type} file: {e}")
        return None

def compute_file_sha256(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return str(uuid.uuid4())

def load_document_id_map() -> dict:
    try:
        os.makedirs("uploads", exist_ok=True)
        map_path = os.path.join("uploads", "document_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_document_id_map(mapping: dict) -> None:
    try:
        map_path = os.path.join("uploads", "document_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def detect_element_type(text: str) -> str:
    """Best-effort element type detection using simple heuristics."""
    if not text:
        return "NarrativeText"
    first_line = text.strip().splitlines()[0] if text.strip().splitlines() else ""
    if first_line.startswith(('#', '##', '###')):
        return "Title"
    if text.strip().startswith('|') and '\n|' in text:
        return "Table"
    if re.match(r"^\s*[-*•] ", first_line):
        return "ListItem"
    return "NarrativeText"

def build_standard_json(documents, splits, document_id: str):
    """Build a standard JSON structure from original documents and their splits."""
    if not documents:
        return None
    source_filename = documents[0].metadata.get("file_name")
    source_path = documents[0].metadata.get("source_path")
    ingestion_timestamp = documents[0].metadata.get("upload_time", datetime.now().isoformat())
    permission_groups = documents[0].metadata.get("permission_groups", ["public"])    
    payload = {
        "document_metadata": {
            "document_id": document_id,
            "source_filename": source_filename,
            "source_path": source_path,
            "ingestion_timestamp": ingestion_timestamp,
            "permission_groups": permission_groups,
        },
        "elements": []
    }
    for split in splits:
        element_id = str(uuid.uuid4())
        elem_type = detect_element_type(split.page_content)
        elem_meta = {
            "page_number": split.metadata.get("page") or split.metadata.get("page_number"),
            "parent_id": None,
            "language": split.metadata.get("language", "vi"),
        }
        payload["elements"].append({
            "element_id": element_id,
            "type": elem_type,
            "content": split.page_content,
            "content_format": "markdown",
            "metadata": elem_meta,
        })
        # Also store ids into split metadata for vector store traceability
        split.metadata.update({
            "document_id": document_id,
            "element_id": element_id,
            "element_type": elem_type,
        })
    return payload

def process_documents(documents, vectorstore):
    """Process and store documents"""
    if not documents:
        return False
        
    try:
        # Smart split: giữ nguyên cấu trúc Markdown và bảng
        def smart_split(docs):
            has_md = any(d.metadata.get("file_type") == "md" for d in docs)
            if has_md:
                sections = []
                for d in docs:
                    if d.metadata.get("file_type") == "md":
                        splitter = MarkdownHeaderTextSplitter(
                            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
                            strip_headers=False,
                        )
                        parts = splitter.split_text(d.page_content)
                        # Convert to Documents, giữ metadata gốc
                        for p in parts:
                            from langchain.schema import Document
                            sections.append(Document(page_content=p.page_content, metadata={**d.metadata}))
                    else:
                        sections.append(d)
                # Sau đó cắt theo độ dài nhưng lớn hơn để tránh vỡ bảng
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                return text_splitter.split_documents(sections)
            # Non-markdown
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len
            )
            return text_splitter.split_documents(docs)

        splits = smart_split(documents)

        # Build and persist standard JSON alongside vectorization
        # Create stable document_id based on file content hash
        file_hash = compute_file_sha256(documents[0].metadata.get("source_path", ""))
        id_map = load_document_id_map()
        document_id = id_map.get(file_hash) or documents[0].metadata.get("document_id", str(uuid.uuid4()))
        id_map[file_hash] = document_id
        save_document_id_map(id_map)
        std_json = build_standard_json(documents, splits, document_id)
        try:
            os.makedirs("uploads/structured", exist_ok=True)
            json_path = os.path.join("uploads/structured", f"{document_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(std_json, f, ensure_ascii=False, indent=2)
        except Exception as json_err:
            st.warning(f"⚠️ Không thể lưu JSON cấu trúc: {json_err}")
        
        # Store in vector database
        vectorstore.add_documents(splits)

        # Cập nhật BM25 retriever (sparse) để hybrid search
        try:
            st.session_state.corpus_docs.extend(splits)
            st.session_state.bm25_retriever = BM25Retriever.from_documents(st.session_state.corpus_docs)
            st.session_state.bm25_retriever.k = 20
        except Exception as bm25_err:
            st.warning(f"⚠️ Không thể xây BM25 retriever: {bm25_err}")
        
        st.success(f"✅ Processed {len(splits)} chunks from {len(documents)} documents")
        return True
    except Exception as e:
        st.error(f"❌ Error processing documents: {e}")
        return False

def main():
    # Initialize vector store
    vectorstore = init_vectorstore()
    if not vectorstore:
        st.error("❌ Không thể kết nối vector store")
        return
    
    # Initialize chat components
    if 'qa_chain' not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1200
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        qa_template = (
            "Bạn là trợ lý dữ liệu, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.\n"
            "YÊU CẦU NGHIÊM NGẶT: Không suy đoán, không dùng kiến thức ngoài ngữ cảnh.\n"
            "Hướng dẫn trả lời (bằng tiếng Việt):\n"
            "- Nếu có số liệu/bảng: trích đúng số, kèm đơn vị (ví dụ: TWh, %, tỷ lệ). Trả lời đầy đủ, KHÔNG tóm tắt.\n"
            "- Ưu tiên số liệu đúng NĂM/ĐỊA ĐIỂM được hỏi; nếu nhiều mục (ví dụ theo ngành), liệt kê rõ ràng.\n"
            "- Nếu có bảng phù hợp: xuất lại bảng Markdown đầy đủ từ dữ liệu trong ngữ cảnh (không lược bớt cột chính).\n"
            "- Nếu ngữ cảnh không đủ thông tin: trả lời đúng câu sau: 'Không tìm thấy thông tin trong tài liệu đã cung cấp.'\n\n"
            "[Ngữ cảnh]:\n{context}\n\n"
            "[Lịch sử hội thoại]:\n{chat_history}\n\n"
            "[Câu hỏi]: {question}\n\n"
            "Xuất trả lời ở dạng Markdown, có thể bao gồm bảng, bullet. ĐẦY ĐỦ theo ngữ cảnh."
        )
        qa_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=qa_template,
        )
        # Hybrid retriever: dense (Qdrant) + sparse (BM25)
        dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})
        if st.session_state.bm25_retriever is not None:
            hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, st.session_state.bm25_retriever], weights=[0.6, 0.4])
        else:
            hybrid_retriever = dense_retriever
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )
    
    # Nếu đã có lịch sử chat → ẩn khu vực upload/help
    if st.session_state.chat_history:
        st.session_state.show_upload = False

    # Sidebar - Quản lý file gọn gàng
    st.sidebar.markdown("**📁 Quản lý Tài liệu**")
    
    # Hiển thị danh sách file đã xử lý với thanh cuộn
    if st.session_state.processed_files:
        st.sidebar.markdown(f"**📊 Tổng cộng: {len(st.session_state.processed_files)} file(s)**")
        
        # Container cho danh sách files với thanh cuộn
        st.markdown('<div class="sidebar-files-container">', unsafe_allow_html=True)
        for i, file_name in enumerate(st.session_state.processed_files):
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                # Hiển thị tên file thông minh
                if len(file_name) <= 20:
                    display_name = file_name
                else:
                    name_parts = file_name.rsplit('.', 1)
                    if len(name_parts) == 2:
                        base_name, extension = name_parts
                        if len(base_name) > 12:
                            display_name = f"{base_name[:8]}...{base_name[-4:]}.{extension}"
                        else:
                            display_name = file_name
                    else:
                        display_name = f"{file_name[:8]}...{file_name[-4:]}" if len(file_name) > 12 else file_name
                
                st.markdown(f"✅ {display_name}")
            with col2:
                if st.button("🗑️", key=f"del_{i}", help="Xóa file", use_container_width=True):
                    st.session_state.processed_files.remove(file_name)
                    save_processed_files()
                    st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.sidebar.button("🗑️ Xóa tất cả", type="secondary", use_container_width=True):
            st.session_state.processed_files = []
            save_processed_files()
            st.experimental_rerun()
    else:
        st.sidebar.info("📝 Chưa có tài liệu nào được xử lý")
        st.sidebar.markdown("*Click nút + để upload file và bắt đầu!*")
    
    # Main chat interface - giống ChatGPT
    st.title("🤖 RAG Chatbot")
    
    # Welcome message + upload/help chỉ hiển thị khi show_upload = True
    if st.session_state.show_upload:
        st.markdown("""
        **Tôi có thể giúp bạn:**
        - 📚 Trả lời câu hỏi về tài liệu đã upload
        - 🔍 Tìm kiếm thông tin trong documents
        - 💡 Phân tích và giải thích nội dung
        """)
    
    # Chat history display
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat["question"])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(chat["answer"], unsafe_allow_html=True)
            
            # Show sources if available
            if "sources" in chat:
                with st.expander("📚 Nguồn tham khảo"):
                    for j, source in enumerate(chat["sources"]):
                        st.write(f"**Nguồn #{j+1}:** {source}")
    
    # Upload area chỉ hiển thị khi show_upload = True
    if st.session_state.show_upload:
        st.markdown("**📁 Upload files**")
        uploaded_files_chat = st.file_uploader(
            "Browse files",
            type=['pdf', 'docx', 'txt', 'md', 'xlsx', 'xls'],
            accept_multiple_files=True,
            key="chat_uploader",
            help="Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, XLS. Tối đa 200MB mỗi file."
        )
        trigger_process = st.button("🚀 Xử lý", type="primary", use_container_width=True, key="process_btn") if uploaded_files_chat else False
    else:
        uploaded_files_chat, trigger_process = None, False

    # Khi nhấn xử lý
    if uploaded_files_chat and trigger_process == True:
        with st.spinner("🔄 Đang xử lý tài liệu..."):
            processed_files = []
            failed_files = []
            for uploaded_file in uploaded_files_chat:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    documents = load_document(tmp_file_path, file_type)
                    if documents:
                        success = process_documents(documents, vectorstore)
                        if success:
                            processed_files.append(uploaded_file.name)
                            if uploaded_file.name not in st.session_state.processed_files:
                                st.session_state.processed_files.append(uploaded_file.name)
                                save_processed_files()
                        else:
                            failed_files.append(uploaded_file.name)
                    else:
                        failed_files.append(uploaded_file.name)
                    os.unlink(tmp_file_path)
                except Exception as e:
                    failed_files.append(uploaded_file.name)
                    st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")
            if processed_files:
                st.success(f"✅ **Đã xử lý thành công {len(processed_files)} file(s)**")
                st.info("💡 **Bây giờ bạn có thể hỏi về nội dung của các file này!**")
                st.session_state.show_upload = False
                st.experimental_rerun()
            if failed_files:
                st.error(f"❌ **Xử lý thất bại {len(failed_files)} file(s)**")
                for file_name in failed_files:
                    st.write(f"• ❌ {file_name}")

    # Chat input (phải đặt bên ngoài columns)
    prompt = st.chat_input("Ask anything...")

    # Xử lý chat input
    if prompt:
        # Add user message to chat
        st.chat_message("user").write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    result = st.session_state.qa_chain.invoke({"question": prompt})
                    # Nếu câu trả lời rỗng/không tìm thấy → thử lại với k lớn hơn & MMR
                    answer_text = (result.get("answer") or "").strip().lower()
                    not_found_phrase = "không tìm thấy thông tin trong tài liệu đã cung cấp"
                    if (not answer_text) or (answer_text in ["không biết", "tôi không biết", "i don't know"]) or (not_found_phrase in answer_text):
                        # rebuild retriever with higher k for this turn
                        st.session_state.qa_chain.retriever = st.session_state.qa_chain.retriever if hasattr(st.session_state.qa_chain, 'retriever') else vectorstore.as_retriever()
                        try:
                            better = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 30, "fetch_k": 80, "lambda_mult": 0.3}),
                                memory=st.session_state.qa_chain.memory,
                                return_source_documents=True,
                                chain_type="stuff",
                                combine_docs_chain_kwargs={"prompt": qa_prompt},
                            )
                            result = better.invoke({"question": prompt})
                        except Exception:
                            pass
                    
                    # Display answer (full markdown, no summarization)
                    st.markdown(result["answer"], unsafe_allow_html=True)
                    
                    # Store sources for later display
                    sources = []
                    if result["source_documents"]:
                        with st.expander("📚 Nguồn tham khảo"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.write(f"**Nguồn #{i+1} - {doc.metadata.get('file_name', 'Unknown')}:**")
                                st.write(f"*{doc.page_content[:200]}...*")
                                sources.append(f"{doc.metadata.get('file_name', 'Unknown')}")
                    
                    # Add to chat history and save
                    st.session_state.chat_history.append({
                        "question": prompt,
                        "answer": result["answer"],
                        "sources": sources
                    })
                    save_chat_history()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_history = []
            save_chat_history()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
