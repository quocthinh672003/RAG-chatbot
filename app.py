import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Import Haystack services instead of LangChain
from services.ingest_service import ingestion_service
from services.query_service import query_service
from config import config

# Download NLTK data if not available
try:
    import nltk
    import ssl

    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data for markdown processing"""
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("words", quiet=True)
        nltk.download("stopwords", quiet=True)
        return True
    except Exception as e:
        st.warning(f"⚠️ Warning: Could not download NLTK data: {e}")
        return False


# Download NLTK data on startup
download_nltk_data()

# Page config
st.set_page_config(
    page_title="Haystack RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS để làm gọn gàng hơn
st.markdown(
    """
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
    
    /* File uploader theme tối */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel-2) !important;
        border: 1px dashed var(--border) !important;
        color: var(--text) !important;
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
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "show_upload" not in st.session_state:
    # Hiển thị khu vực upload lúc khởi động; sẽ ẩn sau khi người dùng gửi prompt
    st.session_state.show_upload = True


# Load processed files from persistent storage
def load_processed_files():
    """Load processed files list from file"""
    try:
        if os.path.exists("processed_files.txt"):
            with open("processed_files.txt", "r", encoding="utf-8") as f:
                files = [line.strip() for line in f.readlines() if line.strip()]
                st.session_state.processed_files = files
    except Exception as e:
        st.warning(f"⚠️ Could not load processed files: {e}")


def save_processed_files():
    """Save processed files list to file"""
    try:
        with open("processed_files.txt", "w", encoding="utf-8") as f:
            for file_name in st.session_state.processed_files:
                f.write(f"{file_name}\n")
    except Exception as e:
        st.error(f"❌ Could not save processed files: {e}")


# Load chat history from persistent storage
def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists("chat_history.json"):
            import json

            with open("chat_history.json", "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load chat history: {e}")


def save_chat_history():
    """Save chat history to file"""
    try:
        import json

        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ Could not save chat history: {e}")


# Load data on startup
load_processed_files()
load_chat_history()


def main():
    # Check OpenAI API key
    if not config.openai_api_key:
        st.error("❌ OPENAI_API_KEY not found in environment variables")
        st.info("Please set your OpenAI API key in the .env file")
        return

    # Nếu đã có lịch sử chat → ẩn khu vực upload/help
    if st.session_state.chat_history:
        st.session_state.show_upload = False

    # Sidebar - Quản lý file gọn gàng
    st.sidebar.markdown("**📁 Quản lý Tài liệu**")

    # Hiển thị thông tin hybrid pipeline
    try:
        pipeline_info = query_service.get_pipeline_info()
        st.sidebar.markdown(f"**🔧 Pipeline:** {pipeline_info['pipeline_type']}")
        st.sidebar.markdown(f"**⚡ Active:** {pipeline_info['active_pipeline']}")
        st.sidebar.markdown(f"**📊 Documents:** {pipeline_info['document_count']}")
    except:
        st.sidebar.markdown("**🔧 Pipeline:** Hybrid RAG")
        st.sidebar.markdown("**⚡ Active:** Loading...")
        st.sidebar.markdown("**📊 Documents:** Loading...")

    # Hiển thị danh sách file đã xử lý với thanh cuộn
    if st.session_state.processed_files:
        st.sidebar.markdown(
            f"**📊 Tổng cộng: {len(st.session_state.processed_files)} file(s)**"
        )

        # Container cho danh sách files với thanh cuộn
        st.markdown('<div class="sidebar-files-container">', unsafe_allow_html=True)
        for i, file_name in enumerate(st.session_state.processed_files):
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                # Hiển thị tên file thông minh
                if len(file_name) <= 20:
                    display_name = file_name
                else:
                    name_parts = file_name.rsplit(".", 1)
                    if len(name_parts) == 2:
                        base_name, extension = name_parts
                        if len(base_name) > 12:
                            display_name = (
                                f"{base_name[:8]}...{base_name[-4:]}.{extension}"
                            )
                        else:
                            display_name = file_name
                    else:
                        display_name = (
                            f"{file_name[:8]}...{file_name[-4:]}"
                            if len(file_name) > 12
                            else file_name
                        )

                st.markdown(f"✅ {display_name}")
            with col2:
                if st.button(
                    "🗑️", key=f"del_{i}", help="Xóa file", use_container_width=True
                ):
                    st.session_state.processed_files.remove(file_name)
                    save_processed_files()
                    st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.sidebar.button(
            "🗑️ Xóa tất cả", type="secondary", use_container_width=True
        ):
            st.session_state.processed_files = []
            save_processed_files()
            st.experimental_rerun()
    else:
        st.sidebar.info("📝 Chưa có tài liệu nào được xử lý")
        st.sidebar.markdown("*Click nút + để upload file và bắt đầu!*")

    # Main chat interface - giống ChatGPT
    st.title("🤖 Hybrid RAG Chatbot")

    # Welcome message + upload/help chỉ hiển thị khi show_upload = True
    if st.session_state.show_upload:
        st.markdown(
            """
        **Tôi có thể giúp bạn:**
        - 📚 Trả lời câu hỏi về tài liệu đã upload
        - 🔍 Tìm kiếm thông tin trong documents
        - 💡 Phân tích và giải thích nội dung
        """
        )

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
            type=["pdf", "docx", "txt", "md", "xlsx", "xls", "csv", "html", "json"],
            accept_multiple_files=True,
            key="chat_uploader",
            help="Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, XLS, CSV, HTML, JSON. Tối đa 200MB mỗi file.",
        )
        trigger_process = (
            st.button(
                "🚀 Xử lý với Haystack",
                type="primary",
                use_container_width=True,
                key="process_btn",
            )
            if uploaded_files_chat
            else False
        )
    else:
        uploaded_files_chat, trigger_process = None, False

    # Khi nhấn xử lý
    if uploaded_files_chat and trigger_process == True:
        with st.spinner("🔄 Đang xử lý tài liệu với Hybrid RAG..."):
            processed_files = []
            failed_files = []
            for uploaded_file in uploaded_files_chat:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Use hybrid ingestion service
                    doc_id = ingestion_service.ingest_document(tmp_file_path)

                    if doc_id:
                        processed_files.append(uploaded_file.name)
                        if uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)
                            save_processed_files()
                    else:
                        failed_files.append(uploaded_file.name)

                    os.unlink(tmp_file_path)

                except Exception as e:
                    failed_files.append(uploaded_file.name)
                    st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")

            if processed_files:
                st.success(
                    f"✅ **Đã xử lý thành công {len(processed_files)} file(s) với Hybrid RAG!**"
                )
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

        # Get AI response using hybrid RAG
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ với Hybrid RAG..."):
                try:
                    # Use hybrid query service
                    result = query_service.query(prompt)

                    # Display answer
                    st.markdown(result["answer"], unsafe_allow_html=True)

                    # Store sources for later display
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Nguồn tham khảo"):
                            for i, source in enumerate(sources):
                                st.write(f"**Nguồn #{i+1}:** {source}")

                    # Add to chat history and save
                    st.session_state.chat_history.append(
                        {
                            "question": prompt,
                            "answer": result["answer"],
                            "sources": sources,
                        }
                    )
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
