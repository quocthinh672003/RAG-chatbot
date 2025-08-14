import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
from typing import List

# Load environment variables
load_dotenv()

# Import Haystack services instead of LangChain
from services.ingest_service import ingestion_service
from services.query_service import query_service
from services.image_database import image_db
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
if "file_mapping" not in st.session_state:
    # Mapping từ tên file tạm thời sang tên file gốc
    st.session_state.file_mapping = {}
if "force_show_upload" not in st.session_state:
    # Flag để force hiển thị upload area
    st.session_state.force_show_upload = False
if "add_file_clicked" not in st.session_state:
    # Track button click
    st.session_state.add_file_clicked = False


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


def get_original_filename(temp_filename: str) -> str:
    """Get original filename from temporary filename"""
    if hasattr(st.session_state, 'file_mapping'):
        return st.session_state.file_mapping.get(temp_filename, temp_filename)
    return temp_filename

def find_relevant_images(query: str, image_files: List[str]) -> List[tuple]:
    """Find images that are relevant to the query based on context"""
    relevant_images = []
    
    # Extract meaningful words from query (remove common words)
    query_lower = query.lower()
    stop_words = ['hình', 'ảnh', 'của', 'tại', 'vào', 'giờ', 'một', 'các', 'vùng', 'bị', 'ảnh', 'hưởng', 'thực', 'tế', 'từ']
    query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
    
    for img_path in image_files:
        # Try to load context from metadata file
        metadata_path = img_path.replace('.png', '_metadata.json')
        context = ""
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    context = metadata.get('context', '').lower()
            except:
                pass
        
        # Score based on word overlap between query and image context
        score = 0
        context_words = context.split()
        
        # Count exact word matches
        for query_word in query_words:
            if query_word in context_words:
                score += 2  # Higher weight for exact matches
        
        # Count partial matches (substring)
        for query_word in query_words:
            for context_word in context_words:
                if query_word in context_word or context_word in query_word:
                    score += 1
        
        # Bonus for important keywords
        important_keywords = ['giao thông', 'đường', 'xe', 'ô tô', 'xe máy', 'đông đúc', 'hà nội', 'tuyến đường', 'trung tâm']
        for keyword in important_keywords:
            if keyword in query_lower and keyword in context:
                score += 3  # Extra bonus for important matches
        
        # Only add images with positive score
        if score > 0:
            relevant_images.append((img_path, context, score))
    
    # Sort by relevance score
    relevant_images.sort(key=lambda x: x[2], reverse=True)
    
    return relevant_images


# Load data on startup
load_processed_files()
load_chat_history()


def main():
    # Check OpenAI API key
    if not config.openai_api_key:
        st.error("❌ OPENAI_API_KEY not found in environment variables")
        st.info("Please set your OpenAI API key in the .env file")
        return

    # Logic hiển thị upload area
    if st.session_state.force_show_upload:
        # Force hiển thị khi user nhấn nút "Thêm file mới"
        st.session_state.show_upload = True
    elif st.session_state.processed_files or st.session_state.chat_history:
        # Ẩn khu vực upload khi có file đã xử lý hoặc có chat history
        st.session_state.show_upload = False
    else:
        # Chỉ hiển thị upload khi không có file và không có chat history
        st.session_state.show_upload = True
    
    # Debug info removed for clean interface

    # Sidebar - Quản lý file gọn gàng
    st.sidebar.markdown("**📁 Quản lý Tài liệu**")

    # Nút thêm file mới
    if st.sidebar.button("📁 Thêm file mới", type="primary", use_container_width=True, key="add_file_btn"):
        st.session_state.force_show_upload = True
        st.session_state.show_upload = True
        st.experimental_rerun()


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
            # Clear database and files only
            try:
                import shutil
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                if os.path.exists("uploads"):
                    shutil.rmtree("uploads")
                print("Cleared FAISS database and uploads folder")
            except Exception as e:
                print(f"Error clearing database: {e}")
            
            st.session_state.processed_files = []
            st.session_state.file_mapping = {}  # Xóa file mapping
            st.session_state.show_upload = True  # Hiển thị lại khu vực upload
            save_processed_files()
            # Không xóa chat history - giữ lại lịch sử chat
            st.experimental_rerun()
    else:
        st.sidebar.info("📝 Chưa có tài liệu nào được xử lý")
    # Main chat interface 
    st.title("🤖 RAG Chatbot")

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
                    # Lọc và hiển thị tên file gốc duy nhất
                    unique_sources = []
                    for source in chat["sources"]:
                        original_name = get_original_filename(source)
                        if original_name not in unique_sources:
                            unique_sources.append(original_name)
                    
                    for source in unique_sources:
                        st.write(f"**📄 {source}**")

    # Chat input (phải đặt bên ngoài columns)
    prompt = st.chat_input("Ask anything...")

    # Upload area và chat input layout
    if st.session_state.show_upload:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            uploaded_files_chat = st.file_uploader(
                "📁 Upload files",
                type=["pdf", "docx", "txt", "md", "xlsx", "xls", "csv", "html", "json"],
                accept_multiple_files=True,
                key="chat_uploader",
                help="Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, XLS, CSV, HTML, JSON. Tối đa 200MB mỗi file.",
            )
            # Chỉ hiển thị nút xử lý khi có file được chọn
            trigger_process = False
            if uploaded_files_chat:
                trigger_process = st.button(
                    "🚀 Xử lý",
                    type="primary",
                    use_container_width=True,
                    key="process_btn",
                )
    else:
        uploaded_files_chat, trigger_process = None, False

    # Khi nhấn xử lý
    if uploaded_files_chat and trigger_process == True:
        with st.spinner("🔄 Đang xử lý tài liệu với Hybrid RAG..."):
            processed_files = []
            failed_files = []
            
            # Debug info
            st.info(f"📁 Đang xử lý {len(uploaded_files_chat)} file(s)")
            
            for uploaded_file in uploaded_files_chat:
                try:
                    st.write(f"🔄 Đang xử lý: {uploaded_file.name}")
                    
                    # Save file to uploads folder for persistence
                    uploads_dir = "uploads"
                    if not os.path.exists(uploads_dir):
                        os.makedirs(uploads_dir)
                    
                    # Fix double extensions
                    file_name = uploaded_file.name
                    if file_name.endswith('.pdf.pdf'):
                        file_name = file_name.replace('.pdf.pdf', '.pdf')
                    elif file_name.endswith('.docx.docx'):
                        file_name = file_name.replace('.docx.docx', '.docx')
                    elif file_name.endswith('.xlsx.xlsx'):
                        file_name = file_name.replace('.xlsx.xlsx', '.xlsx')
                    elif file_name.endswith('.md.md'):
                        file_name = file_name.replace('.md.md', '.md')
                    
                    file_path = os.path.join(uploads_dir, file_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    tmp_file_path = file_path

                    # Use hybrid ingestion service
                    doc_id = ingestion_service.ingest_document(tmp_file_path)
                    
                    st.write(f"📄 Kết quả: {doc_id}")

                    # Extract REAL images from ANY file type
                    try:
                        extracted_images = image_db.extract_images_from_any_file(tmp_file_path, file_name)
                        
                        if extracted_images:
                            st.success(f"🖼️ Đã trích xuất {len(extracted_images)} ảnh THẬT từ file!")
                            for img in extracted_images:
                                img_type = img.get('type', 'unknown')
                                context = img.get('context', 'Không có context')[:50]
                                st.write(f"📷 {img['filename']} ({img_type}) - Context: {context}...")
                        else:
                            st.warning(f"⚠️ Không trích xuất được ảnh nào từ file này!")
                    except Exception as e:
                        st.warning(f"⚠️ Không thể trích xuất ảnh: {e}")

                    if doc_id:
                        processed_files.append(file_name)  # Use clean name
                        if file_name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(file_name)
                            # Lưu mapping từ tên file tạm thời sang tên file gốc
                            st.session_state.file_mapping[os.path.basename(tmp_file_path)] = file_name
                            save_processed_files()
                    else:
                        failed_files.append(uploaded_file.name)

                    # Don't delete the file since we saved it to uploads folder
                    # os.unlink(tmp_file_path)

                except Exception as e:
                    failed_files.append(uploaded_file.name)
                    st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")

            if processed_files:
                st.success(
                    f"✅ **Đã xử lý thành công {len(processed_files)} file(s) với Hybrid RAG!**"
                )
                st.info("💡 **Bây giờ bạn có thể hỏi về nội dung của các file này!**")
                st.session_state.show_upload = False
                st.session_state.force_show_upload = False  # Reset force flag
                st.session_state.add_file_clicked = False  # Reset button state
                st.experimental_rerun()

            if failed_files:
                st.error(f"❌ **Xử lý thất bại {len(failed_files)} file(s)**")
                for file_name in failed_files:
                    st.write(f"• ❌ {file_name}")

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

                    # Debug: Show context
                    if result.get("documents"):
                        with st.expander("🔍 Debug: Context được sử dụng"):
                            for i, doc in enumerate(result["documents"][:3]):  # Show first 3 docs
                                content = getattr(doc, 'page_content', str(doc))[:500]  # First 500 chars
                                st.write(f"**Document {i+1}:** {content}...")

                    # Display answer
                    st.markdown(result["answer"], unsafe_allow_html=True)

                    # Check if user asked about images
                    if any(keyword in prompt.lower() for keyword in ['ảnh', 'image', 'hình', 'picture']):
                        # Query image database first
                        relevant = image_db.find_relevant_images(prompt, max_results=3)
                        if relevant:
                            st.info("🖼️ **Ảnh liên quan trong tài liệu:**")
                            cols = st.columns(len(relevant))
                            for i, img in enumerate(relevant):
                                with cols[i]:
                                    st.image(img["path"], caption=img.get("context", ""), use_column_width=True)
                                    with open(img["path"], "rb") as fh:
                                        st.download_button("📥 Tải ảnh", fh.read(), file_name=img.get("filename", "image.png"), mime="image/png")
                        else:
                            # Fallback: look for extracted images in uploads folder (legacy extractor)
                            import glob
                            image_files = glob.glob("uploads/extracted_image_*.png")
                            if image_files:
                                st.info("🖼️ **Ảnh đã trích xuất (legacy):**")
                                for img_path in image_files[:3]:
                                    st.image(img_path, width=300)
                                    with open(img_path, "rb") as fh:
                                        st.download_button("📥 Tải ảnh", fh.read(), file_name=os.path.basename(img_path), mime="image/png")

                    # Store sources for later display
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Nguồn tham khảo"):
                            # Lọc và hiển thị tên file gốc duy nhất
                            unique_sources = []
                            for source in sources:
                                original_name = get_original_filename(source)
                                if original_name not in unique_sources:
                                    unique_sources.append(original_name)
                            
                            for i, source in enumerate(unique_sources):
                                st.write(f"**📄 {source}**")

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
            st.session_state.show_upload = True  # Hiển thị lại khu vực upload
            save_chat_history()
            st.experimental_rerun()


if __name__ == "__main__":
    main()
