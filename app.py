"""
Hybrid RAG Chatbot with Image Support
Optimized version with better performance and maintainability
"""

import streamlit as st
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import all services at top level for better performance
from app_factory import initialize_app, get_app_factory
from services.document_service import DocumentService

# Download NLTK data once at startup
try:
    import nltk
    import ssl
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Global constants
CHAT_HISTORY_FILE = "chat_history.json"
UPLOADS_DIR = "uploads"
MAX_FILENAME_LENGTH = 25


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


@lru_cache(maxsize=1)
def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from JSON file with caching"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                logger.info(f"✅ Loaded {len(history)} chat messages from {CHAT_HISTORY_FILE}")
                return history
        else:
            logger.info("📝 No existing chat history file found")
            return []
    except Exception as e:
        logger.error(f"❌ Error loading chat history: {e}")
        return []


def save_chat_history(chat_history: List[Dict[str, Any]]) -> None:
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved {len(chat_history)} chat messages to {CHAT_HISTORY_FILE}")
        # Clear cache to force reload
        load_chat_history.cache_clear()
    except Exception as e:
        logger.error(f"❌ Error saving chat history: {e}")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_uploaded_files() -> List[str]:
    """Get list of uploaded files with caching"""
    if not os.path.exists(UPLOADS_DIR):
        logger.warning("❌ Uploads directory not found")
        return []
    
    files = [f for f in os.listdir(UPLOADS_DIR) 
             if os.path.isfile(os.path.join(UPLOADS_DIR, f))]
    logger.info(f"🔍 Found {len(files)} files in uploads directory")
    return files


def auto_reload_documents(rag_pipeline, image_database) -> None:
    """Auto-reload documents from uploads folder if they exist"""
    files = get_uploaded_files()
    if not files:
        logger.warning("❌ No files found in uploads directory")
        return
    
    # Check if we need to reload (no processed files in session state)
    if not st.session_state.get("processed_files"):
        st.session_state.processed_files = []
        
        # Process each file in uploads folder
        for filename in files:
            file_path = os.path.join(UPLOADS_DIR, filename)
            try:
                logger.info(f"🔄 Processing {filename}...")
                
                # Ingest document
                doc_service = DocumentService()
                documents = doc_service.convert_file(file_path)
                rag_pipeline.add_documents(documents)
                
                # Extract images
                image_database.extract_images_from_any_file(file_path, filename)
                
                # Add to processed files list
                st.session_state.processed_files.append(filename)
                logger.info(f"✅ Successfully processed {filename}")
                
            except Exception as e:
                logger.error(f"❌ Error auto-reloading {filename}: {e}")
                continue
        
        if st.session_state.processed_files:
            logger.info(f"✅ Auto-reloaded {len(st.session_state.processed_files)} documents from uploads folder")
    else:
        logger.info(f"✅ Documents already loaded: {len(st.session_state.processed_files)} files")


@st.cache_resource
def initialize_services():
    """Initialize services with caching for better performance"""
    initialize_app()
    app_factory = get_app_factory()
    return app_factory


def main():
    """
    Main application function - Optimized logic
    """
    # Initialize services with caching
    app_factory = initialize_services()
    
    # Download NLTK data once
    download_nltk_data()
    
    # Initialize chat history from file if not in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    # Get services from AppFactory
    rag_pipeline = app_factory.get_rag_pipeline()
    image_database = app_factory.get_image_database()
    config = app_factory.get_config()
    
    # Auto-reload documents from uploads folder only once
    if not st.session_state.get("processed_files") and not st.session_state.get("auto_reloaded"):
        auto_reload_documents(rag_pipeline, image_database)
        st.session_state.auto_reloaded = True
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        
        # Show uploaded files list with optimized display
        if st.session_state.get("processed_files"):
            st.subheader("📄 Files")
            for file_name in st.session_state.processed_files:
                # Truncate long filenames with better logic
                display_name = file_name
                if len(file_name) > MAX_FILENAME_LENGTH:
                    display_name = file_name[:21] + "..." + file_name[-4:]
                st.write(f"• {display_name}")
        else:
            st.write("📄 No files uploaded")
        
        # Add new file button
        if st.button("📁 Thêm file mới", type="primary"):
        st.session_state.force_show_upload = True
            st.experimental_rerun()
        
        # Clear chat button
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
                st.session_state.chat_history = []
                save_chat_history([])
        st.session_state.show_upload = True
        st.experimental_rerun()

    # Main content area
    st.title("🤖 Hybrid RAG Chatbot")
    st.markdown("**Hybrid RAG Pipeline với Haystack Core + LangChain Fallback**")
    
    # File upload section - Only show when needed
    if st.session_state.get("show_upload", True) or st.session_state.get("force_show_upload", False):
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=config.processing.supported_formats,
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, MD, MARKDOWN, XLSX, XLS, PPTX, HTML, HTM, JSON, CSV"
        )
        
        if uploaded_files:
            # Process uploaded files
            process_uploaded_files_old(uploaded_files, rag_pipeline, image_database)
            
            # Hide upload area after processing
            st.session_state.show_upload = False
            st.session_state.force_show_upload = False
            st.experimental_rerun()
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    if st.session_state.get("chat_history"):
        for message in st.session_state.chat_history:
        with st.chat_message("user"):
                st.write(message["question"])
        with st.chat_message("assistant"):
                st.write(message["answer"])
                if message.get("sources"):
                with st.expander("📚 Nguồn tham khảo"):
                        for source in message["sources"]:
                            st.write(f"📄 {source}")
    
    # Chat input
    prompt = st.chat_input("Ask anything...")

    # Process chat input
    if prompt:
        process_chat_input_old(prompt, rag_pipeline, image_database)


def fix_double_extension(filename: str) -> str:
    """Fix common double extension issues"""
    extensions_map = {
        '.pdf.pdf': '.pdf',
        '.docx.docx': '.docx', 
        '.xlsx.xlsx': '.xlsx',
        '.md.md': '.md'
    }
    
    for old_ext, new_ext in extensions_map.items():
        if filename.endswith(old_ext):
            return filename.replace(old_ext, new_ext)
    return filename


def save_uploaded_file(uploaded_file, uploads_dir: str) -> str:
    """Save uploaded file to disk with proper error handling"""
                    if not os.path.exists(uploads_dir):
                        os.makedirs(uploads_dir)
                    
                    # Fix double extensions
    file_name = fix_double_extension(uploaded_file.name)
    file_path = os.path.join(uploads_dir, file_name)
    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
    return file_path, file_name


def process_uploaded_files_old(uploaded_files, rag_pipeline, image_database) -> None:
    """Process uploaded files using optimized logic"""
    try:
        processed_files = []
        failed_files = []
        
        st.info(f"📁 Đang xử lý {len(uploaded_files)} file(s)")
        
        for uploaded_file in uploaded_files:
            try:
                st.info(f"🔄 Đang xử lý: {uploaded_file.name}")
                
                # Save file to uploads folder
                file_path, file_name = save_uploaded_file(uploaded_file, UPLOADS_DIR)
                
                # Process document with RAG pipeline
                doc_service = DocumentService()
                documents = doc_service.convert_file(file_path)
                rag_pipeline.add_documents(documents)
                doc_id = len(documents)
                
                st.success(f"✅ Processed: {file_name}")

                # Extract images
                try:
                    extracted_images = image_database.extract_images_from_any_file(file_path, file_name)
                        if extracted_images:
                        st.success(f"🖼️ Đã trích xuất {len(extracted_images)} ảnh từ {file_name}")
                    except Exception as e:
                        st.warning(f"⚠️ Không thể trích xuất ảnh: {e}")

                    if doc_id:
                    processed_files.append(file_name)
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []
                        if file_name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(file_name)
                    else:
                        failed_files.append(uploaded_file.name)

                except Exception as e:
                    failed_files.append(uploaded_file.name)
                    st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")

            if processed_files:
            st.success(f"🎉 Successfully processed {len(processed_files)} files!")

            if failed_files:
            st.error(f"❌ Failed to process {len(failed_files)} files")
            
    except Exception as e:
        st.error(f"❌ Error processing files: {e}")


def process_chat_input_old(prompt, rag_pipeline, image_database):
    """Process chat input using existing logic"""
    # Hiển thị câu hỏi của user trong chat
        st.chat_message("user").write(prompt)

    # Lấy câu trả lời từ AI sử dụng Hybrid RAG
        with st.chat_message("assistant"):
        try:
            # Sử dụng RAG pipeline để tìm câu trả lời
            result = rag_pipeline.query(prompt)

                    # Display answer
                    st.markdown(result["answer"], unsafe_allow_html=True)

            # Tìm và hiển thị ảnh liên quan
            relevant_images = []
            
            # Lấy source documents để tìm file nào chứa câu trả lời
            source_docs = result.get("source_documents", [])
            source_files = set()
            
            for doc in source_docs:
                # Trích xuất tên file từ document metadata
                if hasattr(doc, 'metadata'):
                    source_file = doc.metadata.get('source', '')
                else:
                    # Thử trích xuất từ document content
                    content = str(doc)
                    import re
                    file_match = re.search(r'uploads[/\\]([^/\s]+)', content)
                    if file_match:
                        source_file = file_match.group(1)
                    else:
                        source_file = ''
                
                if source_file:
                    source_files.add(source_file)
            
            # Tìm ảnh từ cùng source files (ưu tiên cao nhất)
            for source_file in source_files:
                images_from_source = image_database.get_images_by_source(source_file)
                relevant_images.extend(images_from_source)
            
            # Nếu không tìm thấy ảnh từ source files, tìm kiếm theo query
            if not relevant_images:
                relevant_images = image_database.find_relevant_images(prompt, max_results=3)
            
            # Loại bỏ trùng lặp và giới hạn kết quả
            unique_images = []
            seen_paths = set()
            for img in relevant_images:
                if img["path"] not in seen_paths:
                    unique_images.append(img)
                    seen_paths.add(img["path"])
            
            # Hiển thị ảnh (giới hạn 1 ảnh để tránh trùng lặp)
            if unique_images:
                            st.info("🖼️ **Ảnh liên quan trong tài liệu:**")
                display_images = unique_images[:1]  # Chỉ hiển thị 1 ảnh
                
                # Store displayed images for persistence
                st.session_state.last_displayed_images = display_images
                
                # Hiển thị ảnh
                for i, img in enumerate(display_images):
                    st.image(img["path"], caption=img.get("context", "")[:100] + "...")
                                    with open(img["path"], "rb") as fh:
                        st.download_button("📥 Tải ảnh", fh.read(), file_name=img.get("filename", "image.png"), mime="image/png", key=f"download_img_{i}_{img.get('filename', 'image')}")
            
            # Nếu không có ảnh mới, hiển thị ảnh từ câu hỏi trước
            elif st.session_state.get("last_displayed_images"):
                st.info("🖼️ **Ảnh liên quan từ câu hỏi trước:**")
                display_images = st.session_state.last_displayed_images[:1]  # Chỉ 1 ảnh
                for i, img in enumerate(display_images):
                    st.image(img["path"], caption=img.get("context", "")[:100] + "...")
                    with open(img["path"], "rb") as fh:
                        st.download_button("📥 Tải ảnh", fh.read(), file_name=img.get("filename", "image.png"), mime="image/png", key=f"persist_img_{i}_{img.get('filename', 'image')}")

                    # Store sources for later display
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Nguồn tham khảo"):
                            # Lọc và hiển thị tên file gốc duy nhất
                            unique_sources = []
                            for source in sources:
                        if source not in unique_sources:
                            unique_sources.append(source)
                            
                            for i, source in enumerate(unique_sources):
                                st.write(f"**📄 {source}**")

                    # Add to chat history and save
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
                    st.session_state.chat_history.append(
                        {
                            "question": prompt,
                            "answer": result["answer"],
                            "sources": sources,
                    "timestamp": datetime.now().isoformat(),
                        }
                    )
            # Save chat history after each interaction
            save_chat_history(st.session_state.chat_history)

                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")


def extract_source_files(source_documents):
    """Extract source files from documents"""
    source_files = set()
    
    for doc in source_documents:
        if hasattr(doc, 'metadata'):
            source_file = doc.metadata.get('source', '')
        else:
            # Try to extract from document content
            content = str(doc)
            import re
            file_match = re.search(r'uploads[/\\]([^/\s]+)', content)
            if file_match:
                source_file = file_match.group(1)
            else:
                source_file = ''
        
        if source_file:
            source_files.add(source_file)
    
    return list(source_files)


if __name__ == "__main__":
    main()
