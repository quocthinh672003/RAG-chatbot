# -*- coding: utf-8 -*-
"""
🤖 Hybrid RAG Chatbot with Image Support
========================================

📋 Mô tả: Ứng dụng RAG chatbot hybrid với Haystack + LangChain fallback
🎯 Mục đích: Xử lý tài liệu đa định dạng và chat tương tác
🔧 Architecture: Hybrid approach với auto-fallback cho reliability cao

📊 Luồng đi chính:
1. 🚀 Initialization: Load services, NLTK data, chat history
2. 📁 File Management: Upload, process, auto-reload documents
3. 💬 Chat Interface: Process queries với Hybrid RAG pipeline
4. 🖼️ Image Support: Extract và display relevant images
5. 💾 Persistence: Save chat history và session state

🔧 Key Components:
- @st.cache_resource: Cache services để performance
- @lru_cache: Cache chat history để tránh re-reading
- Session State: Quản lý UI state và user data
- Error Handling: Graceful degradation với fallback
- File Processing: Multi-format support với DocumentService

📈 Performance Optimizations:
- Caching strategies cho services và data
- Lazy loading cho documents
- Pagination cho file lists
- Smart search và filtering
- Memory efficient processing

🔄 Hybrid Pipeline Logic:
- Primary: Haystack Core (fast, feature-rich)
- Fallback: LangChain (reliable, simple)
- Auto-switch: Exception-based fallback
- Unified: Consistent result format

📁 File Structure:
- uploads/: Stored uploaded files
- image_database/: Extracted images
- chat_history.json: Persistent chat data
- Session state: Runtime data management

🎨 UI Features:
- Smart file list với search và pagination
- File type icons và truncation
- Real-time chat interface
- Image display với download
- Progress indicators và error handling
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
    """
    📚 Download required NLTK data for markdown processing

    🔄 Luồng đi:
    1. Download punkt: Sentence tokenization
    2. Download averaged_perceptron_tagger: POS tagging
    3. Download maxent_ne_chunker: Named entity recognition
    4. Download words: Word list
    5. Download stopwords: Stop words removal

    📊 Mục đích: Hỗ trợ xử lý văn bản tiếng Việt và tiếng Anh
    """
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
    """
    📚 Load chat history từ JSON file với caching

    🔄 Luồng đi:
    1. Check file tồn tại
    2. Load JSON với UTF-8 encoding
    3. Cache result để tránh re-reading
    4. Return empty list nếu file không tồn tại

    📊 Performance: @lru_cache(maxsize=1) để cache result
    💾 Persistence: UTF-8 encoding cho Vietnamese text
    """
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history
        else:
            return []
    except Exception as e:
        logger.error(f"❌ Error loading chat history: {e}")
        return []


def save_chat_history(chat_history: List[Dict[str, Any]]) -> None:
    """
    💾 Save chat history vào JSON file

    🔄 Luồng đi:
    1. Write JSON với UTF-8 encoding
    2. Clear cache để force reload
    3. Log success/failure

    📊 Format: JSON với ensure_ascii=False cho Vietnamese
    🔄 Cache: Clear cache sau khi save để consistency
    """
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        # Clear cache to force reload
        load_chat_history.cache_clear()
    except Exception as e:
        logger.error(f"❌ Error saving chat history: {e}")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_uploaded_files() -> List[str]:
    """
    📁 Get list of uploaded files với caching

    🔄 Luồng đi:
    1. Check uploads directory tồn tại
    2. List all files trong directory
    3. Filter chỉ files (không folders)
    4. Cache result cho 1 hour

    📊 Performance: @st.cache_data(ttl=3600) để cache 1 giờ
    🔍 Purpose: Tránh re-scanning directory mỗi lần
    """
    if not os.path.exists(UPLOADS_DIR):
        return []

    files = [
        f
        for f in os.listdir(UPLOADS_DIR)
        if os.path.isfile(os.path.join(UPLOADS_DIR, f))
    ]
    return files


def auto_reload_documents(rag_pipeline, image_database):
    """Auto-reload documents from uploads folder with minimal logging"""
    if not os.path.exists(UPLOADS_DIR):
        return

    files = [
        f
        for f in os.listdir(UPLOADS_DIR)
        if os.path.isfile(os.path.join(UPLOADS_DIR, f))
    ]

    if not files:
        return

    # Suppress ALL detailed logging
    import logging

    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)  # Only critical errors

    try:
        for file_name in files:
            file_path = os.path.join(UPLOADS_DIR, file_name)

            # Process document silently
            documents = DocumentService().convert_file(file_path)
            if documents and rag_pipeline and hasattr(rag_pipeline, "add_documents"):
                rag_pipeline.add_documents(documents)

                # Extract images silently
                try:
                    image_database.extract_images_from_any_file(file_path, file_name)
                except:
                    pass  # Silently ignore image extraction errors

                # Update session state
                if "processed_files" not in st.session_state:
                    st.session_state.processed_files = []
                if file_name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(file_name)

        # Only show final success message
        logger.info(f"✅ Loaded {len(files)} documents")

    finally:
        # Restore logging level
        logging.getLogger().setLevel(original_level)


@st.cache_resource
def initialize_services():
    """
    🔧 Initialize services với caching cho performance tối ưu

    🔄 Luồng đi:
    1. Gọi initialize_app() từ app_factory
    2. Lấy AppFactory instance
    3. Cache result để tránh re-initialization
    4. Return factory để access services

    📊 Performance: @st.cache_resource để cache services
    🔄 Purpose: Tránh re-initializing services mỗi lần
    """
    initialize_app()
    app_factory = get_app_factory()
    return app_factory


def debug_session_state():
    """Debug session state with minimal output"""
    # Only log if there are issues
    if not st.session_state.get("processed_files"):
        logger.info("📄 No files in session state")

    # Only show files in uploads dir if needed for debugging
    if os.path.exists(UPLOADS_DIR):
        files = [
            f
            for f in os.listdir(UPLOADS_DIR)
            if os.path.isfile(os.path.join(UPLOADS_DIR, f))
        ]
        if not files:
            logger.info("📁 Uploads directory is empty")


def test_text_processing():
    """
    🔍 Test text processing functions directly
    """
    from services.rag_pipeline import TextProcessor

    test_query = "OKVIP - TUYỂN DỤNG IT DEV"
    test_content = "OKVIP - TUYỂN DỤNG IT DEV (APP DEVELOPER) MÔ TẢ CÔNG VIỆC: THAM GIA PHÁT TRIỂN VÀ BẢO TRÌ ỨNG DỤNG DI ĐỘNG"

    st.write("🔍 Text Processing Test:")
    st.write(f"Original query: '{test_query}'")
    st.write(f"Original content: '{test_content}'")

    # Test normalization
    normalized_query = TextProcessor.normalize_text(test_query)
    normalized_content = TextProcessor.normalize_text(test_content)

    st.write(f"Normalized query: '{normalized_query}'")
    st.write(f"Normalized content: '{normalized_content}'")

    # Test search variations
    variations = TextProcessor.create_search_variations(test_query)
    st.write(f"Search variations: {variations}")

    # Test matching
    matches = []
    for variation in variations:
        if variation in normalized_content:
            matches.append(variation)

    st.write(f"Matches found: {matches}")

    return matches


def main():
    """
    🚀 Main application function - Khởi tạo và quản lý toàn bộ ứng dụng

    🔄 Luồng đi:
    1. Initialize services với caching (@st.cache_resource)
    2. Download NLTK data một lần duy nhất
    3. Load chat history từ file JSON
    4. Auto-reload documents từ uploads folder
    5. Render UI components:
       - Sidebar với file management
       - Main content với chat interface
       - File upload section
       - Chat history display

    📊 Session State Management:
    - chat_history: Chat messages
    - processed_files: Uploaded files
    - auto_reloaded: Auto-reload flag
    - force_show_upload: Show upload area
    - show_all_files: File list pagination
    - last_displayed_images: Last shown images
    - file_search: File search term

    🎨 UI Components:
    - Smart file list với search và pagination
    - File type icons và truncation
    - Real-time chat interface
    - Image display với download
    - Progress indicators và error handling
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

    # Debug session state
    debug_session_state()

    # Auto-reload documents if not already done
    if not st.session_state.get("auto_reloaded", False):
        if rag_pipeline:
            auto_reload_documents(rag_pipeline, image_database)
        st.session_state.auto_reloaded = True

    # Display files in sidebar (no logging needed)
    if st.session_state.get("processed_files"):
        pass  # Files will be displayed in sidebar

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")

        # Show uploaded files list with scrollable container
        if st.session_state.get("processed_files"):
            st.subheader("📄 Files")

            # Add search/filter for files
            search_term = st.text_input(
                "🔍 Tìm file...", placeholder="Nhập tên file", key="file_search"
            )

            # Filter files based on search
            files_to_show = st.session_state.processed_files
            if search_term:
                files_to_show = [
                    f for f in files_to_show if search_term.lower() in f.lower()
                ]

            # Show file count
            st.caption(
                f"📊 {len(files_to_show)}/{len(st.session_state.processed_files)} files"
            )

            # Scrollable container for files
            with st.container():
                # Limit display to first 10 files, show more button if needed
                max_display = 10
                files_display = files_to_show[:max_display]

                for i, file_name in enumerate(files_display):
                    # Truncate long filenames
                    display_name = file_name
                    if len(file_name) > 25:
                        display_name = file_name[:20] + "..." + file_name[-4:]

                    # Add file type icon
                    file_ext = (
                        file_name.split(".")[-1].lower() if "." in file_name else "txt"
                    )
                    icon_map = {
                        "pdf": "📄",
                        "docx": "📝",
                        "xlsx": "📊",
                        "pptx": "📈",
                        "txt": "📃",
                        "md": "📋",
                        "json": "📋",
                        "csv": "📊",
                    }
                    icon = icon_map.get(file_ext, "📄")

                    st.write(f"{icon} {display_name}")

                # Show "more files" button if needed
                if len(files_to_show) > max_display:
                    if st.button(
                        f"📋 Xem thêm {len(files_to_show) - max_display} files...",
                        key="show_more_files",
                    ):
                        st.session_state.show_all_files = True
                        st.rerun()

                # Show all files if requested
                if st.session_state.get("show_all_files", False):
                    st.write("---")
                    for file_name in files_to_show[max_display:]:
                        display_name = file_name
                        if len(file_name) > 25:
                            display_name = file_name[:20] + "..." + file_name[-4:]
                        file_ext = (
                            file_name.split(".")[-1].lower()
                            if "." in file_name
                            else "txt"
                        )
                        icon = icon_map.get(file_ext, "📄")
                        st.write(f"{icon} {display_name}")

                    if st.button("📋 Thu gọn", key="collapse_files"):
                        st.session_state.show_all_files = False
                        st.rerun()
        else:
            st.write("📄 No files uploaded")

        # Add new file button
        if st.button("📁 Thêm file mới", type="primary"):
            st.session_state.force_show_upload = True
            st.rerun()

        # Delete all files button
        if st.session_state.get("processed_files"):
            if st.button(
                "🗑️ Xóa hết file",
                type="secondary",
                help="Xóa tất cả files khỏi database và uploads",
            ):
                # Clear files from database
                if rag_pipeline:
                    rag_pipeline.clear_documents()

                # Clear files from uploads folder
                import shutil

                if os.path.exists(UPLOADS_DIR):
                    shutil.rmtree(UPLOADS_DIR)
                    os.makedirs(UPLOADS_DIR)

                # Clear session state
                st.session_state.processed_files = []
                st.session_state.auto_reloaded = False

                st.success("✅ Đã xóa hết files!")
                st.rerun()

        # Clear chat button
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
                st.session_state.chat_history = []
                save_chat_history([])
                st.rerun()

        # Debug button
        if st.button("🔍 Debug Session", type="secondary", help="Debug session state"):
            st.write("🔍 Debug Info:")
            st.write(
                f"- processed_files: {st.session_state.get('processed_files', 'NOT SET')}"
            )
            st.write(
                f"- auto_reloaded: {st.session_state.get('auto_reloaded', 'NOT SET')}"
            )
            st.write(
                f"- force_show_upload: {st.session_state.get('force_show_upload', 'NOT SET')}"
            )

            if os.path.exists(UPLOADS_DIR):
                files = os.listdir(UPLOADS_DIR)
                st.write(f"- Files in uploads dir: {files}")
            else:
                st.write("- Uploads directory does not exist")

        # Debug RAG Pipeline button
        if st.button(
            "🔍 Debug RAG Pipeline", type="secondary", help="Debug RAG pipeline search"
        ):
            if rag_pipeline:
                st.write("🔍 RAG Pipeline Debug:")

                # Test direct search
                test_query = "OKVIP - TUYỂN DỤNG IT DEV"
                st.write(f"Testing query: '{test_query}'")

                # Get all documents
                try:
                    all_docs = rag_pipeline.document_store.get_all_documents()
                    st.write(f"Total documents in store: {len(all_docs)}")

                    # Show first few documents
                    for i, doc in enumerate(all_docs[:3]):
                        st.write(f"Document {i+1}:")
                        st.write(f"Content: {doc.content[:300]}...")
                        st.write(f"Meta: {doc.meta}")
                        st.write("---")

                    # Test direct retrieval
                    try:
                        if hasattr(rag_pipeline.retriever, "retrieve"):
                            retrieved = rag_pipeline.retriever.retrieve(test_query)
                            st.write(f"Retrieved documents: {len(retrieved)}")
                            for i, doc in enumerate(retrieved[:2]):
                                st.write(f"Retrieved {i+1}: {doc.content[:200]}...")
                    except Exception as e:
                        st.error(f"Error debugging RAG: {e}")
                except Exception as e:
                    st.error(f"Error debugging RAG: {e}")
        else:
            st.error("RAG Pipeline not available")

        # Test Text Processing button
        if st.button(
            "🔍 Test Text Processing",
            type="secondary",
            help="Test text processing functions",
        ):
            test_text_processing()

        # Force Reload Documents button
        if st.button(
            "🔄 Force Reload Documents",
            type="secondary",
            help="Force clear and reload all documents",
        ):
            if rag_pipeline:
                # Clear documents
                rag_pipeline.clear_documents()
                st.success("✅ Cleared documents")

                # Clear session state
                st.session_state.processed_files = []
                st.session_state.auto_reloaded = False

                # Reload documents
                auto_reload_documents(rag_pipeline, image_database)
                st.session_state.auto_reloaded = True

                st.success("✅ Documents reloaded!")
                st.rerun()
            else:
                st.error("RAG Pipeline not available")

        # Reset session button
        if st.button(
            "🔄 Reset Session", type="secondary", help="Reset all session state"
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Session state reset!")
            st.rerun()

    # Main content area
    st.title("🤖 Hybrid RAG Chatbot")
    st.markdown("**Hybrid RAG Pipeline với Haystack Core + LangChain Fallback**")

    # Show files summary when files exist
    if st.session_state.get("processed_files"):
        st.subheader("📚 Documents Ready")
        st.success(
            f"✅ {len(st.session_state.processed_files)} documents loaded and ready for chat!"
        )

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
        if rag_pipeline is None:
            st.error(
                "❌ RAG Pipeline không khả dụng. Vui lòng kiểm tra cài đặt Haystack."
            )
        else:
            process_chat_input_old(prompt, rag_pipeline, image_database)

    # File upload section - Only show when no files or force show
    has_files = bool(st.session_state.get("processed_files"))
    force_show = st.session_state.get("force_show_upload", False)

    if not has_files or force_show:
        st.subheader("📁 Upload Documents")
        st.info(
            "💡 Upload documents để chat với AI. Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, CSV, HTML, JSON"
        )

        uploaded_files = st.file_uploader(
            "Choose files",
            type=config.processing.supported_formats,
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, MD, MARKDOWN, XLSX, XLS, PPTX, HTML, HTM, JSON, CSV",
        )

        if uploaded_files:
            # Process uploaded files
            process_uploaded_files_old(uploaded_files, rag_pipeline, image_database)

            # Hide upload area after processing
            st.session_state.force_show_upload = False
            st.rerun()


def fix_double_extension(filename: str) -> str:
    """
    🔧 Fix common double extension issues

    🔄 Luồng đi:
    1. Define extensions mapping
    2. Check filename có double extension
    3. Replace với single extension
    4. Return fixed filename

    📊 Purpose: Xử lý lỗi upload file có double extension
    🔍 Common cases: .pdf.pdf, .docx.docx, .xlsx.xlsx, .md.md
    """
    extensions_map = {
        ".pdf.pdf": ".pdf",
        ".docx.docx": ".docx",
        ".xlsx.xlsx": ".xlsx",
        ".md.md": ".md",
    }

    for old_ext, new_ext in extensions_map.items():
        if filename.endswith(old_ext):
            return filename.replace(old_ext, new_ext)
    return filename


def save_uploaded_file(uploaded_file, uploads_dir: str) -> str:
    """
    💾 Save uploaded file to disk với proper error handling

    🔄 Luồng đi:
    1. Create uploads directory nếu không tồn tại
    2. Fix double extensions
    3. Generate file path
    4. Write file content
    5. Return file path

    📊 Error Handling: Create directory nếu không tồn tại
    🔧 File Processing: Fix double extensions trước khi save
    """
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Fix double extensions
    file_name = fix_double_extension(uploaded_file.name)
    file_path = os.path.join(uploads_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return file_path


def process_uploaded_files_old(uploaded_files, rag_pipeline, image_database) -> None:
    """Process uploaded files with enhanced error handling and status display"""
    if not uploaded_files:
        return

    # Initialize processed_files if not exists
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    processed_count = 0
    total_files = len(uploaded_files)

    # Create a status container
    status_container = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Show simple status
            status_container.info(
                f"🔄 Đang xử lý: {uploaded_file.name} ({i+1}/{total_files})"
            )

            # Save file
            file_path = save_uploaded_file(uploaded_file, UPLOADS_DIR)

            # Process with document service
            documents = DocumentService().convert_file(file_path)

            if documents:
                # Add to RAG pipeline
                rag_pipeline.add_documents(documents)

                # Extract images
                images = image_database.extract_images_from_any_file(
                    file_path, uploaded_file.name
                )

                # Update session state
                if uploaded_file.name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(uploaded_file.name)

                processed_count += 1

                # Show success status
                status_container.success(f"✅ Đã xử lý: {uploaded_file.name}")

                # Show image extraction status if images found
                if images:
                    status_container.info(
                        f"🖼️ Đã trích xuất {len(images)} ảnh từ {uploaded_file.name}"
                    )

            else:
                status_container.warning(f"⚠️ Không thể xử lý: {uploaded_file.name}")

        except Exception as e:
            logger.error(f"❌ Error processing {uploaded_file.name}: {e}")
            status_container.error(f"❌ Lỗi xử lý: {uploaded_file.name}")

    # Final success message
    if processed_count > 0:
        status_container.success(f"🚀 Đã xử lý thành công {processed_count} file!")
        st.rerun()
    else:
        status_container.error("❌ Không có file nào được xử lý thành công")


def process_chat_input_old(prompt, rag_pipeline, image_database):
    """
    💬 Process chat input với Hybrid RAG

    🔄 Luồng đi:
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

    🎯 Hybrid Pipeline:
    - Primary: Haystack Core (fast, feature-rich)
    - Fallback: LangChain (reliable, simple)
    - Auto-switch: Exception-based fallback

    🖼️ Image Display Logic:
    - Priority 1: Images từ source files
    - Priority 2: Query-based image search
    - Priority 3: Images từ previous question
    - Limit: 1 image để tránh clutter

    💾 Persistence:
    - Save chat history sau mỗi interaction
    - Store displayed images cho persistence
    - UTF-8 encoding cho Vietnamese text
    """
    # Hiển thị câu hỏi của user trong chat
    st.chat_message("user").write(prompt)

    # Lấy câu trả lời từ AI sử dụng Hybrid RAG
    with st.chat_message("assistant"):
        try:
            # Sử dụng RAG pipeline để tìm câu trả lời
            result = rag_pipeline.query(prompt)

            # Display answer
            st.markdown(result["answer"], unsafe_allow_html=True)

            # Only show images if RAG found relevant information
            if "Không tìm thấy thông tin" not in result["answer"]:
                # Tìm và hiển thị ảnh liên quan
                relevant_images = []

                # Lấy source documents để tìm file nào chứa câu trả lời
                source_docs = result.get("source_documents", [])
                source_files = set()

                for doc in source_docs:
                    # Trích xuất tên file từ document metadata
                    if hasattr(doc, "metadata"):
                        source_file = doc.metadata.get("source", "")
                    else:
                        # Thử trích xuất từ document content
                        content = str(doc)
                        import re

                        file_match = re.search(r"uploads[/\\]([^/\s]+)", content)
                        if file_match:
                            source_file = file_match.group(1)
                        else:
                            source_file = ""

                    if source_file:
                        source_files.add(source_file)

                # Tìm ảnh từ cùng source files (ưu tiên cao nhất)
                for source_file in source_files:
                    images_from_source = image_database.get_images_by_source(
                        source_file
                    )
                    relevant_images.extend(images_from_source)

                # Nếu không tìm thấy ảnh từ source files, tìm kiếm theo query
                if not relevant_images:
                    relevant_images = image_database.find_relevant_images(
                        prompt, max_results=3
                    )

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
                        st.image(
                            img["path"], caption=img.get("context", "")[:100] + "..."
                        )
                        with open(img["path"], "rb") as fh:
                            st.download_button(
                                "📥 Tải ảnh",
                                fh.read(),
                                file_name=img.get("filename", "image.png"),
                                mime="image/png",
                                key=f"download_img_{i}_{img.get('filename', 'image')}",
                            )

                # Nếu không có ảnh mới, hiển thị ảnh từ câu hỏi trước
                elif st.session_state.get("last_displayed_images"):
                    st.info("🖼️ **Ảnh liên quan từ câu hỏi trước:**")
                    display_images = st.session_state.last_displayed_images[
                        :1
                    ]  # Chỉ 1 ảnh
                    for i, img in enumerate(display_images):
                        st.image(
                            img["path"], caption=img.get("context", "")[:100] + "..."
                        )
                        with open(img["path"], "rb") as fh:
                            st.download_button(
                                "📥 Tải ảnh",
                                fh.read(),
                                file_name=img.get("filename", "image.png"),
                                mime="image/png",
                                key=f"persist_img_{i}_{img.get('filename', 'image')}",
                            )
            else:
                # Clear last displayed images when no relevant info found
                if "last_displayed_images" in st.session_state:
                    del st.session_state.last_displayed_images

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
    """
    📄 Extract source files từ documents

    🔄 Luồng đi:
    1. Iterate qua source documents
    2. Extract source file từ metadata
    3. Fallback to content parsing nếu không có metadata
    4. Use regex để extract filename từ path
    5. Return unique list of source files

    📊 Purpose: Tìm source files để image search
    🔍 Fallback: Parse content nếu metadata không có
    🎯 Result: List unique source files
    """
    source_files = set()

    for doc in source_documents:
        if hasattr(doc, "metadata"):
            source_file = doc.metadata.get("source", "")
        else:
            # Try to extract from document content
            content = str(doc)
            import re

            file_match = re.search(r"uploads[/\\]([^/\s]+)", content)
            if file_match:
                source_file = file_match.group(1)
            else:
                source_file = ""

        if source_file:
            source_files.add(source_file)

    return list(source_files)


if __name__ == "__main__":
    main()
