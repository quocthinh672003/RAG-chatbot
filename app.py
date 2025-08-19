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

# Reduce noisy third‑party logs (httpx/OpenAI/Weaviate/Haystack)
for _name in ["httpx", "haystack", "weaviate", "openai", "urllib3"]:
    try:
        logging.getLogger(_name).setLevel(logging.WARNING)
    except Exception:
        pass

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


def safe_rerun() -> None:
    """Cross-version rerun wrapper for Streamlit."""
    try:
        fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if callable(fn):
            fn()
    except Exception:
        pass

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


def restore_chat_from_weaviate(rag_pipeline) -> int:
    """Load chats from Weaviate 'Chats' collection into session and persist to file."""
    try:
        if not hasattr(rag_pipeline, "document_store"):
            return 0
        store = rag_pipeline.document_store
        if not hasattr(store, "client"):
            return 0
        coll = store.client.collections.get("Chats")
        rows: List[Dict[str, Any]] = []
        for obj in coll.iterator():
            props = getattr(obj, "properties", {}) or {}
            rows.append({
                "question": props.get("question", ""),
                "answer": props.get("answer", ""),
                "sources": props.get("sources", "").split(", ") if props.get("sources") else [],
                "timestamp": props.get("timestamp", ""),
            })
        rows.sort(key=lambda r: r.get("timestamp", ""))
        st.session_state.chat_history = rows
        save_chat_history(rows)
        return len(rows)
    except Exception as e:
        logger.error(f"❌ Restore chats from Weaviate failed: {e}")
        return 0


def clear_weaviate_chats(rag_pipeline) -> bool:
    """Delete Chats collection on Weaviate and recreate empty one."""
    try:
        if not hasattr(rag_pipeline, "document_store"):
            return False
        store = rag_pipeline.document_store
        if not hasattr(store, "client"):
            return False
        try:
            store.client.collections.delete("Chats")
        except Exception:
            pass
        if hasattr(store, "_ensure_collection"):
            store._ensure_collection()
        return True
    except Exception as e:
        logger.error(f"❌ Clear Weaviate chats failed: {e}")
        return False


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
    # DISABLED: This function is disabled to prevent infinite loops
    # Documents are now loaded manually in main()
    pass


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



    # DISABLED: Auto-reload documents to prevent infinite loop
    # Only set flag to prevent future calls
    if not st.session_state.get("auto_reloaded", False):
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
                        safe_rerun()

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
                        safe_rerun()
        else:
            st.write("📄 No files uploaded")

        # Add new file button
        if st.button("📁 Thêm file mới", type="primary"):
            st.session_state.force_show_upload = True
            safe_rerun()

        # (Removed) Manual process button for uploads per user request

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

                # Also remove all extracted images that came from these files
                try:
                    # Remove by each processed file name
                    for fname in list(st.session_state.get("processed_files", [])):
                        try:
                            image_database.remove_images_by_source(fname)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Clear files from uploads folder
                import shutil

                if os.path.exists(UPLOADS_DIR):
                    shutil.rmtree(UPLOADS_DIR)
                    os.makedirs(UPLOADS_DIR)

                # Clear session state
                st.session_state.processed_files = []
                st.session_state.auto_reloaded = False

                st.success("✅ Đã xóa hết files!")
                safe_rerun()

        # Clear chat button
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_history = []
            save_chat_history([])
            safe_rerun()

        # (Removed) Buttons for Weaviate chat operations per user request



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
                # Persisted images per message
                imgs = message.get("images") or []
                if imgs:
                    st.info("🖼️ **Ảnh liên quan trong tài liệu:**")
                    for i, img in enumerate(imgs[:1]):  # show at most 1 image per message
                        try:
                            img_path = img.get("path") or ""
                            img_name = img.get("filename", "")
                            # Try to auto-fix double extensions if file not found
                            def _normalize_path(p: str) -> str:
                                if not isinstance(p, str):
                                    return ""
                                for ext in [".pdf", ".docx", ".xlsx", ".md", ".pptx", ".csv", ".txt"]:
                                    double = f"{ext}{ext}"
                                    if p.endswith(double):
                                        return p[:-len(ext)]
                                return p
                            display_path = img_path
                            if not os.path.exists(display_path):
                                fixed_path = _normalize_path(display_path)
                                if fixed_path != display_path and os.path.exists(fixed_path):
                                    display_path = fixed_path
                                    # Update in-memory history so subsequent renders use the fixed path
                                    img["path"] = fixed_path
                                    # Also fix filename if needed
                                    if img_name:
                                        for ext in [".pdf", ".docx", ".xlsx", ".md", ".pptx", ".csv", ".txt"]:
                                            double = f"{ext}{ext}"
                                            if img_name.endswith(double):
                                                img["filename"] = img_name[:-len(ext)]
                                                break

                            if os.path.exists(display_path):
                                st.image(display_path, caption=(img.get("context", "")[:100] + "..."))
                            else:
                                st.warning(f"⚠️ Ảnh không tồn tại: {img.get('filename', 'Unknown')}")
                        except Exception as e:
                            st.warning(f"⚠️ Lỗi hiển thị ảnh: {str(e)}")
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
            "💡 Upload documents để chat với AI. Hỗ trợ: PDF, DOCX, MD, XLSX"
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
            safe_rerun()


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
    # More comprehensive extension mapping
    extensions_map = {
        ".pdf.pdf": ".pdf",
        ".docx.docx": ".docx",
        ".xlsx.xlsx": ".xlsx",
        ".md.md": ".md",
        ".txt.txt": ".txt",
        ".csv.csv": ".csv",
        ".json.json": ".json",
        ".html.html": ".html",
        ".htm.htm": ".htm",
        ".pptx.pptx": ".pptx",
        ".xls.xls": ".xls",
    }

    # Check for double extensions
    for old_ext, new_ext in extensions_map.items():
        if filename.endswith(old_ext):
            return filename.replace(old_ext, new_ext)
    
    # Also check for any double extension pattern
    import re
    # Pattern to match double extensions like .ext.ext
    double_ext_pattern = r'\.([^.]+)\.\1$'
    match = re.search(double_ext_pattern, filename)
    if match:
        ext = match.group(1)
        return filename.replace(f'.{ext}.{ext}', f'.{ext}')
    
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
            # Show simple status with fixed filename
            fixed_filename = fix_double_extension(uploaded_file.name)
            status_container.info(
                f"🔄 Đang xử lý: {fixed_filename} ({i+1}/{total_files})"
            )

            # Save file
            file_path = save_uploaded_file(uploaded_file, UPLOADS_DIR)

            # Process with document service
            documents = DocumentService().convert_file(file_path)

            if documents and rag_pipeline:
                # Add to RAG pipeline
                rag_pipeline.add_documents(documents)

                # Extract images
                # Use the fixed filename (the one actually saved) so source matching works
                images = image_database.extract_images_from_any_file(
                    file_path, fix_double_extension(uploaded_file.name)
                )

                # Update session state with fixed filename
                fixed_filename = fix_double_extension(uploaded_file.name)
                if fixed_filename not in st.session_state.processed_files:
                    st.session_state.processed_files.append(fixed_filename)

                processed_count += 1

                # Show success status
                status_container.success(f"✅ Đã xử lý: {fixed_filename}")

                # Show image extraction status if images found
                if images:
                    status_container.info(
                        f"🖼️ Đã trích xuất {len(images)} ảnh từ {fixed_filename}"
                    )

            else:
                status_container.warning(f"⚠️ Không thể xử lý: {fixed_filename}")

        except Exception as e:
            logger.error(f"❌ Error processing {fixed_filename}: {e}")
            status_container.error(f"❌ Lỗi xử lý: {fixed_filename}")

    # Final success message
    if processed_count > 0:
        status_container.success(f"🚀 Đã xử lý thành công {processed_count} file!")
        safe_rerun()
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
            if not isinstance(result, dict):
                result = {"answer": str(result), "sources": []}

            # Display answer (parse JSON to render tables in Markdown when available)
            displayed_answer = str(result.get("answer", ""))
            parsed = None
            try:
                parsed = json.loads(displayed_answer)
            except Exception:
                try:
                    start = displayed_answer.find("{")
                    end = displayed_answer.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(displayed_answer[start : end + 1])
                except Exception:
                    parsed = None

            if isinstance(parsed, dict):
                ans = parsed.get("answer") or parsed.get("short_answer")
                det = parsed.get("details")
                tables = parsed.get("tables") or []
                if ans:
                    st.markdown(ans)
                if det:
                    st.markdown(det)
                for tbl in tables:
                    if isinstance(tbl, str) and "|" in tbl:
                        st.markdown(tbl)

                # Merge JSON sources with pipeline sources for display
                json_sources = []
                for s in parsed.get("sources", []) or []:
                    if isinstance(s, dict):
                        label = s.get("file") or s.get("source") or "Unknown"
                    else:
                        label = str(s)
                    if label:
                        json_sources.append(label)
                result["sources"] = list({*(result.get("sources", []) or []), *json_sources})

                # Prepare concise answer for history
                displayed_answer = (ans or "").strip()
                if det:
                    displayed_answer += ("\n\n" + det.strip())
            else:
                st.markdown(displayed_answer, unsafe_allow_html=True)

            # Only show images when the user explicitly asks about images
            image_query = any(k in (prompt or "").lower() for k in ["hình", "ảnh", "hình ảnh", "image", "picture", "photo"])
            if image_query and ("Không tìm thấy thông tin" not in result.get("answer", "")):
                # Collect source files from returned documents
                source_files = set()
                docs = result.get("documents", []) if isinstance(result, dict) else []
                for doc in docs:
                    meta = getattr(doc, "meta", None) or getattr(doc, "metadata", None) or {}
                    if isinstance(meta, dict):
                        src = meta.get("source", "")
                        if src:
                            try:
                                import os as _os
                                src = _os.path.basename(src)
                            except Exception:
                                pass
                            source_files.add(src)

                # If no documents or no sources extracted from documents, also try using labeled sources
                if not source_files:
                    labeled_sources = result.get("sources", []) if isinstance(result, dict) else []
                    for label in labeled_sources:
                        try:
                            src_label = str(label)
                            import os as _os
                            src_label = _os.path.basename(src_label)
                            if src_label:
                                source_files.add(src_label)
                        except Exception:
                            continue

                # Fetch images strictly by matched source files
                relevant_images = []
                for source_file in source_files:
                    relevant_images.extend(image_database.get_images_by_source(source_file))


                # Show at most one image if any matched
                if relevant_images:
                    unique_paths = set()
                    display_images = []
                    for img in relevant_images:
                        if img.get("path") and img["path"] not in unique_paths:
                            display_images.append(img)
                            unique_paths.add(img["path"])
                        if len(display_images) >= 1:
                            break

                    if display_images:
                        st.info("🖼️ **Ảnh liên quan trong tài liệu:**")
                        for i, img in enumerate(display_images):
                            try:
                                # Normalize duplicate extensions anywhere in path/filename
                                def _normalize_double_ext(s: str) -> str:
                                    if not isinstance(s, str):
                                        return ""
                                    exts = [".pdf", ".docx", ".xlsx", ".md", ".pptx", ".csv", ".txt"]
                                    changed = True
                                    while changed:
                                        changed = False
                                        for ext in exts:
                                            double = f"{ext}{ext}"
                                            if double in s:
                                                s = s.replace(double, ext)
                                                changed = True
                                    return s

                                display_path = _normalize_double_ext(img.get("path", ""))
                                display_name = _normalize_double_ext(img.get("filename", ""))

                                # Check if image file exists before displaying
                                if os.path.exists(display_path):
                                    st.image(display_path, caption=img.get("context", "")[:100] + "...")
                                    with open(display_path, "rb") as fh:
                                        st.download_button(
                                            "📥 Tải ảnh",
                                            fh.read(),
                                            file_name=(display_name or img.get("filename", "image.png")),
                                            mime="image/png",
                                            key=f"download_img_{i}_{img.get('filename', 'image')}",
                                        )
                                else:
                                    st.warning(f"⚠️ Ảnh không tồn tại: {display_name or img.get('filename', 'Unknown')}")
                            except Exception as e:
                                st.warning(f"⚠️ Lỗi hiển thị ảnh: {str(e)}")
                        # Persist images with this message
                        persisted_imgs = [
                            {"path": im.get("path"), "filename": im.get("filename"), "context": im.get("context", "")}
                            for im in display_images
                        ]
                    else:
                        persisted_imgs = []
                else:
                    persisted_imgs = []
            else:
                persisted_imgs = []
            # Always show sources and save history (regardless of found/not found)
            sources = result.get("sources", [])
            if sources:
                with st.expander("📚 Nguồn tham khảo"):
                    unique_sources = []
                    for source in sources:
                        if source not in unique_sources:
                            unique_sources.append(source)
                    for i, source in enumerate(unique_sources):
                        st.write(f"**📄 {source}**")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append(
                {
                    "question": prompt,
                    "answer": displayed_answer or result.get("answer", ""),
                    "sources": sources,
                    "images": persisted_imgs,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            save_chat_history(st.session_state.chat_history)

            # Persist chat to Weaviate if available
            try:
                if hasattr(rag_pipeline, "document_store") and hasattr(rag_pipeline.document_store, "write_chat_interaction"):
                    rag_pipeline.document_store.write_chat_interaction(
                        question=prompt,
                        answer=displayed_answer or result.get("answer", ""),
                        sources=sources,
                        timestamp=datetime.now().isoformat(),
                    )
            except Exception as e:
                logger.warning(f"Could not persist chat to Weaviate: {e}")

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
