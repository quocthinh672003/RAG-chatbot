import streamlit as st
import os
import tempfile
from ingest import ingest_document
from query import retrieve_and_answer

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("🤖 RAG Chatbot - Hệ thống Hỏi Đáp Thông Minh")
    st.markdown("---")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Tài liệu")
        
        uploaded_files = st.file_uploader(
            "Chọn file tài liệu",
            type=['pdf', 'docx', 'txt', 'md', 'markdown', 'xlsx', 'xls', 'pptx', 'html', 'htm', 'json', 'csv'],
            accept_multiple_files=True,
            help="Hỗ trợ: PDF, DOCX, TXT, MD, XLSX, PPTX, HTML, JSON, CSV"
        )
        
        if uploaded_files:
            st.write(f"Đã chọn {len(uploaded_files)} file(s)")
            
            if st.button("🚀 Xử lý và Lưu trữ", type="primary"):
                with st.spinner("Đang xử lý tài liệu..."):
                    for uploaded_file in uploaded_files:
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Ingest document
                            doc_id = ingest_document(tmp_file_path)
                            
                            # Clean up temp file
                            os.unlink(tmp_file_path)
                            
                            st.success(f"✅ Đã xử lý: {uploaded_file.name} (ID: {doc_id[:8]}...)")
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat với AI")
        
        # Chat input
        user_question = st.text_area(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Nội dung chính của tài liệu là gì?",
            height=100
        )
        
        # Advanced options
        with st.expander("⚙️ Tùy chọn nâng cao"):
            top_k = st.slider("Số lượng nguồn tham khảo:", 3, 20, 10)
            
            # File type filter
            file_types = ["Tất cả", "PDF", "DOCX", "TXT", "MD", "XLSX", "JSON", "CSV", "HTML"]
            selected_file_type = st.selectbox("Lọc theo loại file:", file_types)
        
        if st.button("🔍 Tìm kiếm và Trả lời", type="primary") and user_question:
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    result = retrieve_and_answer(user_question, top_k)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "total_sources": result["total_sources"]
                    })
                    
                    # Display answer
                    st.subheader("🤖 Câu trả lời:")
                    st.write(result["answer"])
                    
                    # Display sources with enhanced info
                    if result["sources"]:
                        st.subheader(f"📚 Nguồn tham khảo ({result['total_sources']} nguồn):")
                        for source in result["sources"]:
                            with st.expander(f"📄 #{source['rank']} - {source['source']} ({source['file_type'].upper()})"):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.write(f"**Nội dung:** {source['content']}")
                                with col_b:
                                    st.write(f"**Trang:** {source['page']}")
                                    if source['score']:
                                        st.write(f"**Độ liên quan:** {source['score']:.3f}")
                                    st.write(f"**Kích thước:** {source['file_size']} bytes")
                                    st.write(f"**Processor:** {source['processor_type']}")
                                    st.write(f"**Chunk size:** {source['chunk_size']}")
                                    st.write(f"**Chunk overlap:** {source['chunk_overlap']}")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
    
    with col2:
        st.header("📋 Lịch sử Chat")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💬 {chat['question'][:50]}..."):
                    st.write(f"**Câu hỏi:** {chat['question']}")
                    st.write(f"**Trả lời:** {chat['answer']}")
                    st.write(f"**Số nguồn:** {chat['total_sources']}")
                    
                    # Show file types used
                    if chat['sources']:
                        file_types = list(set([s['file_type'] for s in chat['sources']]))
                        st.write(f"**Loại file:** {', '.join(file_types)}")
        else:
            st.info("Chưa có lịch sử chat")
        
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 RAG Chatbot - Hệ thống Hỏi Đáp Thông Minh dựa trên Tài liệu</p>
            <p>Sử dụng OpenAI GPT-4o-mini và text-embedding-3-small</p>
            <p>Powered by Haystack Framework với specialized components</p>
            <p>📊 Specialized processing cho từng loại file: PDF, DOCX, XLSX, JSON, CSV, HTML</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
