#!/usr/bin/env python3
"""
Test script cho RAG Chatbot
"""
import os
import tempfile
import requests
import json
from pathlib import Path

def test_qdrant_connection():
    """Test kết nối Qdrant"""
    print("🔍 Kiểm tra kết nối Qdrant...")
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant đang hoạt động")
            return True
        else:
            print("❌ Qdrant không phản hồi đúng")
            return False
    except Exception as e:
        print(f"❌ Không thể kết nối Qdrant: {e}")
        return False

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Kiểm tra API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API đang hoạt động: {data}")
            return True
        else:
            print("❌ API không phản hồi đúng")
            return False
    except Exception as e:
        print(f"❌ Không thể kết nối API: {e}")
        return False

def test_upload_document():
    """Test upload tài liệu"""
    print("🔍 Test upload tài liệu...")
    
    # Tạo file test
    test_content = """
    Đây là tài liệu test cho RAG Chatbot.
    
    Nội dung chính:
    - Hệ thống RAG sử dụng OpenAI models
    - Embedding: text-embedding-3-small
    - LLM: gpt-4o-mini
    - Vector database: Qdrant
    
    Các tính năng:
    1. Upload và xử lý tài liệu
    2. Tìm kiếm tương đồng
    3. Tạo câu trả lời thông minh
    4. Trích dẫn nguồn
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            data = {'permission_groups': 'public'}
            
            response = requests.post(
                "http://localhost:8000/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload thành công: {result}")
            return result.get('document_id')
        else:
            print(f"❌ Upload thất bại: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi upload: {e}")
        return None
    finally:
        os.unlink(temp_file_path)

def test_query_document(doc_id):
    """Test truy vấn tài liệu"""
    print("🔍 Test truy vấn tài liệu...")
    
    query_data = {
        "question": "Hệ thống RAG sử dụng model nào?",
        "permission_groups": ["public"],
        "top_k": 5
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Truy vấn thành công!")
            print(f"🤖 Câu trả lời: {result['answer']}")
            print(f"📚 Số nguồn: {len(result['sources'])}")
            return True
        else:
            print(f"❌ Truy vấn thất bại: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi truy vấn: {e}")
        return False

def test_list_documents():
    """Test lấy danh sách tài liệu"""
    print("🔍 Test lấy danh sách tài liệu...")
    
    try:
        response = requests.get("http://localhost:8000/documents", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Lấy danh sách thành công: {result['total_count']} tài liệu")
            for doc in result['documents']:
                print(f"  - {doc['source_name']} ({doc['document_id'][:8]}...)")
            return True
        else:
            print(f"❌ Lấy danh sách thất bại: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi lấy danh sách: {e}")
        return False

def main():
    print("🧪 RAG Chatbot - Test Hệ thống")
    print("=" * 50)
    
    # Test 1: Kết nối Qdrant
    if not test_qdrant_connection():
        print("❌ Qdrant không hoạt động. Vui lòng khởi động Qdrant trước.")
        return
    
    # Test 2: API health
    if not test_api_health():
        print("❌ API không hoạt động. Vui lòng khởi động API server trước.")
        return
    
    # Test 3: Upload tài liệu
    doc_id = test_upload_document()
    if not doc_id:
        print("❌ Upload tài liệu thất bại")
        return
    
    # Test 4: Truy vấn tài liệu
    if not test_query_document(doc_id):
        print("❌ Truy vấn tài liệu thất bại")
        return
    
    # Test 5: Lấy danh sách tài liệu
    if not test_list_documents():
        print("❌ Lấy danh sách tài liệu thất bại")
        return
    
    print("=" * 50)
    print("🎉 Tất cả test đều thành công!")
    print("✅ Hệ thống RAG Chatbot hoạt động bình thường")

if __name__ == "__main__":
    main()
