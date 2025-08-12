#!/usr/bin/env python3
"""
Script để chạy RAG Chatbot
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def check_docker():
    """Kiểm tra Docker có sẵn không"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_qdrant():
    """Khởi động Qdrant bằng Docker"""
    print("🐳 Khởi động Qdrant...")
    
    if not check_docker():
        print("❌ Docker không được cài đặt. Vui lòng cài đặt Docker trước.")
        return False
    
    try:
        # Kiểm tra container đã tồn tại chưa
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=rag-qdrant", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        
        if "rag-qdrant" in result.stdout:
            # Container đã tồn tại, start nó
            subprocess.run(["docker", "start", "rag-qdrant"], check=True)
            print("✅ Qdrant đã được khởi động")
        else:
            # Tạo container mới
            subprocess.run([
                "docker", "run", "-d",
                "--name", "rag-qdrant",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant:latest"
            ], check=True)
            print("✅ Qdrant đã được tạo và khởi động")
        
        # Đợi Qdrant khởi động
        print("⏳ Đợi Qdrant khởi động...")
        time.sleep(5)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khởi động Qdrant: {e}")
        return False

def check_qdrant():
    """Kiểm tra Qdrant có hoạt động không"""
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        return response.status_code == 200
    except:
        return False

def install_dependencies():
    """Cài đặt dependencies"""
    print("📦 Cài đặt dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirement.txt"], check=True)
        print("✅ Dependencies đã được cài đặt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt dependencies: {e}")
        return False

def run_streamlit():
    """Chạy ứng dụng Streamlit"""
    print("🚀 Khởi động ứng dụng Streamlit...")
    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi chạy Streamlit: {e}")

def main():
    print("🤖 RAG Chatbot - Khởi động hệ thống")
    print("=" * 50)
    
    # Kiểm tra file requirements
    if not Path("requirement.txt").exists():
        print("❌ Không tìm thấy file requirement.txt")
        return
    
    # Cài đặt dependencies
    if not install_dependencies():
        return
    
    # Khởi động Qdrant
    if not start_qdrant():
        print("❌ Không thể khởi động Qdrant")
        return
    
    # Kiểm tra Qdrant
    if not check_qdrant():
        print("❌ Qdrant không phản hồi")
        return
    
    print("✅ Hệ thống đã sẵn sàng!")
    print("🌐 Truy cập: http://localhost:8501")
    print("📚 API docs: http://localhost:8000/docs")
    print("=" * 50)
    
    # Chạy Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()
