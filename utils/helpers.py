"""
Helper utilities for RAG Chatbot
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def get_file_extension(file_path: str) -> str:
    """Get file extension from path"""
    return os.path.splitext(file_path)[1].lower()


def get_file_name(file_path: str) -> str:
    """Get file name from path"""
    return os.path.basename(file_path)


def is_supported_file_type(file_path: str, supported_types: List[str]) -> bool:
    """Check if file type is supported"""
    ext = get_file_extension(file_path)
    return ext[1:] in supported_types if ext else False


def create_metadata(
    file_path: str, additional_meta: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create metadata for document"""
    metadata = {
        "source_name": get_file_name(file_path),
        "file_type": (
            get_file_extension(file_path)[1:]
            if get_file_extension(file_path)
            else "unknown"
        ),
        "file_path": file_path,
        "processed_at": datetime.now().isoformat(),
    }

    if additional_meta:
        metadata.update(additional_meta)

    return metadata


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def safe_read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Safely read file content"""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""
