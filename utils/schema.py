from datetime import datetime
import uuid
import os

def create_document_data( source_path, permission_group, language = 'en', title = None, author = None, keywords = None) -> dict:
    """
    Create a document data dictionary with metadata.
    """
    filename = os.path.basename(source_path)
    file_size = None
    
    try:
        file_size = os.path.getsize(source_path)
    except Exception:
        file_size = None

    return {
        "document_metadata": {
            "document_id": str(uuid.uuid4()),
            "source_name": filename,
            "source_path": source_path,
            "ingestion_timestamp": datetime.now().isoformat(),
            "file_size": file_size,
            "file_type": os.path.splitext(filename)[-1].lower(),
            "permission_group": permission_group or [],
            "language": language,
            "title": title or filename,
            "author": author,
            "keywords": keywords or [],
            "page_count": None,
        }
    }

def make_element_data( element_id, element_type, content, page_number, image_path = None, language = "vi") -> dict:
    """
    Create an element data dictionary with metadata.
    """
    return {
        "element_id": element_id,
        "element_type": element_type,
        "content": content,
        "page_number": page_number,
        "image_path": image_path,
        "language": language,
        "coordinates": None,
        "parent_element_id": None,
        "font_size": None,
        "font_weight": None,
    }