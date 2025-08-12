"""
use unstructured.partition để parse các file 
return về document_data và element_data
"""
import os
from unstructured.partition.auto import partition
from utils.schema import create_document_data, make_element_data
from typing import List, Dict, Any

def parse_file_to_elements(file_path: str, permission_group: List[str] = None, language: str = 'en') -> Dict[str, Any]:
    """
    Parse a file into document and element data.
    """

    elements = partition( file_name = file_path)

    document_data = create_document_data(
        source_path=file_path,
        permission_group=permission_group,
        language=language
    )
    
    output_data = []

    for index, element in enumerate(elements):
        # get element_type
        element_type = getattr( element, 'element_type', type(element).__name__.lower())
        # get content
        content = (getattr(element, 'text', str(element)) or getattr(element, 'value', None) or "" ).strip()
        # get image_path if available
        image_path = getattr(element, 'image_path', None)
        
        # create element data
        element_data = make_element_data(
            element_id = f"element_{index}_{os.path.basename(file_path)}",
            element_type = element_type,
            content = content,
            page_number = getattr(element, 'page_number', None),
            image_path = image_path,
            language = language
        )
        output_data.append(element_data)

    return {
        "document_data": document_data,
        "elements": output_data
    }