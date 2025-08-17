"""
Document Service - Enhanced with Standard JSON Structure

Mục đích:
- Xử lý documents theo cấu trúc JSON chuẩn
- Implement FR1.1-FR1.4 requirements
- Tạo ra JSON structure theo yêu cầu dự án
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import document processing libraries
try:
    import pypdf
    from docx import Document
    import pandas as pd
    import markdown
    from bs4 import BeautifulSoup
    import re
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Some document processing libraries not available: {e}")
    LIBRARIES_AVAILABLE = False
    # Create dummy imports to prevent errors
    pypdf = None
    Document = None
    pd = None
    markdown = None
    BeautifulSoup = None
    import re  # re is built-in, should always be available

# Try to import pandas separately
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from config import config
from utils.helpers import create_metadata


class DocumentProcessor:
    """Hybrid document processor with Haystack + LangChain fallback"""

    def __init__(self):
        self.use_haystack = HAYSTACK_AVAILABLE
        self.use_langchain = not HAYSTACK_AVAILABLE and LANGCHAIN_AVAILABLE
        
        if self.use_haystack:
            self._init_haystack()
        elif self.use_langchain:
            self._init_langchain()
        else:
            raise ImportError("Neither Haystack nor LangChain available")

    def _init_haystack(self):
        """Initialize Haystack converters"""
        self.pdf_converter = PDFToTextConverter()
        self.docx_converter = DocxToTextConverter()
        self.text_converter = TextConverter()
        
        self.preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="sentence",
            split_length=config.processing.chunk_size,
            split_overlap=config.processing.chunk_overlap,
        )

    def _init_langchain(self):
        """Initialize LangChain components"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap,
            length_function=len
        )

    def _get_haystack_converter(self, file_path: str):
        """Get appropriate Haystack converter"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.pdf_converter
        elif ext == '.docx':
            return self.docx_converter
        else:
            return self.text_converter

    def _get_langchain_loader(self, file_path: str):
        """Get appropriate LangChain loader"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        else:
            return TextLoader(file_path)

    def process(self, file_path: str) -> List:
        """Process document from file path"""
        try:
            if self.use_haystack:
                return self._process_haystack(file_path)
            elif self.use_langchain:
                return self._process_langchain(file_path)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    def _process_haystack(self, file_path: str) -> List:
        """Process with Haystack"""
        converter = self._get_haystack_converter(file_path)
        documents = converter.convert(file_path)
        
        if not documents:
            print(f"⚠️ No documents extracted from {file_path}")
            return []
        
        # Add metadata
        metadata = create_metadata(file_path)
        for doc in documents:
            doc.meta.update(metadata)
        
        # Preprocess documents
        processed_docs = self.preprocessor.process(documents)
        print(f"✅ Processed {len(processed_docs)} chunks from {file_path} (Haystack)")
        return processed_docs

    def _process_langchain(self, file_path: str) -> List:
        """Process with LangChain"""
        print(f"🔍 Processing {file_path} with LangChain...")
        
        loader = self._get_langchain_loader(file_path)
        documents = loader.load()
        
        print(f"📄 Loaded {len(documents)} documents from {file_path}")
        
        if not documents:
            print(f"⚠️ No documents extracted from {file_path}")
            return []
        
        # Add metadata
        metadata = create_metadata(file_path)
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✅ Processed {len(split_docs)} chunks from {file_path} (LangChain)")
        
        # Debug: show first chunk content
        if split_docs:
            first_chunk = split_docs[0].page_content[:200] + "..." if len(split_docs[0].page_content) > 200 else split_docs[0].page_content
            print(f"📝 First chunk preview: {first_chunk}")
        
        return split_docs


class DocumentService:
    """
    Document Service với cấu trúc JSON chuẩn
    
    Chức năng:
    1. Convert files thành JSON structure chuẩn
    2. Text splitting/chunking
    3. Element classification (Title, NarrativeText, Table, Image)
    4. Metadata extraction
    """

    def __init__(self):
        self.supported_formats = config.processing.supported_formats
        self.chunk_size = config.processing.chunk_size
        self.chunk_overlap = config.processing.chunk_overlap

    def convert_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Convert file thành cấu trúc JSON chuẩn
        
        Returns:
            List[Dict]: Danh sách documents với cấu trúc JSON chuẩn
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            
            # Generate document ID
            document_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Create document metadata
            document_metadata = {
                "document_id": document_id,
                "source_filename": filename,
                "source_path": file_path,
                "ingestion_timestamp": datetime.now().isoformat() + "Z",
                "permission_groups": ["default_access"],
                "file_type": file_extension,
                "file_size": os.path.getsize(file_path)
            }
            
            # Extract content based on file type
            if file_extension == '.pdf':
                elements = self._process_pdf(file_path, document_id)
            elif file_extension == '.docx':
                elements = self._process_docx(file_path, document_id)
            elif file_extension in ['.txt', '.md']:
                elements = self._process_text(file_path, document_id)
            elif file_extension in ['.xlsx', '.xls']:
                elements = self._process_excel(file_path, document_id)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create standard JSON structure
            document_structure = {
                "document_metadata": document_metadata,
                "elements": elements
            }
            
            # Save complete JSON structure to file
            self._save_json_structure(document_structure, document_id)
            
            # Convert to LangChain documents
            documents = self._convert_to_langchain_documents(document_structure)
            
            logger.info(f"✅ Converted {filename} to {len(elements)} elements")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error converting file {file_path}: {e}")
            raise e

    def _save_json_structure(self, document_structure: Dict[str, Any], document_id: str) -> None:
        """Save complete JSON structure to file for persistence"""
        # Completely disabled JSON storage to prevent spam
        pass

    def get_json_structure(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete JSON structure from file"""
        try:
            json_file_path = os.path.join("json_storage", f"{document_id}_structure.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"❌ Error loading JSON structure: {e}")
            return None

    def _process_pdf(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process PDF file and extract elements"""
        elements = []
        element_counter = 0
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():
                        # Split text into chunks
                        chunks = self._split_text(text)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            element_counter += 1
                            element_id = f"elem_{document_id}_{element_counter:04d}"
                            
                            # Classify element type
                            element_type = self._classify_element(chunk)
                            
                            element = {
                                "element_id": element_id,
                                "type": element_type,
                                "content": chunk,
                                "metadata": {
                                    "page_number": page_num + 1,
                                    "chunk_index": chunk_idx,
                                    "language": "vi",
                                    "coordinates": {
                                        "page": page_num + 1,
                                        "system": "PDF"
                                    },
                                    "parent_id": document_id
                                }
                            }
                            elements.append(element)
                            
        except Exception as e:
            logger.error(f"❌ Error processing PDF {file_path}: {e}")
            
        return elements

    def _process_docx(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process DOCX file and extract elements"""
        elements = []
        element_counter = 0
        
        try:
            doc = Document(file_path)
            
            for para_idx, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                
                if text:
                    element_counter += 1
                    element_id = f"elem_{document_id}_{element_counter:04d}"
                    
                    # Classify element type based on paragraph style
                    element_type = self._classify_docx_element(paragraph)
                    
                    element = {
                        "element_id": element_id,
                        "type": element_type,
                        "content": text,
                        "metadata": {
                            "paragraph_index": para_idx,
                            "language": "vi",
                            "parent_id": document_id,
                            "style": paragraph.style.name if paragraph.style else "Normal"
                        }
                    }
                    elements.append(element)
                    
        except Exception as e:
            logger.error(f"❌ Error processing DOCX {file_path}: {e}")
            
        return elements

    def _process_text(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process text files (TXT, MD) and extract elements"""
        elements = []
        element_counter = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split content into chunks
                chunks = self._split_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    element_counter += 1
                    element_id = f"elem_{document_id}_{element_counter:04d}"
                    
                    # Classify element type
                    element_type = self._classify_element(chunk)
                    
                    element = {
                        "element_id": element_id,
                        "type": element_type,
                        "content": chunk,
                        "metadata": {
                            "chunk_index": chunk_idx,
                            "language": "vi",
                            "parent_id": document_id
                        }
                    }
                    elements.append(element)
                    
        except Exception as e:
            logger.error(f"❌ Error processing text file {file_path}: {e}")
            
        return elements

    def _process_excel(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process Excel files and extract tables"""
        elements = []
        element_counter = 0
        
        if not PANDAS_AVAILABLE:
            logger.error("❌ Pandas not available for Excel processing")
            return elements
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if not df.empty:
                    element_counter += 1
                    element_id = f"elem_{document_id}_{element_counter:04d}"
                    
                    # Convert DataFrame to Markdown table
                    table_content = df.to_markdown(index=False)
                    
                    # Add sheet title
                    full_content = f"## Bảng: {sheet_name}\n\n{table_content}"
                    
                    element = {
                        "element_id": element_id,
                        "type": "Table",
                        "content": full_content,
                        "metadata": {
                            "sheet_name": sheet_name,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "language": "vi",
                            "parent_id": document_id,
                            "table_title": f"Table from {sheet_name}",
                            "column_names": df.columns.tolist(),
                            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                        }
                    }
                    elements.append(element)
                    
                    # Also create a summary element
                    element_counter += 1
                    summary_element_id = f"elem_{document_id}_{element_counter:04d}"
                    
                    summary_content = f"Bảng '{sheet_name}' chứa {len(df)} dòng và {len(df.columns)} cột. Các cột bao gồm: {', '.join(df.columns.tolist())}."
                    
                    summary_element = {
                        "element_id": summary_element_id,
                        "type": "NarrativeText",
                        "content": summary_content,
                        "metadata": {
                            "sheet_name": sheet_name,
                            "parent_id": document_id,
                            "language": "vi",
                            "is_summary": True
                        }
                    }
                    elements.append(summary_element)
                    
        except Exception as e:
            logger.error(f"❌ Error processing Excel file {file_path}: {e}")
            
        return elements

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk = text[start:]
            else:
                # Try to break at sentence boundary
                chunk = text[start:end]
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size * 0.5:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
            
        return chunks

    def _classify_element(self, text: str) -> str:
        """Classify element type based on content"""
        text = text.strip()
        
        # Check for titles (short text, ends with colon, or all caps)
        if len(text) < 100 and (text.endswith(':') or text.isupper()):
            return "Title"
        
        # Check for tables (contains | or tabular structure)
        if '|' in text or '\t' in text:
            return "Table"
        
        # Check for lists
        if re.match(r'^[\s]*[-*•]\s', text, re.MULTILINE):
            return "ListItem"
        
        # Default to narrative text
        return "NarrativeText"

    def _classify_docx_element(self, paragraph) -> str:
        """Classify DOCX element based on paragraph style"""
        text = paragraph.text.strip()
        
        # Check paragraph style
        if paragraph.style:
            style_name = paragraph.style.name.lower()
            if 'heading' in style_name or 'title' in style_name:
                return "Title"
        
        # Check content patterns
        if len(text) < 100 and (text.endswith(':') or text.isupper()):
            return "Title"
        
        if re.match(r'^[\s]*[-*•]\s', text):
            return "ListItem"
        
        return "NarrativeText"

    def _convert_to_langchain_documents(self, document_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert standard JSON structure to LangChain documents"""
        documents = []
        
        for element in document_structure["elements"]:
            # Create LangChain document
            doc = {
                "page_content": element["content"],
                "metadata": {
                    "source_name": document_structure["document_metadata"]["source_filename"],
                    "document_id": document_structure["document_metadata"]["document_id"],
                    "element_id": element["element_id"],
                    "element_type": element["type"],
                    "page_number": element["metadata"].get("page_number", 1),
                    "language": element["metadata"].get("language", "vi"),
                    **element["metadata"]
                }
            }
            documents.append(doc)
        
        return documents
