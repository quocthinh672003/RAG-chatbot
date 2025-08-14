"""
Hybrid Document processing service: Haystack + LangChain fallback
"""

from typing import List, Dict, Any, Union
import os
from datetime import datetime
import uuid

# Fix Pydantic DataFrame conflict - set environment variable
os.environ['PYDANTIC_ARBITRARY_TYPES_ALLOWED'] = 'true'

# Try Haystack for file processing
try:
    from haystack.nodes import PDFToTextConverter, DocxToTextConverter, TextConverter
    from haystack.nodes import PreProcessor, UnstructuredFileConverter
    HAYSTACK_AVAILABLE = True
    print("Haystack loaded successfully with Pydantic fix")
except Exception as e:
    print(f"Haystack import failed: {e}")
    HAYSTACK_AVAILABLE = False

# LangChain fallback
try:
    from langchain.schema import Document as LangChainDocument
    from langchain_community.document_loaders import (
        PyPDFLoader, Docx2txtLoader, TextLoader, 
        UnstructuredExcelLoader, UnstructuredMarkdownLoader,
        CSVLoader, UnstructuredHTMLLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Image extraction
try:
    import fitz  # PyMuPDF for PDF image extraction
    import io
    from PIL import Image
    IMAGE_EXTRACTION_AVAILABLE = True
except ImportError:
    IMAGE_EXTRACTION_AVAILABLE = False
    print("Image extraction not available - install PyMuPDF and Pillow")

from config import config
from utils.helpers import create_metadata


class DocumentProcessor:
    """Hybrid document processor with Haystack + LangChain fallback"""

    def __init__(self):
        # Force LangChain only for now (Haystack has Pydantic issues)
        self.use_haystack = False
        self.use_langchain = True
        
        if LANGCHAIN_AVAILABLE:
            self._init_langchain()
        else:
            raise ImportError("LangChain not available")

    def _init_haystack(self):
        """Initialize Haystack converters"""
        self.pdf_converter = PDFToTextConverter()
        self.docx_converter = DocxToTextConverter()
        self.text_converter = TextConverter()
        self.unstructured_converter = UnstructuredFileConverter()
        
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
        elif ext in ['.xlsx', '.xls', '.md', '.html', '.htm', '.csv']:
            return self.unstructured_converter
        else:
            return self.text_converter

    def _get_langchain_loader(self, file_path: str):
        """Get appropriate LangChain loader"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        elif ext in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        elif ext == '.csv':
            return CSVLoader(file_path)
        elif ext in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path)
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
            print(f"Error processing {file_path}: {e}")
            return []

    def _process_haystack(self, file_path: str) -> List:
        """Process with Haystack"""
        converter = self._get_haystack_converter(file_path)
        documents = converter.convert(file_path)
        
        if not documents:
            print(f"No documents extracted from {file_path}")
            return []
        
        # Add metadata
        metadata = create_metadata(file_path)
        for doc in documents:
            doc.meta.update(metadata)
        
        # Preprocess documents
        processed_docs = self.preprocessor.process(documents)
        
        # Ensure metadata is preserved after preprocessing
        for doc in processed_docs:
            if not hasattr(doc, 'meta') or not doc.meta:
                doc.meta = metadata.copy()
            elif 'source_name' not in doc.meta:
                doc.meta.update(metadata)
        
        print(f"Processed {len(processed_docs)} chunks from {file_path} (Haystack)")
        return processed_docs

    def _process_langchain(self, file_path: str) -> List:
        """Process with LangChain"""
        loader = self._get_langchain_loader(file_path)
        documents = loader.load()
        
        if not documents:
            print(f"No documents extracted from {file_path}")
            return []
        
        # Add metadata
        metadata = create_metadata(file_path)
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Processed {len(split_docs)} chunks from {file_path} (LangChain)")
        return split_docs

    def extract_images_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF file with multiple methods"""
        if not IMAGE_EXTRACTION_AVAILABLE:
            return []
        
        images = []
        try:
            doc = fitz.open(file_path)
            print(f"PDF has {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Method 1: Get embedded images (XObject)
                image_list = page.get_images()
                print(f"Page {page_num + 1}: Found {len(image_list)} embedded images")
                
                # Method 2: Get drawings (might include images)
                drawings = page.get_drawings()
                print(f"Page {page_num + 1}: Found {len(drawings)} drawings")
                
                # Method 3: Get all XObjects (including images)
                xobjects = page.get_xobjects()
                print(f"Page {page_num + 1}: Found {len(xobjects)} XObjects")
                
                # Method 4: Get all annotations (might contain images)
                annotations = page.annots()
                print(f"Page {page_num + 1}: Found {len(annotations) if annotations else 0} annotations")
                
                # Method 5: Get all widgets (form elements, might contain images)
                widgets = page.widgets()
                print(f"Page {page_num + 1}: Found {len(widgets) if widgets else 0} widgets")
                
                # Process embedded images
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Get image rectangle for better context mapping
                        img_rect = None
                        try:
                            # Try to get image rectangle from page
                            img_rect = page.get_image_bbox(img)
                        except:
                            pass
                        
                        # Extract context around image (if possible)
                        context = ""
                        if img_rect:
                            # Get text in a larger area around the image
                            expanded_rect = img_rect * 1.5  # Expand rectangle
                            context = page.get_text("text", clip=expanded_rect)
                        else:
                            # Fallback: get text from the whole page
                            context = page_text[:500]  # First 500 chars as context
                        
                        # Save image to file
                        img_filename = f"extracted_image_page{page_num+1}_img{img_index+1}.png"
                        img_path = os.path.join("uploads", img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Save metadata to separate JSON file
                        metadata_path = img_path.replace('.png', '_metadata.json')
                        metadata_info = {
                            "filename": img_filename,
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "size": len(img_data),
                            "context": context.strip()[:200],
                            "rect": str(img_rect) if img_rect else None
                        }
                        
                        try:
                            import json
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata_info, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            print(f"Error saving metadata for {img_filename}: {e}")
                        
                        images.append({
                            "filename": img_filename,
                            "path": img_path,
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "size": len(img_data),
                            "context": context.strip()[:200],  # Store context for better mapping
                            "rect": str(img_rect) if img_rect else None
                        })
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num}: {e}")
                        continue
                
                # Process XObjects (might contain additional images)
                for xobj_index, xobj in enumerate(xobjects):
                    try:
                        if xobj.get("Subtype") == "Image":
                            print(f"Found image in XObject {xobj_index} on page {page_num + 1}")
                            # Try to extract image from XObject
                            xref = xobj.get("xref")
                            if xref:
                                pix = fitz.Pixmap(doc, xref)
                                if pix.n - pix.alpha < 4:  # GRAY or RGB
                                    img_data = pix.tobytes("png")
                                else:  # CMYK: convert to RGB first
                                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                    img_data = pix1.tobytes("png")
                                    pix1 = None
                                
                                # Save image from XObject
                                img_filename = f"extracted_xobject_page{page_num+1}_xobj{xobj_index+1}.png"
                                img_path = os.path.join("uploads", img_filename)
                                
                                with open(img_path, "wb") as img_file:
                                    img_file.write(img_data)
                                
                                # Save metadata
                                metadata_path = img_path.replace('.png', '_metadata.json')
                                metadata_info = {
                                    "filename": img_filename,
                                    "page": page_num + 1,
                                    "type": "xobject",
                                    "index": xobj_index + 1,
                                    "size": len(img_data),
                                    "context": page_text[:200],
                                    "source": "XObject"
                                }
                                
                                try:
                                    import json
                                    with open(metadata_path, 'w', encoding='utf-8') as f:
                                        json.dump(metadata_info, f, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    print(f"Error saving XObject metadata: {e}")
                                
                                images.append({
                                    "filename": img_filename,
                                    "path": img_path,
                                    "page": page_num + 1,
                                    "type": "xobject",
                                    "index": xobj_index + 1,
                                    "size": len(img_data),
                                    "context": page_text[:200],
                                    "source": "XObject"
                                })
                                
                                pix = None
                    except Exception as e:
                        print(f"Error processing XObject {xobj_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
            # Method 6: Create screenshots for ALL pages (to capture any images)
            print("Creating screenshots for all pages to capture any images...")
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text().lower()
                
                # Create screenshot for every page
                print(f"Creating screenshot for page {page_num + 1}...")
                
                # Create page screenshot with high quality
                mat = fitz.Matrix(3, 3)  # 3x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Save screenshot
                img_filename = f"page_screenshot_page{page_num+1}.png"
                img_path = os.path.join("uploads", img_filename)
                
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                
                # Save metadata
                metadata_path = img_path.replace('.png', '_metadata.json')
                metadata_info = {
                    "filename": img_filename,
                    "page": page_num + 1,
                    "type": "screenshot",
                    "size": len(img_data),
                    "context": page_text[:200],
                    "source": "Page Screenshot (All Pages)"
                }
                
                try:
                    import json
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_info, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving screenshot metadata: {e}")
                
                images.append({
                    "filename": img_filename,
                    "path": img_path,
                    "page": page_num + 1,
                    "type": "screenshot",
                    "size": len(img_data),
                    "context": page_text[:200],
                    "source": "Page Screenshot (All Pages)"
                })
                
                pix = None
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting images from {file_path}: {e}")
        
        return images


class DocumentService:
    """Main document service with hybrid approach"""

    def __init__(self):
        self.processor = DocumentProcessor()

    def convert_file(self, file_path: str) -> List:
        """Convert and process a single file"""
        return self.processor.process(file_path)

    def convert_files(self, file_paths: List[str]) -> List:
        """Convert and process multiple files"""
        all_documents = []
        for file_path in file_paths:
            documents = self.convert_file(file_path)
            all_documents.extend(documents)
        return all_documents

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "active_processor": "Haystack" if self.processor.use_haystack else "LangChain",
            "available_processors": {
                "haystack": HAYSTACK_AVAILABLE,
                "langchain": LANGCHAIN_AVAILABLE
            }
        }

    def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from document"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.processor.extract_images_from_pdf(file_path)
        else:
            return []  # Only PDF images supported for now
