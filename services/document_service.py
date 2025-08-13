"""
Hybrid Document processing service: Haystack + LangChain fallback
"""

from typing import List, Dict, Any, Union
import os
from datetime import datetime
import uuid

# Try Haystack first
try:
    from haystack import Document as HaystackDocument
    from haystack.nodes import PDFToTextConverter, DocxToTextConverter, TextConverter
    from haystack.nodes import PreProcessor
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

# LangChain fallback
try:
    from langchain.schema import Document as LangChainDocument
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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
        loader = self._get_langchain_loader(file_path)
        documents = loader.load()
        
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
        return split_docs


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
