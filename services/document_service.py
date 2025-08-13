"""
Document processing service
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from haystack import Document
from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    DOCXToDocument,
    MarkdownToDocument,
    CSVToDocument,
)
from haystack.components.preprocessors import DocumentSplitter
import os
from config import config
from core.constants import SUPPORTED_FILE_TYPES, TEXT_SPLITTER_SEPARATORS
from utils.helpers import create_metadata, safe_read_file, is_supported_file_type


class DocumentConverter(ABC):
    """Abstract document converter"""

    @abstractmethod
    def convert(self, file_path: str) -> List[Document]:
        pass

    @abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        pass


class HaystackTextConverter(DocumentConverter):
    """Text file converter using Haystack"""

    def __init__(self):
        self.converter = TextFileToDocument()

    def convert(self, file_path: str) -> List[Document]:
        result = self.converter.run(paths=[file_path])
        return result["documents"]

    def supports_file_type(self, file_path: str) -> bool:
        return is_supported_file_type(file_path, ["txt"])


class HaystackPDFConverter(DocumentConverter):
    """PDF file converter using Haystack"""

    def __init__(self):
        self.converter = PyPDFToDocument()

    def convert(self, file_path: str) -> List[Document]:
        result = self.converter.run(paths=[file_path])
        return result["documents"]

    def supports_file_type(self, file_path: str) -> bool:
        return is_supported_file_type(file_path, ["pdf"])


class HaystackDocxConverter(DocumentConverter):
    """DOCX file converter using Haystack"""

    def __init__(self):
        self.converter = DOCXToDocument()

    def convert(self, file_path: str) -> List[Document]:
        result = self.converter.run(paths=[file_path])
        return result["documents"]

    def supports_file_type(self, file_path: str) -> bool:
        return is_supported_file_type(file_path, ["docx"])


class HaystackMarkdownConverter(DocumentConverter):
    """Markdown file converter using Haystack"""

    def __init__(self):
        self.converter = MarkdownToDocument()

    def convert(self, file_path: str) -> List[Document]:
        result = self.converter.run(paths=[file_path])
        return result["documents"]

    def supports_file_type(self, file_path: str) -> bool:
        return is_supported_file_type(file_path, ["md", "markdown"])


class HaystackCsvConverter(DocumentConverter):
    """CSV file converter using Haystack"""

    def __init__(self):
        self.converter = CSVToDocument()

    def convert(self, file_path: str) -> List[Document]:
        result = self.converter.run(paths=[file_path])
        return result["documents"]

    def supports_file_type(self, file_path: str) -> bool:
        return is_supported_file_type(file_path, ["csv"])


class DocumentProcessor:
    """Document processor"""

    def __init__(self):
        self.splitter = DocumentSplitter(
            split_by="sentence",
            split_length=config.processing.chunk_size,
            split_overlap=config.processing.chunk_overlap,
        )

    def process(self, documents: List[Document], file_path: str) -> List[Document]:
        # Add metadata
        metadata = create_metadata(file_path)

        for doc in documents:
            doc.meta.update(metadata)

        # Process documents using Haystack DocumentSplitter
        processed_docs = self.splitter.run(documents)
        return processed_docs["documents"]


class DocumentService:
    """Main document service"""

    def __init__(self):
        self.converters = [
            HaystackTextConverter(),
            HaystackPDFConverter(),
            HaystackDocxConverter(),
            HaystackMarkdownConverter(),
            HaystackCsvConverter(),
        ]
        self.processor = DocumentProcessor()

    def convert_file(self, file_path: str) -> List[Document]:
        # Find appropriate converter
        converter = None
        for conv in self.converters:
            if conv.supports_file_type(file_path):
                converter = conv
                break

        if not converter:
            converter = HaystackTextConverter()  # Default fallback

        # Convert and process
        raw_docs = converter.convert(file_path)
        processed_docs = self.processor.process(raw_docs, file_path)

        return processed_docs
