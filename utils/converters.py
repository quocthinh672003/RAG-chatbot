"""
Enhanced file converters using Haystack FileTypeRouter and specialized components
"""
from typing import List, Dict, Any
from haystack import Document
from haystack.nodes import (
    # Converters
    PDFMinerToDocument,
    DOCXToDocument, 
    TextFileToDocument,
    MarkdownToDocument,
    XLSXToDocument,
    PPTXToDocument,
    HTMLToDocument,
    JSONConverter,
    CSVToDocument,
    # Router
    FileTypeRouter,
    # PreProcessors
    DocumentCleaner,
    RecursiveDocumentSplitter,
    HierarchicalDocumentSplitter,
    # Specialized processors
    CSVDocumentCleaner,
    CSVDocumentSplitter
)
import os

def get_file_type_router():
    """Create FileTypeRouter for automatic file type detection and routing"""
    return FileTypeRouter(
        supported_types=[
            "pdf", "docx", "txt", "md", "markdown",
            "xlsx", "xls", "pptx", "html", "htm", 
            "json", "csv"
        ]
    )

def get_specialized_processor(file_type: str):
    """Get specialized processor based on file type"""
    
    # Base cleaner for all documents
    base_cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_substrings=None,
        remove_regex_substrings=None,
    )
    
    # Specialized processors for different file types
    processors = {
        # Text-based files - use recursive splitter
        "txt": RecursiveDocumentSplitter(
            split_length=1000,
            split_overlap=100,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        "md": RecursiveDocumentSplitter(
            split_length=800,  # Shorter for markdown
            split_overlap=80,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        "markdown": RecursiveDocumentSplitter(
            split_length=800,
            split_overlap=80,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        
        # Structured documents - use hierarchical splitter
        "pdf": HierarchicalDocumentSplitter(
            split_length=1000,
            split_overlap=100,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        "docx": HierarchicalDocumentSplitter(
            split_length=1000,
            split_overlap=100,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        
        # HTML - use recursive splitter with HTML cleaning
        "html": RecursiveDocumentSplitter(
            split_length=800,
            split_overlap=80,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        "htm": RecursiveDocumentSplitter(
            split_length=800,
            split_overlap=80,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
        
        # JSON - use recursive splitter for structured data
        "json": RecursiveDocumentSplitter(
            split_length=600,  # Shorter for JSON
            split_overlap=60,
            split_respect_sentence_boundary=False,  # JSON doesn't have sentences
            split_respect_word_boundary=True,
            split_by="word"
        ),
        
        # Spreadsheets - use CSV specialized processors
        "csv": CSVDocumentSplitter(
            split_length=500,  # Shorter for CSV
            split_overlap=50
        ),
        "xlsx": CSVDocumentSplitter(
            split_length=500,
            split_overlap=50
        ),
        "xls": CSVDocumentSplitter(
            split_length=500,
            split_overlap=50
        ),
        
        # Presentations - use hierarchical splitter
        "pptx": HierarchicalDocumentSplitter(
            split_length=800,
            split_overlap=80,
            split_respect_sentence_boundary=True,
            split_respect_word_boundary=True,
            split_by="word"
        ),
    }
    
    return processors.get(file_type, RecursiveDocumentSplitter(
        split_length=1000,
        split_overlap=100,
        split_respect_sentence_boundary=True,
        split_respect_word_boundary=True,
        split_by="word"
    ))

def get_converter_for_file(file_path: str):
    """Get appropriate Haystack converter based on file extension"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    converters = {
        '.pdf': PDFMinerToDocument(),
        '.docx': DOCXToDocument(),
        '.txt': TextFileToDocument(),
        '.md': MarkdownToDocument(),
        '.markdown': MarkdownToDocument(),
        '.xlsx': XLSXToDocument(),
        '.xls': XLSXToDocument(),
        '.pptx': PPTXToDocument(),
        '.html': HTMLToDocument(),
        '.htm': HTMLToDocument(),
        '.json': JSONConverter(),
        '.csv': CSVToDocument(),
    }
    
    return converters.get(file_ext, TextFileToDocument())

def convert_file_to_documents(file_path: str) -> List[Document]:
    """Convert file to Haystack Documents using specialized processing pipeline"""
    
    # Get file type
    file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # 1. Get the right converter
    converter = get_converter_for_file(file_path)
    
    # 2. Convert file to documents
    documents = converter.convert(file_paths=[file_path])
    
    # 3. Add metadata to each document
    for doc in documents:
        doc.meta.update({
            "source_name": filename,
            "source_path": file_path,
            "file_size": file_size,
            "file_type": file_ext,
            "original_file_type": file_ext,
        })
    
    # 4. Get specialized processor for this file type
    processor = get_specialized_processor(file_ext)
    
    # 5. Process documents with specialized processor
    processed_docs = processor.process(documents)
    
    # 6. Add processing metadata
    for doc in processed_docs:
        doc.meta.update({
            "processor_type": type(processor).__name__,
            "chunk_size": getattr(processor, 'split_length', 'unknown'),
            "chunk_overlap": getattr(processor, 'split_overlap', 'unknown'),
        })
    
    return processed_docs
