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

# Check Haystack availability
try:
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.nodes import PreProcessor
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

# Check LangChain availability  
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from config import config
from utils.helpers import create_metadata





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
                "file_size": os.path.getsize(file_path),
            }

            # Extract content based on file type
            if file_extension == ".pdf":
                elements = self._process_pdf(file_path, document_id)
            elif file_extension == ".docx":
                elements = self._process_docx(file_path, document_id)
            elif file_extension in [".txt", ".md"]:
                elements = self._process_text(file_path, document_id)
            elif file_extension in [".xlsx", ".xls"]:
                elements = self._process_excel(file_path, document_id)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Create standard JSON structure
            document_structure = {
                "document_metadata": document_metadata,
                "elements": elements,
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

    def _save_json_structure(
        self, document_structure: Dict[str, Any], document_id: str
    ) -> None:
        """Save complete JSON structure to file for persistence"""
        # Completely disabled JSON storage to prevent spam
        pass

    def get_json_structure(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete JSON structure from file"""
        try:
            json_file_path = os.path.join(
                "json_storage", f"{document_id}_structure.json"
            )
            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
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
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()

                    if text.strip():
                        # Try to extract table blocks from raw page text
                        table_blocks = self._extract_markdown_tables(text)
                        for tbl_idx, table_md in enumerate(table_blocks):
                            element_counter += 1
                            table_element_id = (
                                f"elem_{document_id}_{element_counter:04d}"
                            )
                            elements.append(
                                {
                                    "element_id": table_element_id,
                                    "type": "Table",
                                    "content": table_md,
                                    "metadata": {
                                        "page_number": page_num + 1,
                                        "language": "vi",
                                        "parent_id": document_id,
                                        "table_index": tbl_idx,
                                    },
                                }
                            )

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
                                        "system": "PDF",
                                    },
                                    "parent_id": document_id,
                                },
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
                            "style": paragraph.style.name
                            if paragraph.style
                            else "Normal",
                        },
                    }
                    elements.append(element)

            # Additionally extract DOCX tables as Markdown
            try:
                tables = getattr(doc, "tables", []) or []
                for tbl_idx, table in enumerate(tables):
                    try:
                        rows = table.rows
                        if not rows:
                            continue
                        header_cells = [c.text.strip() for c in rows[0].cells]
                        headers = (
                            header_cells
                            if any(header_cells)
                            else [f"Cột {i + 1}" for i in range(len(rows[0].cells))]
                        )
                        md_lines = [
                            "| " + " | ".join(headers) + " |",
                            "|" + "|".join(["---" for _ in headers]) + "|",
                        ]
                        for r in rows[1:]:
                            values = [
                                c.text.strip().replace("\n", " ") for c in r.cells
                            ]
                            if len(values) < len(headers):
                                values += [""] * (len(headers) - len(values))
                            md_lines.append(
                                "| " + " | ".join(values[: len(headers)]) + " |"
                            )
                        table_md = "\n".join(md_lines)

                        element_counter += 1
                        table_element_id = f"elem_{document_id}_{element_counter:04d}"
                        elements.append(
                            {
                                "element_id": table_element_id,
                                "type": "Table",
                                "content": table_md,
                                "metadata": {
                                    "language": "vi",
                                    "parent_id": document_id,
                                    "table_index": tbl_idx,
                                },
                            }
                        )
                    except Exception as te:
                        logger.warning(f"⚠️ Error extracting DOCX table: {te}")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"❌ Error processing DOCX {file_path}: {e}")

        return elements

    def _process_text(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process text files (TXT, MD) and extract elements"""
        elements = []
        element_counter = 0

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # Extract table blocks from text files as Markdown tables
                table_blocks = self._extract_markdown_tables(content)
                for tbl_idx, table_md in enumerate(table_blocks):
                    element_counter += 1
                    table_element_id = f"elem_{document_id}_{element_counter:04d}"
                    elements.append(
                        {
                            "element_id": table_element_id,
                            "type": "Table",
                            "content": table_md,
                            "metadata": {
                                "language": "vi",
                                "parent_id": document_id,
                                "table_index": tbl_idx,
                            },
                        }
                    )

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
                            "parent_id": document_id,
                        },
                    }
                    elements.append(element)

        except Exception as e:
            logger.error(f"❌ Error processing text file {file_path}: {e}")

        return elements

    def _extract_markdown_tables(self, text: str) -> List[str]:
        """Detect simple table-like blocks and convert to Markdown tables.
        Heuristics:
        - Block starts after a line beginning with 'Bảng' (case-insensitive) or contains at least one line with >1 column separated by tabs or 2+ spaces.
        - Block ends at the first empty line.
        """
        lines = text.splitlines()
        i = 0
        markdown_tables: List[str] = []
        while i < len(lines):
            line = lines[i].strip()
            trigger = line.lower().startswith("bảng")
            if trigger:
                # Collect until blank line
                block: List[str] = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    block.append(lines[i])
                    i += 1
                md = self._lines_to_markdown_table(block)
                if md:
                    markdown_tables.append(md)
            i += 1
        return markdown_tables

    def _lines_to_markdown_table(self, block_lines: List[str]) -> Optional[str]:
        """Convert a list of lines into a Markdown table using whitespace/tab splitting."""
        if not block_lines:
            return None
        import re as _re

        def split_cols(s: str) -> List[str]:
            parts = [p.strip() for p in _re.split(r"\t+|\s{2,}", s) if p.strip()]
            return parts

        rows = [split_cols(l) for l in block_lines if split_cols(l)]
        if not rows or len(rows[0]) < 2:
            return None
        headers = rows[0]
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---" for _ in headers]) + "|",
        ]
        for r in rows[1:]:
            # Normalize to header length
            if len(r) < len(headers):
                r = r + [""] * (len(headers) - len(r))
            md_lines.append("| " + " | ".join(r[: len(headers)]) + " |")
        return "\n".join(md_lines)

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
                            "data_types": {
                                col: str(dtype) for col, dtype in df.dtypes.items()
                            },
                        },
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
                            "is_summary": True,
                        },
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
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > start + self.chunk_size * 0.5:
                    chunk = text[start : start + break_point + 1]
                    end = start + break_point + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - self.chunk_overlap

        return chunks

    def _classify_element(self, text: str) -> str:
        """Classify element type based on content"""
        text = text.strip()

        # Check for titles (short text, ends with colon, or all caps)
        if len(text) < 100 and (text.endswith(":") or text.isupper()):
            return "Title"

        # Check for tables (contains | or tabular structure)
        if "|" in text or "\t" in text:
            return "Table"

        # Check for lists
        if re.match(r"^[\s]*[-*•]\s", text, re.MULTILINE):
            return "ListItem"

        # Default to narrative text
        return "NarrativeText"

    def _classify_docx_element(self, paragraph) -> str:
        """Classify DOCX element based on paragraph style"""
        text = paragraph.text.strip()

        # Check paragraph style
        if paragraph.style:
            style_name = paragraph.style.name.lower()
            if "heading" in style_name or "title" in style_name:
                return "Title"

        # Check content patterns
        if len(text) < 100 and (text.endswith(":") or text.isupper()):
            return "Title"

        if re.match(r"^[\s]*[-*•]\s", text):
            return "ListItem"

        return "NarrativeText"

    def _convert_to_langchain_documents(
        self, document_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert standard JSON structure to LangChain documents"""
        documents = []

        for element in document_structure["elements"]:
            # Create LangChain document
            doc = {
                "page_content": element["content"],
                "metadata": {
                    "source_name": document_structure["document_metadata"][
                        "source_filename"
                    ],
                    "document_id": document_structure["document_metadata"][
                        "document_id"
                    ],
                    "element_id": element["element_id"],
                    "element_type": element["type"],
                    "page_number": element["metadata"].get("page_number", 1),
                    "language": element["metadata"].get("language", "vi"),
                    **element["metadata"],
                },
            }
            documents.append(doc)

        return documents
