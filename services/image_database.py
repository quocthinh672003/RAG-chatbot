"""
Image Database Service - Extract REAL images from ALL uploaded files
Stores actual images extracted from any document type
"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path

class ImageDatabase:
    """Image database service that extracts REAL images from ANY uploaded files"""
    
    def __init__(self):
        self.images_dir = "image_database"
        self.metadata_file = os.path.join(self.images_dir, "image_metadata.json")
        self._ensure_database_exists()
        self._load_metadata()
    
    def _ensure_database_exists(self):
        """Create image database directory and structure"""
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Create subdirectories for different categories
        categories = ["extracted", "screenshots", "documents", "general"]
        for category in categories:
            category_dir = os.path.join(self.images_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
    
    def _load_metadata(self):
        """Load or create image metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def extract_images_from_any_file(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract REAL images from ANY file type using multiple methods"""
        extracted_images = []
        
        # Method 1: Try Unstructured library (most comprehensive)
        unstructured_images = self._extract_with_unstructured(file_path, source_filename)
        if unstructured_images:
            extracted_images.extend(unstructured_images)
            print(f"Unstructured extracted {len(unstructured_images)} images")
        
        # Method 2: Try specific file type extractors
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            pdf_images = self._extract_from_pdf(file_path, source_filename)
            extracted_images.extend(pdf_images)
            print(f"PDF extractor found {len(pdf_images)} images")
        
        elif file_ext in ['.docx', '.doc']:
            docx_images = self._extract_from_docx(file_path, source_filename)
            extracted_images.extend(docx_images)
            print(f"DOCX extractor found {len(docx_images)} images")
        
        elif file_ext in ['.xlsx', '.xls']:
            excel_images = self._extract_from_excel(file_path, source_filename)
            extracted_images.extend(excel_images)
            print(f"Excel extractor found {len(excel_images)} images")
        
        elif file_ext in ['.pptx', '.ppt']:
            ppt_images = self._extract_from_powerpoint(file_path, source_filename)
            extracted_images.extend(ppt_images)
            print(f"PowerPoint extractor found {len(ppt_images)} images")
        
        elif file_ext in ['.html', '.htm']:
            html_images = self._extract_from_html(file_path, source_filename)
            extracted_images.extend(html_images)
            print(f"HTML extractor found {len(html_images)} images")
        
        # Method 3: Create document screenshot as fallback
        if not extracted_images:
            screenshot = self._create_document_screenshot(file_path, source_filename)
            if screenshot:
                extracted_images.append(screenshot)
                print("Created document screenshot as fallback")
        
        # Save all extracted images to database
        for img in extracted_images:
            self._save_image_to_database(img)
        
        self._save_metadata()
        print(f"Total extracted {len(extracted_images)} images from {source_filename}")
        return extracted_images
    
    def _extract_with_unstructured(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images using Unstructured library (most comprehensive)"""
        try:
            from unstructured.partition.auto import partition
            from unstructured.documents.elements import Image
            
            # Extract all elements including images
            elements = partition(file_path, include_metadata=True)
            
            extracted_images = []
            for i, element in enumerate(elements):
                if isinstance(element, Image):
                    try:
                        # Get image data
                        img_data = element.metadata.get('image_data')
                        if img_data:
                            # Save image
                            img_filename = f"unstructured_{source_filename}_img{i+1}.png"
                            img_path = os.path.join(self.images_dir, "extracted", img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            # Get context from surrounding text
                            context = ""
                            if i > 0:
                                context += str(elements[i-1])[:100]
                            if i < len(elements) - 1:
                                context += str(elements[i+1])[:100]
                            
                            extracted_images.append({
                                "path": img_path,
                                "filename": img_filename,
                                "source_file": source_filename,
                                "type": "unstructured",
                                "context": context[:200],
                                "keywords": self._extract_keywords_from_context(context)
                            })
                    except Exception as e:
                        print(f"Error processing unstructured image {i}: {e}")
                        continue
            
            return extracted_images
            
        except ImportError:
            print("Unstructured library not available")
            return []
        except Exception as e:
            print(f"Error with unstructured extraction: {e}")
            return []
    
    def _extract_from_pdf(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images from PDF using PyMuPDF"""
        try:
            import fitz
        except ImportError:
            return []
        
        extracted_images = []
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Extract embedded images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Get context around image
                        img_rect = None
                        try:
                            img_rect = page.get_image_bbox(img)
                        except:
                            pass
                        
                        context = ""
                        if img_rect:
                            expanded_rect = img_rect * 1.5
                            context = page.get_text("text", clip=expanded_rect)
                        else:
                            context = page_text[:500]
                        
                        # Save image
                        img_filename = f"pdf_{source_filename}_page{page_num+1}_img{img_index+1}.png"
                        img_path = os.path.join(self.images_dir, "extracted", img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        extracted_images.append({
                            "path": img_path,
                            "filename": img_filename,
                            "source_file": source_filename,
                            "page": page_num + 1,
                            "type": "pdf_extracted",
                            "context": context.strip()[:200],
                            "keywords": self._extract_keywords_from_context(context)
                        })
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error extracting PDF image: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
        
        return extracted_images
    
    def _extract_from_docx(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images from DOCX files"""
        try:
            from docx import Document
            import zipfile
            import io
            
            extracted_images = []
            
            # Method 1: Try python-docx
            try:
                doc = Document(file_path)
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            img_data = rel.target_part.blob
                            img_filename = f"docx_{source_filename}_img{len(extracted_images)+1}.png"
                            img_path = os.path.join(self.images_dir, "extracted", img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            extracted_images.append({
                                "path": img_path,
                                "filename": img_filename,
                                "source_file": source_filename,
                                "type": "docx_extracted",
                                "context": "Image from DOCX document",
                                "keywords": []
                            })
                        except Exception as e:
                            print(f"Error extracting DOCX image: {e}")
                            continue
            except Exception as e:
                print(f"Error with python-docx: {e}")
            
            # Method 2: Try direct ZIP extraction (DOCX is a ZIP file)
            if not extracted_images:
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        for file_info in zip_file.filelist:
                            if file_info.filename.startswith('word/media/'):
                                try:
                                    img_data = zip_file.read(file_info.filename)
                                    img_filename = f"docx_zip_{source_filename}_{os.path.basename(file_info.filename)}"
                                    img_path = os.path.join(self.images_dir, "extracted", img_filename)
                                    
                                    with open(img_path, "wb") as img_file:
                                        img_file.write(img_data)
                                    
                                    extracted_images.append({
                                        "path": img_path,
                                        "filename": img_filename,
                                        "source_file": source_filename,
                                        "type": "docx_zip",
                                        "context": "Image from DOCX media folder",
                                        "keywords": []
                                    })
                                except Exception as e:
                                    print(f"Error extracting ZIP image: {e}")
                                    continue
                except Exception as e:
                    print(f"Error with ZIP extraction: {e}")
            
            return extracted_images
            
        except ImportError:
            print("python-docx not available")
            return []
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
            return []
    
    def _extract_from_excel(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images from Excel files"""
        try:
            import zipfile
            
            extracted_images = []
            
            # Excel files are ZIP files
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.startswith('xl/media/'):
                        try:
                            img_data = zip_file.read(file_info.filename)
                            img_filename = f"excel_{source_filename}_{os.path.basename(file_info.filename)}"
                            img_path = os.path.join(self.images_dir, "extracted", img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            extracted_images.append({
                                "path": img_path,
                                "filename": img_filename,
                                "source_file": source_filename,
                                "type": "excel_extracted",
                                "context": "Image from Excel document",
                                "keywords": []
                            })
                        except Exception as e:
                            print(f"Error extracting Excel image: {e}")
                            continue
            
            return extracted_images
            
        except Exception as e:
            print(f"Error extracting from Excel: {e}")
            return []
    
    def _extract_from_powerpoint(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images from PowerPoint files"""
        try:
            import zipfile
            
            extracted_images = []
            
            # PowerPoint files are ZIP files
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.startswith('ppt/media/'):
                        try:
                            img_data = zip_file.read(file_info.filename)
                            img_filename = f"ppt_{source_filename}_{os.path.basename(file_info.filename)}"
                            img_path = os.path.join(self.images_dir, "extracted", img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            extracted_images.append({
                                "path": img_path,
                                "filename": img_filename,
                                "source_file": source_filename,
                                "type": "powerpoint_extracted",
                                "context": "Image from PowerPoint presentation",
                                "keywords": []
                            })
                        except Exception as e:
                            print(f"Error extracting PowerPoint image: {e}")
                            continue
            
            return extracted_images
            
        except Exception as e:
            print(f"Error extracting from PowerPoint: {e}")
            return []
    
    def _extract_from_html(self, file_path: str, source_filename: str) -> List[Dict[str, Any]]:
        """Extract images from HTML files"""
        try:
            from bs4 import BeautifulSoup
            import requests
            from urllib.parse import urljoin, urlparse
            
            extracted_images = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            img_tags = soup.find_all('img')
            
            for i, img in enumerate(img_tags):
                try:
                    src = img.get('src')
                    if src:
                        # Handle relative URLs
                        if not urlparse(src).scheme:
                            # Assume relative to file location
                            src = os.path.join(os.path.dirname(file_path), src)
                        
                        # Download image
                        if src.startswith('http'):
                            response = requests.get(src, timeout=10)
                            img_data = response.content
                        else:
                            if os.path.exists(src):
                                with open(src, 'rb') as f:
                                    img_data = f.read()
                            else:
                                continue
                        
                        img_filename = f"html_{source_filename}_img{i+1}.png"
                        img_path = os.path.join(self.images_dir, "extracted", img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Get alt text as context
                        alt_text = img.get('alt', '')
                        context = f"HTML image: {alt_text}"
                        
                        extracted_images.append({
                            "path": img_path,
                            "filename": img_filename,
                            "source_file": source_filename,
                            "type": "html_extracted",
                            "context": context[:200],
                            "keywords": self._extract_keywords_from_context(alt_text)
                        })
                        
                except Exception as e:
                    print(f"Error extracting HTML image: {e}")
                    continue
            
            return extracted_images
            
        except ImportError:
            print("BeautifulSoup not available")
            return []
        except Exception as e:
            print(f"Error extracting from HTML: {e}")
            return []
    
    def _create_document_screenshot(self, file_path: str, source_filename: str) -> Optional[Dict[str, Any]]:
        """Create a screenshot of the document as fallback"""
        try:
            # Try to create a preview/screenshot of the document
            # This is a fallback when no images are found
            
            # For now, create a placeholder with file info
            from PIL import Image, ImageDraw, ImageFont
            
            width, height = 400, 300
            img = Image.new('RGB', (width, height), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            text = f"Document Preview\n{source_filename}\n(No images found)"
            draw.text((width//2, height//2), text, fill='black', font=font, anchor='mm')
            
            img_filename = f"preview_{source_filename}.png"
            img_path = os.path.join(self.images_dir, "screenshots", img_filename)
            img.save(img_path)
            
            return {
                "path": img_path,
                "filename": img_filename,
                "source_file": source_filename,
                "type": "preview",
                "context": f"Document preview for {source_filename}",
                "keywords": []
            }
            
        except Exception as e:
            print(f"Error creating document preview: {e}")
            return None
    
    def _save_image_to_database(self, img_data: Dict[str, Any]):
        """Save image data to database"""
        image_id = f"{img_data['type']}_{len(self.metadata)}"
        self.metadata[image_id] = img_data
    
    def _extract_keywords_from_context(self, context: str) -> List[str]:
        """Extract relevant keywords from image context"""
        context_lower = context.lower()
        keywords = []
        
        # Common Vietnamese keywords
        keyword_patterns = [
            "giao thông", "đường", "xe", "ô tô", "xe máy", "đông đúc", "hà nội", "tuyến đường",
            "nông nghiệp", "nông dân", "ruộng", "cây", "tưới", "đồng ruộng", "đồng bằng",
            "đô thị", "thành phố", "nhà cao tầng", "phát triển", "hạ tầng",
            "biến đổi khí hậu", "hạn hán", "lũ lụt", "môi trường", "ô nhiễm"
        ]
        
        for keyword in keyword_patterns:
            if keyword in context_lower:
                keywords.append(keyword)
        
        return keywords
    
    def find_relevant_images(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Find relevant images based on query and context"""
        query_lower = query.lower()
        relevant_images = []
        
        for image_id, image_data in self.metadata.items():
            score = 0
            context = image_data.get("context", "").lower()
            keywords = image_data.get("keywords", [])
            
            # Score based on context matches
            for word in query_lower.split():
                if len(word) > 2 and word in context:
                    score += 2
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 3
            
            # Score based on source file relevance
            source_file = image_data.get("source_file", "").lower()
            if any(word in source_file for word in query_lower.split()):
                score += 1
            
            if score > 0:
                relevant_images.append({
                    "id": image_id,
                    "score": score,
                    "path": image_data.get("path", ""),
                    "filename": image_data.get("filename", ""),
                    "source_file": image_data.get("source_file", ""),
                    "context": image_data.get("context", ""),
                    "type": image_data.get("type", "")
                })
        
        # Sort by score and return top results
        relevant_images.sort(key=lambda x: x["score"], reverse=True)
        return relevant_images[:max_results]
    
    def get_images_by_source(self, source_filename: str) -> List[Dict[str, Any]]:
        """Get all images from a specific source file"""
        images = []
        for image_id, image_data in self.metadata.items():
            if image_data.get("source_file") == source_filename:
                images.append({
                    "id": image_id,
                    **image_data
                })
        return images
    
    def list_all_images(self) -> List[Dict[str, Any]]:
        """List all images in database"""
        return [
            {
                "id": image_id,
                **image_data
            }
            for image_id, image_data in self.metadata.items()
        ]
    
    def clear_database(self):
        """Clear all images and metadata"""
        try:
            import shutil
            if os.path.exists(self.images_dir):
                shutil.rmtree(self.images_dir)
            self.metadata = {}
            self._ensure_database_exists()
            print("Image database cleared")
        except Exception as e:
            print(f"Error clearing database: {e}")


# Global instance
image_db = ImageDatabase()
