"""
Image Database Service - Extract REAL images from ALL uploaded files
Stores actual images extracted from any document type

Mục đích:
- Trích xuất ảnh từ tất cả các loại file (DOCX, PDF, Excel, PowerPoint, HTML)
- Lưu trữ ảnh với metadata (context, keywords, source file)
- Tìm kiếm ảnh liên quan dựa trên query của user
- Cung cấp API để truy xuất ảnh từ database

Công nghệ sử dụng:
- python-docx: Trích xuất ảnh từ DOCX
- PyMuPDF (fitz): Trích xuất ảnh từ PDF
- zipfile: Trích xuất ảnh từ Excel/PowerPoint (vì là ZIP files)
- PIL (Pillow): Xử lý và lưu ảnh
- Unstructured: Trích xuất tổng quát (nếu có)
"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path

class ImageDatabase:
    """
    Image database service that extracts REAL images from ANY uploaded files
    
    Chức năng chính:
    1. Trích xuất ảnh từ các file tài liệu
    2. Lưu trữ ảnh với metadata
    3. Tìm kiếm ảnh liên quan
    4. Quản lý database ảnh
    """
    
    def __init__(self):
        """
        Khởi tạo Image Database
        - Tạo thư mục lưu trữ ảnh
        - Load metadata từ file JSON
        - Đảm bảo cấu trúc database tồn tại
        """
        self.images_dir = "image_database"  # Thư mục chính lưu ảnh
        self.metadata_file = os.path.join(self.images_dir, "image_metadata.json")  # File metadata
        self._ensure_database_exists()  # Tạo cấu trúc thư mục
        self._load_metadata()  # Load metadata hiện có
    
    def _ensure_database_exists(self):
        """
        Tạo cấu trúc thư mục cho image database
        
        Cấu trúc:
        image_database/
        ├── extracted/     # Ảnh trích xuất từ tài liệu
        ├── screenshots/   # Screenshot của tài liệu
        ├── documents/     # Ảnh từ document preview
        └── general/       # Ảnh tổng quát khác
        """
        # Tạo thư mục chính nếu chưa tồn tại
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Tạo các thư mục con cho từng loại ảnh
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
        """
        Trích xuất ảnh từ BẤT KỲ loại file nào sử dụng nhiều phương pháp
        
        Logic:
        1. Thử Unstructured library (toàn diện nhất)
        2. Thử extractor riêng cho từng loại file
        3. Tạo screenshot nếu không tìm thấy ảnh
        
        Args:
            file_path: Đường dẫn đến file
            source_filename: Tên file gốc
            
        Returns:
            List[Dict]: Danh sách ảnh đã trích xuất với metadata
        """
        extracted_images = []
        
        # Method 1: Thử Unstructured library (toàn diện nhất)
        # Unstructured có thể xử lý nhiều loại file và trích xuất ảnh tốt
        unstructured_images = self._extract_with_unstructured(file_path, source_filename)
        if unstructured_images:
            extracted_images.extend(unstructured_images)
            print(f"Unstructured extracted {len(unstructured_images)} images")
        
        # Method 2: Thử extractor riêng cho từng loại file
        # Dựa vào extension để chọn phương pháp phù hợp
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # PDF: Sử dụng PyMuPDF để trích xuất ảnh embedded
            pdf_images = self._extract_from_pdf(file_path, source_filename)
            extracted_images.extend(pdf_images)
            print(f"PDF extractor found {len(pdf_images)} images")
        
        elif file_ext in ['.docx', '.doc']:
            # DOCX: Sử dụng python-docx hoặc ZIP extraction
            docx_images = self._extract_from_docx(file_path, source_filename)
            extracted_images.extend(docx_images)
            print(f"DOCX extractor found {len(docx_images)} images")
        
        elif file_ext in ['.xlsx', '.xls']:
            # Excel: Sử dụng ZIP extraction (Excel là ZIP file)
            excel_images = self._extract_from_excel(file_path, source_filename)
            extracted_images.extend(excel_images)
            print(f"Excel extractor found {len(excel_images)} images")
        
        elif file_ext in ['.pptx', '.ppt']:
            # PowerPoint: Sử dụng ZIP extraction (PPTX là ZIP file)
            ppt_images = self._extract_from_powerpoint(file_path, source_filename)
            extracted_images.extend(ppt_images)
            print(f"PowerPoint extractor found {len(ppt_images)} images")
        
        elif file_ext in ['.html', '.htm']:
            # HTML: Sử dụng BeautifulSoup để tìm thẻ img
            html_images = self._extract_from_html(file_path, source_filename)
            extracted_images.extend(html_images)
            print(f"HTML extractor found {len(html_images)} images")
        
        # Method 3: Tạo document screenshot nếu không tìm thấy ảnh
        # Fallback: Tạo ảnh preview của tài liệu
        if not extracted_images:
            screenshot = self._create_document_screenshot(file_path, source_filename)
            if screenshot:
                extracted_images.append(screenshot)
                print("Created document screenshot as fallback")
        
        # Lưu tất cả ảnh đã trích xuất vào database
        for img in extracted_images:
            self._save_image_to_database(img)
        
        # Lưu metadata
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
        """
        Trích xuất ảnh từ file PDF sử dụng PyMuPDF (fitz)
        
        Logic:
        1. Mở PDF file với PyMuPDF
        2. Duyệt qua từng trang
        3. Tìm ảnh embedded trong mỗi trang
        4. Trích xuất ảnh và context xung quanh
        5. Lưu ảnh với metadata
        
        Args:
            file_path: Đường dẫn đến file PDF
            source_filename: Tên file gốc
            
        Returns:
            List[Dict]: Danh sách ảnh đã trích xuất từ PDF
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("PyMuPDF not available - cannot extract PDF images")
            return []
        
        extracted_images = []
        try:
            # Mở PDF file
            doc = fitz.open(file_path)
            
            # Duyệt qua từng trang trong PDF
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # Load trang
                page_text = page.get_text()     # Lấy text của trang
                
                # Tìm danh sách ảnh embedded trong trang
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Lấy reference ID của ảnh
                        xref = img[0]
                        # Tạo Pixmap từ ảnh
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Xử lý ảnh dựa trên số kênh màu
                        if pix.n - pix.alpha < 4:  # Ảnh RGB hoặc Grayscale
                            img_data = pix.tobytes("png")
                        else:  # Ảnh CMYK hoặc khác, chuyển sang RGB
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None  # Giải phóng bộ nhớ
                        
                        # Lấy context xung quanh ảnh
                        img_rect = None
                        try:
                            # Tìm vùng chứa ảnh trên trang
                            img_rect = page.get_image_bbox(img)
                        except:
                            pass
                        
                        context = ""
                        if img_rect:
                            # Mở rộng vùng để lấy text xung quanh ảnh
                            expanded_rect = img_rect * 1.5
                            context = page.get_text("text", clip=expanded_rect)
                        else:
                            # Fallback: lấy 500 ký tự đầu của trang
                            context = page_text[:500]
                        
                        # Tạo tên file và đường dẫn lưu ảnh
                        img_filename = f"pdf_{source_filename}_page{page_num+1}_img{img_index+1}.png"
                        img_path = os.path.join(self.images_dir, "extracted", img_filename)
                        
                        # Lưu ảnh vào file
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Thêm metadata cho ảnh
                        extracted_images.append({
                            "path": img_path,
                            "filename": img_filename,
                            "source_file": source_filename,
                            "page": page_num + 1,
                            "type": "pdf_extracted",
                            "context": context.strip()[:200],  # Giới hạn context 200 ký tự
                            "keywords": self._extract_keywords_from_context(context)
                        })
                        
                        pix = None  # Giải phóng bộ nhớ
                        
                    except Exception as e:
                        print(f"Error extracting PDF image: {e}")
                        continue
            
            doc.close()  # Đóng PDF file
            
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
                
                # Extract all text for context analysis
                full_text = []
                for paragraph in doc.paragraphs:
                    full_text.append(paragraph.text)
                document_text = "\n".join(full_text)
                
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            img_data = rel.target_part.blob
                            img_filename = f"docx_{source_filename}_img{len(extracted_images)+1}.png"
                            img_path = os.path.join(self.images_dir, "extracted", img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            # Try to find relevant context for this image
                            context = self._find_image_context(document_text, len(extracted_images))
                            
                            extracted_images.append({
                                "path": img_path,
                                "filename": img_filename,
                                "source_file": source_filename,
                                "type": "docx_extracted",
                                "context": context,
                                "keywords": self._extract_keywords_from_context(context)
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
    
    def _find_image_context(self, document_text: str, image_index: int) -> str:
        """Find relevant context for an image based on document text"""
        # Look for image-related sections in the document
        lines = document_text.split('\n')
        
        # Common image context patterns
        image_patterns = [
            "hình", "ảnh", "image", "figure", "photo", "bức ảnh", "hình ảnh",
            "giao thông", "đường", "xe", "ô tô", "xe máy", "đông đúc", "hà nội",
            "nông nghiệp", "nông dân", "ruộng", "cây", "tưới", "đồng ruộng"
        ]
        
        # Find lines that might contain image context
        relevant_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in image_patterns):
                # Get surrounding context (previous and next lines)
                start = max(0, i-2)
                end = min(len(lines), i+3)
                context_block = lines[start:end]
                relevant_lines.extend(context_block)
        
        if relevant_lines:
            # Return the most relevant context for this image
            context = "\n".join(relevant_lines[:5])  # Limit to 5 lines
            return context.strip()
        
        # Fallback: return a section of the document
        return document_text[:500] + "..." if len(document_text) > 500 else document_text

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
        """
        Tìm ảnh liên quan dựa trên query và context
        
        Logic tính điểm:
        1. Context matches: +2 điểm cho mỗi từ khớp trong context
        2. Keyword matches: +3 điểm cho mỗi keyword khớp
        3. Source file relevance: +1 điểm nếu tên file liên quan
        4. General image queries: +5 điểm nếu query hỏi về ảnh nói chung
        
        Args:
            query: Câu hỏi của user (đã chuyển thành lowercase)
            max_results: Số lượng ảnh tối đa trả về
            
        Returns:
            List[Dict]: Danh sách ảnh liên quan, sắp xếp theo điểm số
        """
        query_lower = query.lower()  # Chuyển query thành lowercase để so sánh
        relevant_images = []
        
        # Check if this is a general image query
        general_image_keywords = ["hình", "ảnh", "image", "picture", "photo", "có ko", "gì"]
        is_general_image_query = any(keyword in query_lower for keyword in general_image_keywords)
        
        # Duyệt qua tất cả ảnh trong metadata
        for image_id, image_data in self.metadata.items():
            score = 0  # Điểm số của ảnh này
            context = image_data.get("context", "").lower()  # Context của ảnh
            keywords = image_data.get("keywords", [])        # Keywords của ảnh
            
            # Nếu là câu hỏi chung về ảnh, ưu tiên ảnh có context rõ ràng
            if is_general_image_query:
                if context and len(context) > 10:  # Có context mô tả
                    score += 5
                if keywords:  # Có keywords
                    score += 3
            
            # Tính điểm dựa trên context matches
            # Tách query thành các từ và kiểm tra trong context
            for word in query_lower.split():
                if len(word) > 2 and word in context:  # Chỉ xét từ có >2 ký tự
                    score += 2  # +2 điểm cho mỗi từ khớp
            
            # Tính điểm dựa trên keyword matches
            # Keywords được trích xuất từ context, có trọng số cao hơn
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 3  # +3 điểm cho mỗi keyword khớp
            
            # Tính điểm dựa trên tên file nguồn
            # Nếu tên file chứa từ khóa trong query
            source_file = image_data.get("source_file", "").lower()
            if any(word in source_file for word in query_lower.split()):
                score += 1  # +1 điểm nếu tên file liên quan
            
            # Thêm tất cả ảnh có điểm > 0 hoặc là câu hỏi chung về ảnh
            if score > 0 or is_general_image_query:
                relevant_images.append({
                    "id": image_id,
                    "score": score,
                    "path": image_data.get("path", ""),
                    "filename": image_data.get("filename", ""),
                    "source_file": image_data.get("source_file", ""),
                    "context": image_data.get("context", ""),
                    "type": image_data.get("type", "")
                })
        
        # Sắp xếp theo điểm số giảm dần và trả về top results
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
