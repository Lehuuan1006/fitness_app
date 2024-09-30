import fitz  # PyMuPDF để đọc file PDF
import os
import json
from tqdm import tqdm
from datetime import datetime

# Hàm trích xuất văn bản từ file PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

# Hàm trích xuất metadata từ file PDF
def extract_metadata_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    return metadata

# Hàm xử lý chính
def process_pdf_files(pdf_directory):
    # Tạo thư mục lưu dữ liệu nếu chưa có
    os.makedirs('data/pdf', exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    print(f"Total PDF files found: {len(pdf_files)}")
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        # Trích xuất metadata và nội dung
        metadata = extract_metadata_from_pdf(pdf_path)
        text_content = extract_text_from_pdf(pdf_path)
        
        # Định dạng ngày xuất bản
        if metadata.get('creationDate'):
            try:
                creation_date = datetime.strptime(metadata['creationDate'][2:10], "%Y%m%d")
                formatted_date = creation_date.strftime("%Y-%m-%d")
            except Exception:
                formatted_date = "N/A"
        else:
            formatted_date = "N/A"
        
        # Tạo dữ liệu để lưu
        pdf_data = {
            'file_name': pdf_file,
            'title': metadata.get('title', 'N/A'),
            'author': metadata.get('author', 'N/A'),
            'creation_date': formatted_date,
            'subject': metadata.get('subject', 'N/A'),
            'keywords': metadata.get('keywords', 'N/A'),
            'text': text_content
        }
        
        # Lưu dữ liệu dưới dạng JSON
        json_file = os.path.join('data/pdf', f'{os.path.splitext(pdf_file)[0]}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=4)

        print(f"Processed {pdf_file}, saved as {json_file}")

if __name__ == "__main__":
    pdf_directory = "C:/Code/doan2/data"  # Đường dẫn tới thư mục chứa PDF
    process_pdf_files(pdf_directory)
