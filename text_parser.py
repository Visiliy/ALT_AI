import pdf2image
import pytesseract

pdf_path = "textset/2508.00545v1.pdf"

pages = pdf2image.convert_from_path(pdf_path, dpi=500)

for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang='eng')
    print(f"Текст со страницы {i+1}:\n{text}\n{'='*40}")


# python3 text_parser.py