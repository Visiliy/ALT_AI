import pytesseract
from pdf2image import convert_from_path


pdf_path = "2507.22836v1.pdf"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# poppler_path = r"C:\poppler\poppler_extracted\poppler-24.08.0\Library\bin"


pages = convert_from_path(
    pdf_path,
    dpi=500,
    # poppler_path=poppler_path,
    use_pdftocairo=False
)
global_text = ""
for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang='eng')
    global_text += f"\n {text}"


word = "References"
index = global_text.find(word)

if index != -1:
    global_text = global_text[:index + len(word)]
else:
    global_text = global_text

with open("text.txt", "w", encoding="utf-8") as file:
    file.write(global_text)