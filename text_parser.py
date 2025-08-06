import pytesseract
from pdf2image import convert_from_path
from ollama import Client


client = Client(host='http://localhost:11434')
pdf_path = "textset/2508.03093v1.pdf"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# poppler_path = r"C:\poppler\poppler_extracted\poppler-24.08.0\Library\bin"

prompt = 'Ты — специализированная аналитическая модель для глубокого разбора научных публикаций. Прочитай предоставленный научный текст целиком, не пропуская ни одного фрагмента, выдели все важные детали, включая мельчайшие нюансы, влияющие на выводы авторов. Восстанови и опиши пошаговую логическую цепочку, по которой учёные пришли к ключевому открытию, каждая стадия рассуждений должна быть написана тремя-четырьмя предложениями, показывающими переход от каждого результата к следующему выводу и как он подтверждает или опровергает гипотезу. То есть все рассуждения должны представлять из себя цепочку, где узел — это объект с его характеристиками и описанием, а связь — это действия над объектом, то есть глаголы. Удали лишние, второстепенные или повторяющиеся слова, оставь только важную для понимания исследовательскую информацию. После полной цепочки рассуждений поставь прочерк (—), а затем опиши идею исследования в трёх-четырёх предложениях, максимально лаконично. Не добавляй заголовков и пояснений. Пиши на русском языке, не сокращай числа, формулы и термины. Используй как можно больше слов. Логическая цепочка должна быть как можно более длинная и развёрнутая. Ответ составляй от своего имени и не упоминай имён авторов. Следуй строго этой структуре.'

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

user_prompt = f"{global_text}\n Ответь по-русски, используй как можно больше слов"

messages = [
    {
        'role': 'system',
        'content': prompt
    },
    {
        'role': 'user',
        'content': user_prompt
    }
]

response = client.chat(
    model="blackened/t-lite:latest",
    messages=messages,
    options={
        "num_predict": 8192,
        "temperature": 0.1,
    },
    stream=False
)

print(response['message']['content'])
# python3 text_parser.py
