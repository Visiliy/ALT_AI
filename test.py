import arxiv
import os
from tqdm import tqdm 


query = "Data Structures AND Algorithms"
output_dir = "arxiv_papers"
max_results = 1000

os.makedirs(output_dir, exist_ok=True)


search = arxiv.Search(
    query=query,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in tqdm(search.results(), desc="Downloading papers"):
    try:
        paper.download_pdf(dirpath=output_dir, filename=f"{paper.get_short_id()}.pdf")
    except Exception as e:
        print(f"Ошибка при скачивании {paper.title}: {e}")

print(f"Готово! PDF-файлы сохранены в папку {output_dir}")