import arxiv
import os
from tqdm import tqdm
import time
import json



querys = "Applications; Computation; Machine Learning; Methodology; Other Statistics; Statistics Theory".split("; ")
for query in querys:
    output_dir = "arxiv_papers"
    progress_file = "progress.json"
    max_results = 10000
    batch_size = 100

    os.makedirs(output_dir, exist_ok=True)

    def load_downloaded_ids():
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('downloaded_ids', []))
        return set()

    def save_downloaded_ids(downloaded_ids):
        with open(progress_file, 'w') as f:
            json.dump({'downloaded_ids': list(downloaded_ids)}, f)

    downloaded_ids = load_downloaded_ids()
    print(f"Загружено {len(downloaded_ids)} уже скачанных статей.")

    total_downloaded = len(downloaded_ids)
    while total_downloaded < max_results:
        try:
            search = arxiv.Search(
                query=query,
                max_results=batch_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            papers = list(search.results())

            new_papers = [p for p in papers if p.get_short_id() not in downloaded_ids]

            if not new_papers:
                print("Новых документов не найдено, завершаем загрузку.")
                break

            for paper in tqdm(new_papers, desc=f"Downloading batch, total downloaded={total_downloaded}"):
                try:
                    paper_id = paper.get_short_id()
                    filename = f"{paper_id}.pdf"
                    file_path = os.path.join(output_dir, filename)
                    if not os.path.exists(file_path):
                        paper.download_pdf(dirpath=output_dir, filename=filename)
                    downloaded_ids.add(paper_id)
                    total_downloaded += 1

                    if total_downloaded >= max_results:
                        break
                except Exception as e:
                    print(f"Ошибка при скачивании {paper.title}: {e}")

            save_downloaded_ids(downloaded_ids)

            time.sleep(3)

        except Exception as e:
            print(f"Произошла ошибка: {e}")
            break

    print(f"Готово! Всего скачано: {total_downloaded} PDF-файлов в папку {output_dir}")
print("OK")
