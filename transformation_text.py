import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import networkx as nx
import spacy

# 1) Загрузка моделей
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
nlp = spacy.load("ru_core_news_sm")

# 2) Параметры
ATTENTION_THRESHOLD = 0.2
WINDOW = 3               # сколько слов брать вокруг сущ.
ROOT_LABEL = "DOC_ROOT"
ROOT_EMB = embedder.encode([ROOT_LABEL])[0]


def build_kg_from_text(text, window=WINDOW, add_root=True):
    """
    Строит KG, в котором:
      - узлы = noun_chunks (или одиночные NOUN/PROPN) + их контекст WINDOW
      - рёбра = глагол (lemma) + его модификаторы (advmod, neg, prt)
      - каждое emb_list и emb_mean для узлов и рёбер
      - опционально добавляется корневой узел, чтобы граф был связным
    """
    doc = nlp(text)
    G = nx.DiGraph()

    # --- 1) Собираем noun_chunks + одиночные существительные ---
    noun_spans = list(doc.noun_chunks)
    seen = set()
    for span in noun_spans:
        seen.update(range(span.start, span.end))
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.i not in seen:
            noun_spans.append(token)

    noun_nodes = {}
    for span in noun_spans:
        # Определяем границы span
        if hasattr(span, "start"):
            start_i = span.start
            span_tokens = [t for t in span]
            label = " ".join(t.text for t in span_tokens)
        else:
            start_i = span.i
            span_tokens = [span]
            label = span.text

        # Контекст вокруг span
        left = max(0, start_i - window)
        right = min(len(doc), start_i + len(span_tokens) + window)
        ctx = [t for t in doc[left:right] if t.i < start_i or t.i >= start_i + len(span_tokens)]

        # Тексты и эмбеддинги
        texts = [t.text for t in span_tokens] + [t.text for t in ctx]
        embs = embedder.encode(texts)  # shape=(len(texts), dim)

        G.add_node(
            start_i,
            label=label,
            desc_tokens=[t.text for t in ctx],
            emb_list=embs,
            emb_mean=embs.mean(axis=0)
        )
        noun_nodes[start_i] = span_tokens

    # --- 2) Извлечение глагольных рёбер ---
    for token in doc:
        if token.pos_ == "VERB":
            subs = [t for t in token.lefts if t.dep_ == "nsubj" and t.i in noun_nodes]
            objs = [t for t in token.rights if t.dep_ in ("obj", "iobj") and t.i in noun_nodes]
            for subj in subs:
                for obj in objs:
                    # Модификаторы глагола
                    mods = [t for t in token.children if t.dep_ in ("advmod", "neg", "prt")]
                    texts = [token.lemma_] + [m.text for m in mods]
                    embs = embedder.encode(texts)

                    G.add_edge(
                        subj.i,
                        obj.i,
                        label=token.lemma_,
                        verb_tokens=texts,
                        emb_list=embs,
                        emb_mean=embs.mean(axis=0),
                        role="nsubj->obj"
                    )

    # --- 3) (опционально) Добавляем корень для связности ---
    if add_root:
        G.add_node(
            ROOT_LABEL,
            label=ROOT_LABEL,
            desc_tokens=[],
            emb_list=[ROOT_EMB],
            emb_mean=ROOT_EMB
        )
        for comp in nx.weakly_connected_components(G):
            if ROOT_LABEL in comp:
                continue
            rep = next(iter(comp))
            G.add_edge(
                ROOT_LABEL,
                rep,
                label="root_connect",
                verb_tokens=[ROOT_LABEL],
                emb_list=[ROOT_EMB],
                emb_mean=ROOT_EMB,
                role="root"
            )

    return G


# === Пример использования ===
if __name__ == "__main__":
    sample_text = (
        "Алгоритм сортировки обрабатывает массив и выводит отсортированный результат. "
        "Он также может фильтровать дубликаты."
    )
    kg = build_kg_from_text(sample_text)

    print("Узлы:")
    for nid, data in kg.nodes(data=True):
        print(f" {nid}: '{data['label']}' | контекст={data['desc_tokens']}")

    print("\nРёбра:")
    for u, v, data in kg.edges(data=True):
        subj = kg.nodes[u]['label']
        obj  = kg.nodes[v]['label']
        print(f" {subj} -> {obj} | глагол='{data['label']}' | модификаторы={data['verb_tokens']}")
