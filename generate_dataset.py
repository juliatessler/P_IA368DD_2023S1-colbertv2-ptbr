import csv
import json
import os
import pickle

import pandas as pd
import torch
from pyserini.search import LuceneSearcher
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
searcher = LuceneSearcher("indexes/portuguese-lucene-index-msmarco")
model_name = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
query_file_name = "datasets/portuguese_queries.train.tsv"  # for the small version, use queries_100.tsv


def search_with_bm25(query, k=1000):
    return searcher.search(query, k)


def reranking(docs, max_len=512, max=1000, batch_size=500):
    results = []

    for i in range(0, len(docs), batch_size):
        i_end = i + batch_size
        i_end = len(docs) if i_end > len(docs) else i_end

        batch = docs[i:i_end]

        queries_ = [sample["query"] for sample in batch]
        passages_ = [sample["text"] for sample in batch]

        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                features = tokenizer(
                    queries_,
                    passages_,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_len,
                ).to(model.device)

                predictions = model(**features).logits.flatten()

        for score, result in zip(predictions, batch):
            results.append((result, score.item()))

    sorted_list = sorted(results, key=lambda x: x[1], reverse=True)

    return sorted_list[:max]


def generate_list_with_scores():
    print("***** GENERATE LIST WITH SCORES *****")
    if not os.path.isfile(f"datasets/dataset_list.pickle"):
        dataset_list = []
        col_names = ["id", "text"]
        df_queries = pd.read_csv(f"{query_file_name}", names=col_names, delimiter="\t")

        for index, row in tqdm(df_queries.iterrows(), total=df_queries.shape[0]):
            try:
                # Fisrt stage (BM25)
                hits = search_with_bm25(row["text"], 1000)
                docs = []

                for hit in hits:
                    hit_dict = json.loads(hit.raw)
                    doc = {
                        "passage_id": int(hit.docid),
                        "query_id": row["id"],
                        "query": row["text"],
                        "text": hit_dict["contents"],
                    }
                    docs.append(doc)

                # Second stage (reranking)
                docs_reranking = reranking(docs, max=1000)

                if len(docs_reranking) > 0:
                    item = {
                        "query_id": docs_reranking[0][0]["query_id"],
                        "positive": {
                            "id": docs_reranking[0][0]["passage_id"],
                            "score": docs_reranking[0][1],
                        },
                        "negatives": [],
                    }

                    for doc in docs_reranking[-100:]:
                        negative = {"id": doc[0]["passage_id"], "score": doc[1]}

                        item["negatives"].append(negative)

                    dataset_list.append(item)
            except Exception as ex:
                print(ex)
                with open(f"datasets/dataset_list.pickle", "wb") as f:
                    pickle.dump(dataset_list, f)

        with open(f"datasets/dataset_list.pickle", "wb") as f:
            pickle.dump(dataset_list, f)
    else:
        with open(f"datasets/dataset_list.pickle", "rb") as input_file:
            dataset_list = pickle.load(input_file)

    return dataset_list


def generate_dataset():
    dataset_list = generate_list_with_scores()
    print("***** GENERATE DATASET TRIPLE *****")

    with open(
        f"datasets/triples.train.tsv", "w", encoding="utf8", newline=""
    ) as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        for item in tqdm(dataset_list):
            for negative in item["negatives"]:
                line = f"[{str(item['query_id'])}, [{str(item['positive']['id'])}, {item['positive']['score']}], [{str(negative['id'])}, {negative['score']}]]"
                tsv_writer.writerow([line])

    print("***** FINISH *****")


if __name__ == "__main__":
    generate_dataset()
