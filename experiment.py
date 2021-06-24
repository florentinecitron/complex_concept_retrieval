# need to encode all sentences. start with english, german

# encode all labels (labels positive/negative should be merged first?)
# encode all labels + codebook guidelines
# for each label, find threshold that optimizes accuracy
# find threshold that optimizes global accuracy


# load data/metadata.json and get all items
# iterate through data/texts_and_annotations.jsonl and encode


import itertools as it
import json
import os.path

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import torch
import tqdm
from sentence_transformers import SentenceTransformer, util


# TODO: merge similar labels; consider training model
# i.e. focus on "top-level" categories


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

    with open("data/codebook.json") as i:
        codebook_json = json.load(i)
    df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])
    df_codebook["top_level_title"] = df_codebook.title.map(lambda x: x.split(": ")[0])
    distinct_top_level_titles = df_codebook.top_level_title.unique()
    top_level_to_new_code = {t: str(i) for i, t in enumerate(distinct_top_level_titles)}
    code_to_top_level_code = {r.code: top_level_to_new_code[r.top_level_title] for r in df_codebook.itertuples()}

    label_title_encodings = model.encode(distinct_top_level_titles, convert_to_tensor=True)

    #label_codes = list(df_codebook.code)
    #label2idx = {l: i for i, l in enumerate(label_codes)}

    with open("data/metadata.jsonl") as i:
        df_metadata = pd.DataFrame(it.chain.from_iterable([json.loads(line)["items"] for line in i]))
    df_metadata["key"] = df_metadata.party_id.map(str) + "_" + df_metadata.election_date.map(str)
    key_to_lang = {i.key: i.language for i in df_metadata.itertuples()}

    labels = []
    scores = []
    langs = []

    with open("data/texts_and_annotations.jsonl") as i:
        for line in tqdm.tqdm(i, total=233):
            data = json.loads(line)
            for outer_items in data["items"]:
                #_items = outer_items["items"]

                key = outer_items["key"]
                lang = key_to_lang[key]
                if lang != "english":
                    continue

                items = [i for i in outer_items["items"] if i["cmp_code"] not in ["NA", "H"]]
                sentence_indices = [idx for idx, i in enumerate(outer_items["items"]) if i["cmp_code"] not in ["NA", "H"]]
                labels.extend([code_to_top_level_code[i["cmp_code"]] for i in items])


                emb_file = f"data/text_encodings/{key}.pth"


                langs.extend([lang] * len(items))

                if not os.path.exists(emb_file):
                    texts = [i["text"] for i in items]
                    texts_encodings = model.encode(texts)
                    torch.save(texts_encodings, emb_file)
                else:
                    texts_encodings = torch.load(emb_file)

                cosine_scores = util.pytorch_cos_sim(label_title_encodings, texts_encodings[sentence_indices, :])
                scores.append(cosine_scores)

    all_scores = torch.hstack( scores).T
    top_scores = all_scores.argmax(1)
    max_preds = [str(i) for i in top_scores.tolist()]
    labels_as_ints = list(map(int, labels))
    seen_labels_idxs = list(set(labels_as_ints)) #[i for i in range(len(distinct_top_level_titles)) if distinct_top_level_titles[i] in seen_labels]
    seen_labels_idxs_as_str = list(map(str, seen_labels_idxs))

    df_results = pd.DataFrame({"pred": max_preds, "label": labels, "lang": langs})
    df_results["correct"] = df_results.pred == df_results.label

    # top-k accuracy

    metrics.top_k_accuracy_score(np.array(labels_as_ints), all_scores.numpy()[:, seen_labels_idxs], k=3)
    # 0.33

    # get accuracy per label, with counts per label

    df_label_results = pd.DataFrame(
        {
            "accuracy": df_results.groupby("label")["correct"].mean(),
            "n": df_results.groupby("label").pred.count()
        }
    )
    df_label_results.sort_values("accuracy", ascending=False)

    # get the highest confusion entries

    seen_label_names = [distinct_top_level_titles[i] for i in seen_labels_idxs]#
    confusion_matrix = metrics.confusion_matrix(labels, max_preds, labels=seen_labels_idxs_as_str )

    confusion_values = []
    for i, c1 in enumerate(seen_label_names):
        row = confusion_matrix[i]
        row_norm = row/row.sum()
        for j, c2 in enumerate(seen_label_names):
            if c1 == c2:
                continue
            confusion_values.append((c1, c2, row_norm[j]))

    print(tabulate.tabulate(sorted(confusion_values, key=lambda x: x[2], reverse=True)[:20], headers=["True", "Predicted", "proportion"]))

    # how much of the confusion is attributable to inter-label similarities?