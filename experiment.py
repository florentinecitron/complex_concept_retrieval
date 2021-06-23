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


def encode_labels(model):
    with open("data/codebook.json") as i:
        codebook_json = json.load(i)
    df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])
    title_encodings = model.encode(df_codebook.description_md, convert_to_tensor=True)
    return list(df_codebook.code), title_encodings


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")
    label_codes, label_title_encodings = encode_labels(model)
    label2idx = {l: i for i, l in enumerate(label_codes)}

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
                items = [i for i in outer_items["items"] if i["cmp_code"] not in ["NA", "H"]]
                sentence_indices = [idx for idx, i in enumerate(outer_items["items"]) if i["cmp_code"] not in ["NA", "H"]]
                labels.extend([i["cmp_code"] for i in items])

                key = outer_items["key"]
                emb_file = f"data/text_encodings/{key}.pth"

                lang = key_to_lang[key]
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
    max_preds = [label_codes[i] for i in top_scores]
    labels_as_ints = [label2idx[l] for l in labels]
    seen_labels = set(labels)
    seen_labels_idxs = [i for i in range(len(label_codes)) if label_codes[i] in seen_labels]

    df_results = pd.DataFrame({"pred": max_preds, "label": labels, "lang": langs})
    df_results["correct"] = df_results.pred == df_results.label
    df_results.groupby("lang")["correct"].mean().sort_values(ascending=False)

    english_indexes = [i for i, v in enumerate(langs) if v == "english"]

    metrics.top_k_accuracy_score(np.array(labels_as_ints)[english_indexes], all_scores[english_indexes, seen_labels_idxs].numpy(), k=10)
    # 0.135

    # TODO: focus only on english; merge similar labels; consider training model