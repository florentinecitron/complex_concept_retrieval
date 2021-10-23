import itertools as it
import json
from typing import List, Dict

import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
import collections as co

from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, Subset
import random
import math


import numpy as np
np.random.seed(0)


ignore_categories = set([
    "NA",
    "H",
    '000', # No other category applies
    # these are not complex concepts, but rather particular geographic entities
    '110', # European Community/Union: Negative
    '1011', # Russia/USSR/CIS: Positive
    '1012', # Western States: Positive
    '1013', # Eastern European Countries: Positive
    '1014', # Baltic States: Positive
    '1015', # Nordic Council: Positive
    '1016', # SFR Yugoslavia: Positive
    '1021', # Russia/USSR/CIS: Negative
    '1022', # Western States: Negative
    '1023', # East European Countries: Negative
    '1024', # Baltic States: Negative
    '1025', # Nordic Council: Negative
    '1026', # SFR Yugoslavia: Negative
    '1031', # Russian Army: Negative
    '6011', # The Karabakh Issue: Positive
    '6014', # Cyprus Issue
])


@dataclass
class ManifestoDataset(Dataset):
    texts_encoded: torch.tensor
    labels_encoded: torch.tensor
    labels_texts: List[str]
    labels: List[int]
    keys: List[str]
    df_codebook: pd.DataFrame
    df_metadata: pd.DataFrame
    cmp_code_to_top_level_title: dict
    top_level_title_to_id: dict
    test_queries: List[str]
    test_query_encodings: torch.tensor
    texts: List[str]
    holdout_labels: List[int]

    def __getitem__(self, index):
        anchor, target = self.texts_encoded[index], self.labels[index]
        # now pair this up with the correct query
        posneg = self.labels_encoded[target]
        #return anchor, posneg, target
        return anchor, posneg, target

    def __len__(self):
        return self.texts_encoded.shape[0]

    @classmethod
    def load(cls, data_path=Path("../data"), encoder_model_name='paraphrase-MiniLM-L6-v2'):
        # todo return metadata so the sentence can be linked back to its metadata like year, country, political party

        # labels will be the reduced set of labels
        sentence_labels = []
        # the encoded sentences
        sentence_embeddings = []
        # the ids of the encoded manfestos for indexing back into df_metadata
        sentence_keys = []

        texts = []

        # codebook contains the label names/descriptions in English
        with (data_path / "codebook.json").open() as i:
            codebook_json = json.load(i)
        df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])
        df_codebook["top_level_title"] = df_codebook.title.map(lambda x: x.split(": ")[0])
        cmp_code_to_top_level_title = {
            r.code: r.top_level_title
            for r
            in df_codebook[["code", "top_level_title"]].itertuples()
        }
        #distinct_top_level_titles = df_codebook.top_level_title.unique()
        #top_level_to_new_code = {t: str(i) for i, t in enumerate(distinct_top_level_titles)}
        #code_to_top_level_code = {r.code: top_level_to_new_code[r.top_level_title] for r in df_codebook.itertuples()}
        #df_codebook["top_level_code"] = df_codebook.code.map(code_to_top_level_code.get)

        # for each manifesto, find out what language it is
        with (data_path / "metadata.jsonl").open() as i:
            df_metadata = pd.DataFrame(it.chain.from_iterable([json.loads(line)["items"] for line in i]))
        df_metadata["key"] = df_metadata.party_id.map(str) + "_" + df_metadata.election_date.map(str)
        key_to_lang = {i.key: i.language for i in df_metadata.itertuples()}

        top_level_title_to_id = co.defaultdict(it.count(0).__next__)

        # load saved sentence encodings
        with (data_path / "texts_and_annotations.jsonl").open() as i:
            for line in tqdm.tqdm(i, total=233):
                data = json.loads(line)
                for outer_items in data["items"]:
                    #_items = outer_items["items"]

                    key = outer_items["key"]
                    lang = key_to_lang[key]
                    if lang != "english":
                        continue

                    # cmp_code to top level title
                    # top_level title to next int

                    items = [i for i in outer_items["items"] if i["cmp_code"] not in ignore_categories]

                    sentence_keys.extend([key]*len(items))
                    sentence_indices = [idx for idx, i in enumerate(outer_items["items"]) if i["cmp_code"] not in ignore_categories]
                    sentence_labels.extend([top_level_title_to_id[cmp_code_to_top_level_title[i["cmp_code"]]] for i in items])

                    texts.extend(i["text"] for i in items)

                    emb_file = data_path / "text_encodings" / f"{key}.pth"

                    texts_encodings = torch.tensor(torch.load(emb_file))[sentence_indices, :]
                    sentence_embeddings.append(texts_encodings)

        model = SentenceTransformer(encoder_model_name, device="cpu")

        distinct_top_level_titles = [x[0] for x in sorted(top_level_title_to_id.items(), key=lambda x: x[1])]
        label_title_encodings = model.encode(distinct_top_level_titles, convert_to_tensor=True)
        #top_level_code_to_title_embedding = dict(zip(df_codebook["top_level_code"], label_title_encodings))

        test_queries = [
            "Vaccination policy",
            "Digitalization",
            "Cryptocurrency",
            "Universal basic income",
            "Systemic racism",
            "Misinformation"
        ]
        test_query_encodings = model.encode(
            test_queries,
            convert_to_tensor=True
        )

        #int2label = {idx: label for idx, label in enumerate(sentence_labels)}
        #label2int = {label: idx for idx, label in enumerate(sentence_labels)}
        #labels = [label2int[label] for label in labels]
        #distinct_classes = list(set(labels))

        distinct_labels = set(sentence_labels)
        n_holdout_labels = int(math.floor(len(distinct_labels) * .2))
        holdout_labels = random.sample(list(distinct_labels), k=n_holdout_labels)

        return cls(
            torch.vstack(sentence_embeddings),
            label_title_encodings,
            distinct_top_level_titles,
            sentence_labels,
            sentence_keys,
            df_codebook,
            df_metadata,
            cmp_code_to_top_level_title,
            top_level_title_to_id,
            test_queries,
            test_query_encodings,
            texts,
            holdout_labels
        )

    def split_by_holdout_labels(self):
        train_indices = [idx for idx, label in enumerate(self.labels) if label not in self.holdout_labels]
        holdout_indices = [idx for idx, label in enumerate(self.labels) if label in self.holdout_labels]
        return Subset(self, train_indices), Subset(self, holdout_indices)