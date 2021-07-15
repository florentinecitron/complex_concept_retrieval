import tqdm
import json
import itertools as it
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

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


def get_data():
    labels = []
    embeddings = []

    with open("data/codebook.json") as i:
        codebook_json = json.load(i)
    df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])
    df_codebook["top_level_title"] = df_codebook.title.map(lambda x: x.split(": ")[0])
    distinct_top_level_titles = df_codebook.top_level_title.unique()
    top_level_to_new_code = {t: str(i) for i, t in enumerate(distinct_top_level_titles)}
    code_to_top_level_code = {r.code: top_level_to_new_code[r.top_level_title] for r in df_codebook.itertuples()}
    df_codebook["top_level_code"] = df_codebook.code.map(code_to_top_level_code.get)

    with open("data/metadata.jsonl") as i:
        df_metadata = pd.DataFrame(it.chain.from_iterable([json.loads(line)["items"] for line in i]))
    df_metadata["key"] = df_metadata.party_id.map(str) + "_" + df_metadata.election_date.map(str)
    key_to_lang = {i.key: i.language for i in df_metadata.itertuples()}

    with open("data/texts_and_annotations.jsonl") as i:
        for line in tqdm.tqdm(i, total=233):
            data = json.loads(line)
            for outer_items in data["items"]:
                #_items = outer_items["items"]

                key = outer_items["key"]
                lang = key_to_lang[key]
                if lang != "english":
                    continue

                items = [i for i in outer_items["items"] if i["cmp_code"] not in ignore_categories]
                sentence_indices = [idx for idx, i in enumerate(outer_items["items"]) if i["cmp_code"] not in ignore_categories]
                labels.extend([code_to_top_level_code[i["cmp_code"]] for i in items])

                emb_file = f"data/text_encodings/{key}.pth"

                texts_encodings = torch.tensor(torch.load(emb_file))[sentence_indices, :]
                embeddings.append(texts_encodings)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

    df_codebook["top_level_title"] = df_codebook.title.map(lambda x: x.split(": ")[0])
    distinct_top_level_titles = df_codebook.top_level_title
    label_title_encodings = model.encode(distinct_top_level_titles, convert_to_tensor=True)
    top_level_code_to_embedding = dict(zip(df_codebook["top_level_code"], label_title_encodings))

    return torch.vstack(embeddings), labels, top_level_code_to_embedding