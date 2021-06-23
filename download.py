import json

import more_itertools as mit
import pandas as pd
import requests
import tqdm


base_url = "https://manifesto-project.wzb.eu/api/v1/"
api_key = ...
metadata_versions_url = base_url + "list_metadata_versions"
codebook_url = base_url + "get_core_codebook"
core_url = base_url + "get_core"
metadata_url = base_url + "metadata"
texts_and_annotations_url = base_url + "texts_and_annotations"


metadata_versions_req = requests.get(metadata_versions_url, params={"api_key": api_key})
metadata_versions_json = metadata_versions_req.json()

codebook_req = requests.get(codebook_url, params={"api_key": api_key, "key": "MPDS2020b"})
codebook_json = codebook_req.json()
df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])

core_req = requests.get(core_url, params={"api_key": api_key, "key": "MPDS2020b"})
core_json = core_req.json()
df_core = pd.DataFrame(core_json[1:], columns=core_json[0])

ser_core_metadata_key = df_core.party + "_" + df_core.date

with open("data/codebook.json", "w") as o:
    print(json.dumps(codebook_json), file=o)
with open("data/core.json", "w") as o:
    print(json.dumps(core_json), file=o)

with open("data/metadata.jsonl", "w") as metadata_file, open("data/texts_and_annotations.jsonl", "w") as texts_and_annotations_file:

    for keys in mit.chunked(tqdm.tqdm(ser_core_metadata_key), 20):
        params = [("keys[]", k) for k in keys] + [("api_key", api_key), ("version", "2020-2")]

        metadata_req = requests.get(metadata_url, params=params)
        metadata_json = metadata_req.json()

        texts_and_annotations_req = requests.get(texts_and_annotations_url, params=params)
        texts_and_annotations_json = texts_and_annotations_req.json()

        print(json.dumps(metadata_json), file=metadata_file)
        print(json.dumps(texts_and_annotations_json), file=texts_and_annotations_file)
