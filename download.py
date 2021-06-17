

# https://manifesto-project.wzb.eu/api/v1/list_core_versions?api_key=9846dc71b6d336f4fd124f42de267ec5

# https://manifesto-project.wzb.eu/api/v1/texts_and_annotations?api_key=9846dc71b6d336f4fd124f42de267ec5&keys[]=41320_2009&version=2015-3


#
#
import requests
import pandas as pd
import more_itertools as mit
import urllib.parse

base_url = "https://manifesto-project.wzb.eu/api/v1/"
api_key = "***"
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

# for example, should iterate chunked over ser_core_metadata_key
for keys in [ser_core_metadata_key.tail().to_list()]:
    params = [("keys[]", k) for k in keys] + [("api_key", api_key), ("version", "2020-2")]

    metadata_req = requests.get(metadata_url, params=params)
    metadata_json = metadata_req.json()
    # do something with metadata

    texts_and_annotations_req = requests.get(texts_and_annotations_url, params=params)
    texts_and_annotations_json = texts_and_annotations_req.json()

    # do something with text and annotations data
    sentences = []
    for item in texts_and_annotations_json["items"]:
        for jtem in item["items"]:
            sentences.append(jtem["text"])