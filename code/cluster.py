import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


def run():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

    with open("data/codebook.json") as i:
        codebook_json = json.load(i)
    df_codebook = pd.DataFrame(codebook_json[1:], columns=codebook_json[0])
    df_codebook["top_level_title"] = df_codebook.title.map(lambda x: x.split(": ")[0])
    distinct_top_level_titles = df_codebook.top_level_title.unique()
    top_level_to_new_code = {t: str(i) for i, t in enumerate(distinct_top_level_titles)}
    code_to_top_level_code = {r.code: top_level_to_new_code[r.top_level_title] for r in df_codebook.itertuples()}

    label_title_encodings = model.encode(distinct_top_level_titles, convert_to_tensor=True)

    reduction_model = TSNE()
    reduced = reduction_model.fit_transform(label_title_encodings)

    fig, ax = plt.subplots()
    ax.scatter(*reduced.T)

    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(distinct_top_level_titles[sel.target.index]))


    plt.show()

    # Not good: too much focus on pure lexical overlap
