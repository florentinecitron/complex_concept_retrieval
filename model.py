from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

queries = ["digitalization", "nationalism", "vaccination policy"]
sentences = [] # from download.py

embeddings1 = model.encode(queries, convert_to_tensor=True)
embeddings2 = model.encode(sentences, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

for query, sentence_index in zip(queries, cosine_scores.max(axis=1).indices):
    print(query, "|", sentences[sentence_index])