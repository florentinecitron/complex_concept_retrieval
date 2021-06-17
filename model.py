from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

queries = ["digitalization", "nationalism", "vaccination policy"]
sentences = [] # from download.py

embeddings1 = model.encode(queries, convert_to_tensor=True)
embeddings2 = model.encode(sentences, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

for query, sentence_index in zip(queries, cosine_scores.max(axis=1).indices):
    print(query, "|", sentences[sentence_index])

# digitalization | Rapidly expand access to and use of ICT infrastructure
# nationalism | National symbols can contribute to nation-building and the establishment of a common national identity.
# vaccination policy | To reduce the incidence of cervical cancer of the uterus in women, as from 2014 all girl-children in Grade 4 in government schools will be vaccinated against the human papilloma virus which causes cervical cancer.