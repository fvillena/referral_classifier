import numpy as np

corpus_files = ["data/interim/test_text.txt", "data/interim/train_text.txt"]
embedding_location = "models/random.vec"
dim = 300

tokens = []

for filename in corpus_files:
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            tokens += line.split(" ")

vocab = set(tokens)

embedding = {}

for token in vocab:
    embedding[token] = np.random.rand(300)

with open(embedding_location, "w", encoding="utf-8") as vec:
    vec.write("{} {}\n".format(len(vocab),dim))
    for word,vector in embedding.items():
        vec.write("{} {}\n".format(word," ".join([str(round(float(e),6)) for e in vector])))