import gensim # module for computing word embeddings
import numpy as np # linear algebra module
import sklearn.feature_extraction # package to perform tf-idf vertorization
np.random.seed(11)
def to_vector_random(texto):
    tokens = texto.split() # splits the text by space and returns a list of words
    vec = np.zeros(300) # creates an empty vector of 300 dimensions
    for word in tokens: # iterates over the sentence
        if word in embedding: # checks if the word is both in the word embedding and the tf-idf model
            vec += embedding[word] # adds every word embedding to the vector
        else:
            embedding[word] = np.random.rand(300)
            vec += embedding[word]
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec) # divides the vector by their normal
    else:
        return vec

embedding_location = r'../../models/embeddings/random.vec'
interim_texts_location = r'../../data/interim/texts_unique.txt'
processed_texts_location = r'../../data/processed/texts_unique__averaging__random.npy'

embedding = {}

sentences = []
with open(interim_texts_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        sentences.append(line)

matrix = np.zeros( # creatign an empty matrix
    (
        len(sentences), # the number of rows is equal to the number of data points
        300 # the number of columns is equal to the number of components of the word embedding
    )
)

for i,sentence in enumerate(sentences):
    vector = to_vector_random(sentence)
    matrix[i,] = vector
print(matrix.shape)
np.save(processed_texts_location, matrix)