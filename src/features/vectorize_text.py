import gensim # module for computing word embeddings
import numpy as np # linear algebra module
import sklearn.feature_extraction # package to perform tf-idf vertorization

def to_vector(texto,model):
    """ Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence"""
    tokens = texto.split() # splits the text by space and returns a list of words
    vec = np.zeros(300) # creates an empty vector of 300 dimensions
    for word in tokens: # iterates over the sentence
        if (word in model) & (word in idf): # checks if the word is both in the word embedding and the tf-idf model
            vec += model[word]*idf[word] # adds every word embedding to the vector
    return vec / np.linalg.norm(vec) # divides the vector by their normal

embedding_location = r'../../models/embeddings/waiting_list_corpus.vec'
interim_train_text_location = r'../../data/interim/train_text.txt'
interim_test_text_location = r'../../data/interim/test_text.txt'
processed_train_text_location = r'../../data/processed/train_text.npy'
processed_test_text_location = r'../../data/processed/test_text.npy'

embedding = gensim.models.KeyedVectors.load_word2vec_format( # loading word embeddings
    embedding_location, # using the spanish billion words embeddings
    binary=False # the model is in binary format
)
train_sentences = []
with open(interim_train_text_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        train_sentences.append(line)
test_sentences = []
with open(interim_test_text_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        test_sentences.append(line)
tfidfvectorizer = sklearn.feature_extraction.text.TfidfVectorizer() # instance of the tf-idf vectorizer
tfidfvectorizer.fit(train_sentences) # fitting the vectorizer and transforming the properties
idf = {key:val for key,val in zip(tfidfvectorizer.get_feature_names(),tfidfvectorizer.idf_)}
train_matrix = np.zeros( # creatign an empty matrix
    (
        len(train_sentences), # the number of rows is equal to the number of data points
        len(embedding['paciente']) # the number of columns is equal to the number of components of the word embedding
    )
)
test_matrix = np.zeros( # creatign an empty matrix
    (
        len(test_sentences), # the number of rows is equal to the number of data points
        len(embedding['paciente']) # the number of columns is equal to the number of components of the word embedding
    )
)
for i,sentence in enumerate(train_sentences):
    vector = to_vector(sentence,embedding)
    train_matrix[i,] = vector
for i,sentence in enumerate(test_sentences):
    vector = to_vector(sentence,embedding)
    test_matrix[i,] = vector
print(train_matrix.shape)
print(test_matrix.shape)
np.save(processed_train_text_location, train_matrix)
np.save(processed_test_text_location, test_matrix)