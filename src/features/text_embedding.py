import gensim # module for computing word embeddings
import numpy as np # linear algebra module
import sklearn.feature_extraction # package to perform tf-idf vertorization
import json

def to_vector(texto,model,idf):
    """ Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence"""
    tokens = texto.split() # splits the text by space and returns a list of words
    vec = np.zeros(300) # creates an empty vector of 300 dimensions
    for word in tokens: # iterates over the sentence
        if (word in model) & (word in idf): # checks if the word is both in the word embedding and the tf-idf model
            vec += model[word]*idf[word] # adds every word embedding to the vector
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec) # divides the vector by their normal
    else:
        return vec

class TextVectorizer:
    def __init__(self,embeddings_location,train_texts_location,test_texts_location):
        self.embedding = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_location,
            binary=False 
        )
        self.train_sentences = []
        with open(train_texts_location, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                self.train_sentences.append(line)
        self.test_sentences = []
        with open(test_texts_location, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                self.test_sentences.append(line) 
    def vectorize_text(self,idf_location):
        tfidfvectorizer = sklearn.feature_extraction.text.TfidfVectorizer() # instance of the tf-idf vectorizer
        tfidfvectorizer.fit(self.train_sentences) # fitting the vectorizer and transforming the properties
        self.idf = {key:val for key,val in zip(tfidfvectorizer.get_feature_names(),tfidfvectorizer.idf_)}
        with open(idf_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.idf, json_file, indent=2, ensure_ascii=False)
        self.train_matrix = np.zeros( # creatign an empty matrix
            (
                len(self.train_sentences), # the number of rows is equal to the number of data points
                len(self.embedding['paciente']) # the number of columns is equal to the number of components of the word embedding
            )
        )
        self.test_matrix = np.zeros( # creatign an empty matrix
            (
                len(self.test_sentences), # the number of rows is equal to the number of data points
                len(self.embedding['paciente']) # the number of columns is equal to the number of components of the word embedding
            )
        )
        for i,sentence in enumerate(self.train_sentences):
            vector = to_vector(sentence,self.embedding,self.idf)
            self.train_matrix[i,] = vector
        for i,sentence in enumerate(self.test_sentences):
            vector = to_vector(sentence,self.embedding,self.idf)
            self.test_matrix[i,] = vector
    def write_data(self,prefix,vectorized_text_location):
        np.save(vectorized_text_location + prefix + '_train_text.npy', self.train_matrix)
        np.save(vectorized_text_location + prefix + '_test_text.npy', self.test_matrix)