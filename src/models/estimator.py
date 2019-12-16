import joblib
import gensim
import numpy as np
import json
import re

def normalizer(text, remove_tildes = True): #normalizes a given string to lowercase and changes all vowels to their base form
    text = text.lower() #string lowering
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text) #replaces every punctuation with a space
    if remove_tildes:
        text = re.sub('á', 'a', text) #replaces special vowels to their base forms
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
    return text

def to_vector(texto,model,idf):
    """ Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence"""
    tokens = normalizer(texto).split() # splits the text by space and returns a list of words
    vec = np.zeros(300) # creates an empty vector of 300 dimensions
    for word in tokens: # iterates over the sentence
        if (word in model) & (word in idf): # checks if the word is both in the word embedding and the tf-idf model
            vec += model[word]*idf[word] # adds every word embedding to the vector
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec) # divides the vector by their normal
    else:
        return vec

class GesEstimator:
    def __init__(self,model,scaler,embedding,idf):
        self.model = joblib.load(model)
        self.scaler = joblib.load(scaler)
        self.embedding = self.embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding, binary=False )
        with open(idf, encoding="utf-8") as json_file:
            self.idf = json.load(json_file)
    def predict(self,diagnostic,age):
        age = self.scaler.transform(np.array([age]).reshape(-1,1))[0]
        vector = np.array(to_vector(diagnostic,self.embedding,self.idf))
        features = np.concatenate((vector,age)).reshape(1, -1)
        result = self.model.predict(features)
        return result