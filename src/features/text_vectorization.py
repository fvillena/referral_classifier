import gensim # module for computing word embeddings
import numpy as np # linear algebra module
import sklearn.feature_extraction.text # package to perform tf-idf vertorization
import json
import scipy

class NpEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class BowVectorizer:
    def __init__(self,train_texts_location,test_texts_location):
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
    def vectorize_text(self):
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        self.vectorizer.fit(self.train_sentences + self.test_sentences)
        self.train_matrix = self.vectorizer.transform(self.train_sentences)
        self.test_matrix = self.vectorizer.transform(self.test_sentences)
    def write_data(self,prefix,vectorized_text_location):
        scipy.sparse.save_npz(vectorized_text_location + prefix + '_train_text.npz', self.train_matrix)
        scipy.sparse.save_npz(vectorized_text_location + prefix + '_test_text.npz', self.test_matrix)
        with open(vectorized_text_location + prefix + '_vocab.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.vectorizer.vocabulary_, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)