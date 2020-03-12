import sklearn.naive_bayes
import numpy as np
import joblib
import scipy
import json
import nltk

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

class NbModelTrainer:
    def __init__(self, train_texts, train_labels):
        self.train_texts = scipy.sparse.load_npz(train_texts)
        self.train_labels = []
        with open(train_labels, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                if line == 'True':
                    self.train_labels.append(True)
                else:
                    self.train_labels.append(False)
        self.train_labels = np.asarray([self.train_labels]).T

    def train_model(self,test_texts, test_labels,results_location, serialized_model_location, coefs_location, vocab_location):
        self.test_texts = scipy.sparse.load_npz(test_texts)
        self.test_labels = []
        with open(test_labels, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                if line == 'True':
                    self.test_labels.append(True)
                else:
                    self.test_labels.append(False)
        self.test_labels = np.asarray([self.test_labels]).T
        
        estimator = sklearn.naive_bayes.MultinomialNB()
        estimator.fit(self.train_texts,self.train_labels)
        joblib.dump(estimator, serialized_model_location)
        predictions_class = estimator.predict(self.test_texts)
        predictions_probs = estimator.predict_proba(self.test_texts)
        self.best_results = np.column_stack([self.test_labels,predictions_class,predictions_probs])
        np.savetxt(results_location,self.best_results)
        with open(vocab_location, 'r', encoding='utf-8') as json_file:
            vocab = json.load(json_file)
        importance = {}
        sw = nltk.corpus.stopwords.words("spanish")
        for word,idx in vocab.items():
            if word not in sw:
                importance[word] = estimator.coef_[0][idx]
        importance = {k: v for k, v in sorted(importance.items(), reverse = True, key=lambda item: item[1])}
        with open(coefs_location, 'w', encoding='utf-8') as json_file:
            json.dump(importance, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)