import sklearn.naive_bayes
import numpy as np
import joblib
import scipy

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

    def train_model(self,test_texts, test_labels,results_location, serialized_model_location):
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