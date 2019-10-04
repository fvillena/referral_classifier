import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import numpy as np
import json

models = [
    sklearn.linear_model.LogisticRegression(),
    sklearn.svm.SVC(),
    sklearn.ensemble.RandomForestClassifier(),
    sklearn.neural_network.MLPClassifier()
]

np.random.seed(11)

class GesModelTrainer:
    def __init__(self, train_texts, train_ages, train_labels, models = models):
        self.models = models
        self.train_texts = np.load(train_texts)
        self.train_ages = []
        with open(train_ages, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                self.train_ages.append(float(line))
        self.train_ages = np.asarray([self.train_ages]).T
        self.train_labels = []
        with open(train_labels, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                if line == 'True':
                    self.train_labels.append(True)
                else:
                    self.train_labels.append(False)
        self.train_labels = np.asarray([self.train_labels]).T
        self.train = np.concatenate([self.train_texts, self.train_ages, self.train_labels], axis=1)
    def train_models(self):
        self.predictions = {}
        cv = sklearn.model_selection.KFold(n_splits=10,random_state=11)
        clf = self.models[0]
        clf_name = clf.__class__.__name__
        self.predictions[clf_name] = []
        features = self.train[:,:-1]
        labels = self.train[:,-1]
        for train_index, test_index in cv.split(features):
            clf.fit(features[train_index], labels[train_index])
            predicted = clf.predict(features[test_index])
            self.predictions[clf_name].append((labels,predicted))


