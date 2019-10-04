import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import numpy as np
import json

models = [
    sklearn.linear_model.LogisticRegression(solver='lbfgs'),
    sklearn.svm.SVC(gamma='scale'),
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
        self.scores = {}
        for model in self.models:
            clf = model
            clf_name = clf.__class__.__name__
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            self.scores[clf_name] = sklearn.model_selection.cross_validate(clf,features,labels,n_jobs=4,scoring=['f1_weighted','recall_weighted'],verbose=2,cv=10)
    def generate_report(self,report_location):
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.scores, json_file, indent=2, ensure_ascii=False)

