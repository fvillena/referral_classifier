import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import numpy as np
import json

models = [
    (
        sklearn.linear_model.LogisticRegression(),
        {
            "C":np.logspace(-5,5,11),
            "penalty":["l1","l2"]
        }
    ),
    # sklearn.svm.SVC(),
    # sklearn.ensemble.RandomForestClassifier(),
    # sklearn.neural_network.MLPClassifier()
]

np.random.seed(11)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    def train_models(self,n_jobs=4):
        self.scores = {}
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            grid = model[1]
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            grid_search = sklearn.model_selection.RandomizedSearchCV(
                estimator=estimator,
                param_distributions=grid,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=2,
                random_state=11,
                return_train_score=True,
                cv=3
            )
            grid_search.fit(features,labels)
            self.scores[model_name] = [grid_search.cv_results_,grid_search.best_params_,grid_search.best_score_]
    def generate_report(self,report_location):
        for key,val in self.scores.items():
            with open(report_location + key, 'w', encoding='utf-8') as json_file:
                json.dump(val, json_file, indent=2, ensure_ascii=False, cls=NumpyEncoder)

