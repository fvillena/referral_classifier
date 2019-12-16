import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import numpy as np
import json
import joblib

models = [
    (
        sklearn.linear_model.LogisticRegression(),
        {
            "C":np.logspace(-5,5,11),
            "penalty":["l1","l2"]
        }
    ),
    (
        sklearn.svm.SVC(),
        {
            'C':[1,10,100,1000],
            'gamma':[1,0.1,0.001,0.0001], 
            'kernel':['linear','rbf']
        }
    ),
    (
        sklearn.ensemble.RandomForestClassifier(),
        {
            'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        }
    ),
    (
        sklearn.neural_network.MLPClassifier(),
        {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
    )
]

best_estimator = sklearn.ensemble.RandomForestClassifier()
best_hp = {
    "n_estimators": 1000,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": "auto",
    "max_depth": 70,
    "bootstrap": True
  }


np.random.seed(11)

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

class UrgModelTrainer:
    def __init__(self, train_texts, train_labels, models = models):
        self.models = models
        self.train_texts = np.load(train_texts)
        self.train_labels = []
        with open(train_labels, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                if line == 'True':
                    self.train_labels.append(True)
                else:
                    self.train_labels.append(False)
        self.train_labels = np.asarray([self.train_labels]).T
        self.train = np.concatenate([self.train_texts, self.train_labels], axis=1)
    def grid_search(self,n_jobs=4):
        self.gs_scores = {}
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
            self.gs_scores[model_name] = [grid_search.cv_results_,grid_search.best_params_,grid_search.best_score_]
    def train_models(self, grid_search_results_location,n_jobs):
        self.cv_scores = {}
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            with open(grid_search_results_location + model_name + '.json', "r") as read_file:
                data = json.load(read_file)
            best_hp=data[1]
            estimator.set_params(**best_hp)
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            cv_scores = sklearn.model_selection.cross_validate(
                estimator=estimator,
                X=features,
                y=labels,
                cv=10,
                n_jobs=n_jobs,
                scoring=['accuracy','f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc'],
                verbose=2,
                return_train_score=True
            )
            self.cv_scores[model_name] = cv_scores
    def train_best_model(self,test_texts, test_labels,results_location, serialized_model_location,best_estimator=best_estimator,best_hp=best_hp,n_jobs=-1):
        self.test_texts = np.load(test_texts)
        self.test_labels = []
        with open(test_labels, encoding='utf-8') as file:
            for line in file:
                line = line.rstrip()
                if line == 'True':
                    self.test_labels.append(True)
                else:
                    self.test_labels.append(False)
        self.test_labels = np.asarray([self.test_labels]).T
        self.test = np.concatenate([self.test_texts, self.test_labels], axis=1)
        features_train = self.train[:,:-1]
        labels_train = self.train[:,-1]
        features_test = self.test[:,:-1]
        labels_test = self.test[:,-1]
        estimator = best_estimator
        estimator.set_params(**best_hp,n_jobs=n_jobs,verbose=2)
        estimator.fit(features_train,labels_train)
        joblib.dump(estimator, serialized_model_location)
        predictions_class = estimator.predict(features_test)
        predictions_probs = estimator.predict_proba(features_test)
        self.best_results = np.column_stack([labels_test,predictions_class,predictions_probs])
        np.savetxt(results_location,self.best_results)
    def generate_report(self,report_location):
        for key,val in self.gs_scores.items():
            with open(report_location + 'urg_grid_search/' + key + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(val, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)
        for key,val in self.cv_scores.items():
            with open(report_location + 'urg_cross_val/' + key + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(val, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)

