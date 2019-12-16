import scipy.stats
import numpy as np
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "."))
import itertools
import statsmodels.stats.multitest
import sklearn.metrics
from estimator import GesEstimator
import pandas as pd

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

class StatisticalAnalysis:
    def __init__(self,results_location):
        self.scores = {}
        for filename in os.listdir(results_location):
            estimator_name = filename.split('.')[0]
            self.scores[estimator_name] = {}
            with open(results_location + filename, "r") as read_file:
                data = json.load(read_file)
                self.scores[estimator_name] = data['test_roc_auc']
    def analyze(self):
        self.combinations = list(itertools.combinations(self.scores.items(),2))
        self.summary = {}
        for model in self.scores.keys():
            self.summary[model] = {
                'scores':self.scores[model],
                'normally_distributed':scipy.stats.shapiro(self.scores[model])[1] > 0.05,
                'mean_scores':np.mean(self.scores[model]),
                'standard_deviation_scores': np.std(self.scores[model]),
                'confidence_interval_scores':scipy.stats.norm.interval(0.95,loc=np.mean(self.scores[model]),scale=np.std(self.scores[model]))
            }
        self.p_values = []
        for combination in self.combinations:
            normal_distribution = self.summary[combination[0][0]]['normally_distributed'] & self.summary[combination[1][0]]['normally_distributed']
            if normal_distribution:
                p = scipy.stats.ttest_rel(combination[0][1],combination[1][1]).pvalue
            else:
                p = scipy.stats.wilcoxon(combination[0][1],combination[1][1]).pvalue
            self.p_values.append(p)
        self.p_values_corrected = statsmodels.stats.multitest.multipletests(self.p_values,method='bonferroni',returnsorted=False)[1]
    def generate_report(self,report_location):
        self.report = {
            'summary':self.summary,
            'statistical_report': {
                'combinations' : self.combinations,
                'p_values': self.p_values,
                'p_values_corrected': self.p_values_corrected
            }
        }
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)
class Performance:
    def __init__(self,results_file):
        results = np.loadtxt(results_file)
        self.true = np.array(results[:,0],dtype=bool)
        self.predicted_class = np.array(results[:,1],dtype=bool)
        self.predicted_proba = results[:,3]
    def analyze(self):
        self.classification_report = sklearn.metrics.classification_report(self.true,self.predicted_class,output_dict=True)
        self.roc_curve = sklearn.metrics.roc_curve(self.true,self.predicted_proba)
        self.roc_auc_score = sklearn.metrics.roc_auc_score(self.true,self.predicted_proba)
    def generate_report(self,report_location):
        self.report = {
            'classification_report': self.classification_report,
            'roc_auc_score': self.roc_auc_score,
            'roc_curve': self.roc_curve
        }
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)

class GroundTruthPredictor:
    def __init__(self,model,scaler,embedding,idf,ground_truth_location):
        self.estimator = GesEstimator(model,scaler,embedding,idf)
        self.ground_truth = pd.read_csv(ground_truth_location)
        del self.ground_truth["ges"]
    def predict(self,predictions_location):
        self.ground_truth["ges"] = self.ground_truth.apply(lambda x: self.estimator.predict(x["diagnostic"], x['age']), axis=1)
        self.ground_truth.to_csv(predictions_location, index=False)