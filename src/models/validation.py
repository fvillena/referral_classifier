import scipy.stats
import json
import os
import itertools
import statsmodels.stats.multitest

class StatisticalAnalysis:
    def __init__(self,results_location):
        self.scores = {}
        for filename in os.listdir(results_location):
            estimator_name = filename.split('.')[0]
            self.scores[estimator_name] = {}
            with open(results_location + filename, "r") as read_file:
                data = json.load(read_file)
                self.scores[estimator_name] = data['test_roc_auc']
    def analyze(self,report_location):
        self.combinations = list(itertools.combinations(self.scores.items(),2))
        self.p_values = []
        for combination in self.combinations:
            p = scipy.stats.ttest_rel(combination[0][1],combination[1][1]).pvalue
            self.p_values.append(p)
        self.p_values_corrected = statsmodels.stats.multitest.multipletests(self.p_values,method='bonferroni',returnsorted=False)[1]
        
