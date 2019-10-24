import json
import csv
import collections
import scipy.stats
import numpy as np

class Descriptor:
    def __init__(self, dataset_location, data_class, label_column = None, age_column = 'edad_raw', label_column = None):
        self.dataset_location = dataset_location
        self. data_class = data_class
        self.label_column = label_column
        self.age_column = age_column
        self.label_column = label_column
    def analyze(self):
        if self.data_class == 'corpus':
            self.lines_length = []
            self.counter = collections.Counter()
            with open(self.dataset_location, encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip().split(' ')
                    self.lines_length.append(len(line))
                    self.counter.update(line)
            self.normal_distribution = scipy.stats.shapiro(self.lines_length)[1] > 0.05
            self.mean = np.mean(self.lines_length)
            self.median = np.median(self.lines_length)
            self.sd = np.std(self.lines_length)
            self.length = len(self.lines_length)
            self.n_tokens = np.sum(self.lines_length)
            self.n_vocab = len(self.counter.keys())
        if self.data_class == 'table':
            self.entry_dates = []
            self.labels = []
            self.ages = []