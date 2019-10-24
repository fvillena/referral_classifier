import json
import csv
import collections
import scipy.stats
import numpy as np

class Descriptor:
    def __init__(self, dataset_location, data_class, entry_column = None, age_column = 'edad_raw', label_column = None):
        self.dataset_location = dataset_location
        self. data_class = data_class
        self.entry_column = entry_column
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
            with open(self.dataset_location, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                self.length = 0
                for row in reader:
                    self.entry_dates.append(row[self.entry_column])
                    self.labels.append(row[self.label_column] == 'True')
                    self.ages.append(float(row[self.age_column]))
                    self.length += 1
            self.entry_dates = np.array(self.entry_dates, 'datetime64[D]')
            self.entry_dates.sort()
            self.entry_dates_span = [
                self.entry_dates[int(self.entry_dates.size * 0.0001)],
                self.entry_dates[int(self.entry_dates.size * (1-0.0001))]
            ]
            self.ages_normal_distribution = scipy.stats.shapiro(self.ages)[1] > 0.05
            self.ages_mean = np.mean(self.ages)
            self.ages_median = np.median(self.ages)
            self.ages_sd = np.std(self.ages)
            self.ages = np.array(self.ages)
            self.ages.sort()
            self.ages_interval = [
                self.ages[int(self.ages.size * 0.0001)],
                self.ages[int(self.ages.size * (1-0.0001))]
            ]
            self.labels_ratio = {
                'True' : sum(self.labels),
                'False' : abs(sum(self.labels) - len(self.labels))
            }

