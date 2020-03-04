import os
import csv
import collections
import numpy as np
import statsmodels.stats.inter_rater
import pandas as pd
import json

def checkEqual(iterator):
   return len(set(iterator)) <= 1

class HumanClassification:
    def __init__(self, human_classifications_location,classs):
        self.dataset = {}
        self.classs = classs
        self.human_names = []
        for filename in os.listdir(human_classifications_location):
            human_name = filename.split(".")[0]
            self.human_names.append(human_name)
            with open(human_classifications_location + filename, encoding="utf-8-sig", newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")
                for row in reader:
                    idx = int(row["id"])
                    age = int(row["age"])
                    diagnostic = row["diagnostic"].strip().replace("\r","").replace("\n","")
                    classs = True if row[self.classs] == "True" else False
                    if idx not in self.dataset:
                        self.dataset[idx] = {}
                        self.dataset[idx]["diagnostic"] = diagnostic
                        self.dataset[idx]["age"] = age
                        self.dataset[idx][self.classs] = {}
                        self.dataset[idx][self.classs][human_name] = classs
                    else:
                        self.dataset[idx][self.classs][human_name] = classs
    def calculate_fleiss(self):
        self.matrix_data = []
        for point in self.dataset.values():
            classifications = point[self.classs].values()
            counts = collections.Counter(classifications)
            self.matrix_data.append((counts[True],counts[False]))
        self.matrix_data = np.array(self.matrix_data)
        self.fleiss = statsmodels.stats.inter_rater.fleiss_kappa(self.matrix_data)
    def extract_disagreements(self,disagreements_file_location):
        self.disagreements = {}
        for idx,data in self.dataset.items():
            classifications = data[self.classs].values()
            if not checkEqual(classifications):
                self.disagreements[idx] = {}
                self.disagreements[idx]["diagnostic"] = data["diagnostic"]
                self.disagreements[idx]["age"] = data["age"]
                for name,classification in data[self.classs].items():
                    self.disagreements[idx][name] = classification
        self.disagreements_df = pd.DataFrame.from_dict(self.disagreements, orient='index')
        self.disagreements_df.to_csv(disagreements_file_location, index_label="id")
    def calculate_venn(self):
        self.venn_data={name:[] for name in self.human_names}
        for idx,data in self.dataset.items():
            classifications = data[self.classs]
            for name,classification in classifications.items():
                if classification:
                    self.venn_data[name].append(idx)
                else:
                    self.venn_data[name].append(idx*-1)
    def write_report(self,report_location):
        self.report = {
            "fleiss" : self.fleiss,
            "dataset_n" : len(self.dataset),
            "agreements_n": len(self.dataset) - len(self.disagreements),
            "disagreements_n": len(self.disagreements),
            "venn_data": self.venn_data,
            "disagreements" : self.disagreements,
            "dataset" : self.dataset
        }
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False)

class GroundTruthGenerator:
    def __init__(self, human_dataset, super_dataset,classs,delimiter):
        self.classs = classs
        with open(human_dataset, encoding="utf-8") as json_file:
            self.dataset = json.load(json_file)["dataset"]
        with open(super_dataset, encoding="utf-8-sig", newline='') as csvfile:
            reader = csv.DictReader(csvfile,delimiter=delimiter)
            for row in reader:
                self.dataset[row["id"]][self.classs]["gt"] = True if row[self.classs] == "True" else False
    def write_ground_truth(self, ground_truth_file_location):
        self.ground_truth = {}
        for idx,data in self.dataset.items():
            self.ground_truth[idx] = {}
            self.ground_truth[idx]["diagnostic"] = data["diagnostic"]
            self.ground_truth[idx]["age"] = data["age"]
            if "gt" not in data[self.classs]:
                self.ground_truth[idx][self.classs] = list(data[self.classs].values())[0]
            else:
                self.ground_truth[idx][self.classs] = data[self.classs]["gt"]
            self.ground_truth_df = pd.DataFrame.from_dict(self.ground_truth, orient='index')
            self.ground_truth_df.to_csv(ground_truth_file_location, index_label="id")