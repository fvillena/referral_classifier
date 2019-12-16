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
    def __init__(self, human_classifications_location):
        self.dataset = {}
        for filename in os.listdir(human_classifications_location):
            human_name = filename.split(".")[0]
            with open(human_classifications_location + filename, encoding="utf-8-sig", newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")
                for row in reader:
                    idx = int(row["id"])
                    age = int(row["age"])
                    diagnostic = row["diagnostic"].strip().replace("\r","").replace("\n","")
                    ges = True if row["ges"] == "True" else False
                    if idx not in self.dataset:
                        self.dataset[idx] = {}
                        self.dataset[idx]["diagnostic"] = diagnostic
                        self.dataset[idx]["age"] = age
                        self.dataset[idx]["ges"] = {}
                        self.dataset[idx]["ges"][human_name] = ges
                    else:
                        self.dataset[idx]["ges"][human_name] = ges
    def calculate_fleiss(self):
        self.matrix_data = []
        for point in self.dataset.values():
            classifications = point["ges"].values()
            counts = collections.Counter(classifications)
            self.matrix_data.append((counts[True],counts[False]))
        self.matrix_data = np.array(self.matrix_data)
        self.fleiss = statsmodels.stats.inter_rater.fleiss_kappa(self.matrix_data)
    def extract_disagreements(self,disagreements_file_location):
        self.disagreements = {}
        for idx,data in self.dataset.items():
            classifications = data["ges"].values()
            if not checkEqual(classifications):
                self.disagreements[idx] = {}
                self.disagreements[idx]["diagnostic"] = data["diagnostic"]
                self.disagreements[idx]["age"] = data["age"]
        self.disagreements_df = pd.DataFrame.from_dict(self.disagreements, orient='index')
        self.disagreements_df.to_csv(disagreements_file_location, index_label="id")
    def write_report(self,report_location):
        self.report = {
            "fleiss" : self.fleiss,
            "disagreements" : self.disagreements,
            "dataset" : self.dataset
        }
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False)

