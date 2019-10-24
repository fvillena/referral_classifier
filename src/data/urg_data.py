import pandas as pd
import numpy as np
import datetime as dt
import nltk
import sklearn.preprocessing
import sklearn.model_selection
import re

def normalizer(text, remove_tildes = True): #normalizes a given string to lowercase and changes all vowels to their base form
    text = text.lower() #string lowering
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text) #replaces every punctuation with a space
    if remove_tildes:
        text = re.sub('á', 'a', text) #replaces special vowels to their base forms
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
    return text

class UrgDatasetGenerator:
    def __init__(self, dataset_location):
        self.data = pd.read_csv(dataset_location, 
                   low_memory=False, 
                   error_bad_lines=False,
                   sep=";")
    def preprocess(self):
        self.data=self.data.rename(columns = {'REGLA 16+17':'urg'})
        self.data.F_ENTRADA = pd.to_datetime(self.data.F_ENTRADA)
        self.data=self.data[["urg","SOSPECHA_DIAG","F_ENTRADA"]]
        self.data['SOSPECHA_DIAG'] = self.data.SOSPECHA_DIAG.astype(str)
        self.data['SOSPECHA_DIAG'] = self.data.SOSPECHA_DIAG.apply(normalizer)
        self.data = self.data.dropna(subset=["urg","SOSPECHA_DIAG"])
        self.data['SOSPECHA_DIAG'] = self.data.SOSPECHA_DIAG.apply(nltk.tokenize.word_tokenize)
        self.data = self.data.dropna(subset=["urg","SOSPECHA_DIAG"])
        self.data['urg'] = np.where(self.data['urg'] == 1, True, False)
        self.data = self.data[self.data.SOSPECHA_DIAG.str.len() > 1]
    def split(self, test_size = 0.4):
        train_features, test_features, train_labels, self.test_labels = sklearn.model_selection.train_test_split(self.data['SOSPECHA_DIAG'],self.data['urg'], test_size = test_size, random_state = 11)
        
        train = pd.concat([train_features,train_labels], axis=1)
        count_class_0, _ = train['urg'].value_counts()
        df_class_0 = train[train['urg'] == False]
        df_class_1 = train[train['urg'] == True]
        df_class_1_up = df_class_1.sample(count_class_0, replace=True, random_state=11)
        train_up = pd.concat([df_class_1_up, df_class_0], axis=0)
        train_features = train_up['SOSPECHA_DIAG']
        self.train_labels = train_up['urg']

        train_text = train_features.tolist()
        self.train_text = [' '.join(sentence) for sentence in train_text]
        test_text = test_features.tolist()
        self.test_text = [' '.join(sentence) for sentence in test_text]
    def write_dataset(self, processed_dataset_location, interim_dataset_location):
        self.train_labels.to_csv(processed_dataset_location + 'urg_train_labels.txt', index=False)
        with open(interim_dataset_location + 'urg_train_text.txt', 'w', encoding='utf-8') as file:
            for sentence in self.train_text:
                file.write(sentence)
                file.write('\n')
        self.test_labels.to_csv(processed_dataset_location + 'urg_test_labels.txt', index=False)
        with open(interim_dataset_location + 'urg_test_text.txt', 'w', encoding='utf-8') as file:
            for sentence in self.test_text:
                file.write(sentence)
                file.write('\n')
    

