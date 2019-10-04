import pandas as pd
import numpy as np
import seaborn as sns
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

class GesDatasetGenerator:
    def __init__(self, dataset_location):
        self.data = pd.read_csv(dataset_location, 
                   low_memory=False, 
                   error_bad_lines=False,
                   sep=",")
    def preprocess(self):
        self.data.INGRESO = pd.to_datetime(self.data.INGRESO, errors="coerce")
        self.data.FECHANACIMIENTO = pd.to_datetime(self.data.FECHANACIMIENTO, errors="coerce", format="%d/%m/%y")
        from datetime import timedelta, date
        future = self.data['FECHANACIMIENTO'] > '2018-01-01'
        self.data.loc[future, 'FECHANACIMIENTO'] -= dt.timedelta(days=365.25*100)
        self.data["edad"] = (self.data.INGRESO-self.data.FECHANACIMIENTO).astype('timedelta64[Y]')
        self.data=self.data[["GES","SOSPECHA_DIAGNOSTICA","edad"]]
        self.data['SOSPECHA_DIAGNOSTICA'] = self.data.SOSPECHA_DIAGNOSTICA.astype(str)
        self.data['SOSPECHA_DIAGNOSTICA'] = self.data.SOSPECHA_DIAGNOSTICA.apply(normalizer)
        self.data = self.data.dropna()
        self.data['SOSPECHA_DIAGNOSTICA'] = self.data.SOSPECHA_DIAGNOSTICA.apply(nltk.tokenize.word_tokenize)
        self.data["edad"] = np.where(self.data.edad < 0, np.nan, self.data.edad)
        self.data = self.data.dropna()
        scaler = sklearn.preprocessing.MinMaxScaler()
        self.data["edad"] = scaler.fit_transform(self.data[["edad"]])
        self.data['GES'] = np.where(self.data['GES'] == 'SI', True, False)
        self.data = self.data[self.data.SOSPECHA_DIAGNOSTICA.str.len() > 1]
    def split(self, test_size = 0.1):
        train_features, test_features, train_labels, self.test_labels = sklearn.model_selection.train_test_split(self.data[['SOSPECHA_DIAGNOSTICA','edad']],self.data['GES'], test_size = test_size, random_state = 11)
        train = pd.concat([train_features,train_labels], axis=1)
        _, count_class_1 = train['GES'].value_counts()
        df_class_0 = train[train['GES'] == False]
        df_class_1 = train[train['GES'] == True]
        df_class_0_under = df_class_0.sample(count_class_1)
        train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        train_features = train_under[['SOSPECHA_DIAGNOSTICA','edad']]
        self.train_labels = train_under['GES']
        self.train_age = train_features['edad']
        train_text = train_features['SOSPECHA_DIAGNOSTICA'].tolist()
        self.train_text = [' '.join(sentence) for sentence in train_text]
        self.test_age = test_features['edad']
        test_text = test_features['SOSPECHA_DIAGNOSTICA'].tolist()
        self.test_text = [' '.join(sentence) for sentence in test_text]
    def write_dataset(self, processed_dataset_location, interim_dataset_location):
        self.train_age.to_csv(processed_dataset_location + 'train_age.txt', index=False)
        self.train_labels.to_csv(processed_dataset_location + 'train_labels.txt', index=False)
        with open(interim_dataset_location + 'train_text.txt', 'w', encoding='utf-8') as file:
            for sentence in self.train_text:
                file.write(sentence)
                file.write('\n')
        self.test_age.to_csv(processed_dataset_location + 'test_age.txt', index=False)
        self.test_labels.to_csv(processed_dataset_location + 'test_labels.txt', index=False)
        with open(interim_dataset_location + 'test_text.txt', 'w', encoding='utf-8') as file:
            for sentence in self.test_text:
                file.write(sentence)
                file.write('\n')
    

