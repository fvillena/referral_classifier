import pandas as pd
import numpy as np
import os
import nltk

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

class CorpusGenerator:
    def __init__(self, raw_data_location, ssmso_data_filename):
        self.filenames=[]
        self.ssmso_location = raw_data_location + ssmso_data_filename
        for filename in os.listdir(raw_data_location):
            self.filenames.append(raw_data_location + filename)
    def load_files(self):
        self.ssmso = pd.read_csv(self.ssmso_location, 
                   low_memory=False, 
                   error_bad_lines=False,
                   sep=",")
        self.consolidated = pd.DataFrame()
        for filename in self.filenames:
            if filename.endswith(".csv") and "WL-" in filename: 
                current = pd.read_csv(filename)
                del current["Unnamed: 0"]
                current = current[["FECHA_NAC",'F_ENTRADA','SOSPECHA_DIAG']]
                self.consolidated = pd.concat([self.consolidated,current], ignore_index=True)
                print(filename)
            else:
                pass
    def preprocess(self):
        self.ssmso = self.ssmso.dropna(subset=["FECHANACIMIENTO",'INGRESO','SOSPECHA_DIAGNOSTICA'])
        self.consolidated = self.consolidated.dropna(subset=["FECHA_NAC",'F_ENTRADA','SOSPECHA_DIAG'])
        self.consolidated=self.consolidated[["SOSPECHA_DIAG"]]
        self.ssmso = self.ssmso[['SOSPECHA_DIAGNOSTICA']]
        self.ssmso.columns = ['SOSPECHA_DIAG']
        self.consolidated = self.consolidated.dropna()
        self.ssmso = self.ssmso.dropna()
        self.corpus = pd.concat([self.ssmso,self.consolidated])
        self.corpus["text"] = self.corpus.SOSPECHA_DIAG.apply(normalizer)
        self.corpus = self.corpus.dropna()
        self.corpus["text"] = np.where(self.corpus.text.str.len() < 2, np.nan,self.corpus.text)
        self.corpus = self.corpus.dropna()
    def process(self):
        self.corpus["tokenized"] = self.corpus.text.apply(nltk.tokenize.word_tokenize)
        sentences = self.corpus.tokenized.tolist()
        self.corpus = []
        for sentence in sentences:
            if len(sentence) > 1:
                line = ' '.join(sentence)
                self.corpus.append(line)
    def write_corpus(self,corpus_location):
        with open(corpus_location, mode='w', encoding='utf-8') as f:
            for item in self.corpus:
                f.write("%s\n" % item)