import pandas as pd
import numpy as np
import os
from utils import normalizer
import nltk

raw_data_location = r'../../data/raw/waiting_list_corpus_raw/'
processed_data_location = r'../../data/processed/'

print('loading ssmso wl')

ssmso = pd.read_csv(raw_data_location + r"Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv", 
                   low_memory=False, 
                   error_bad_lines=False,
                   sep=",")

print('ssmso shape: ' + str(ssmso.shape))

print('loading consolidated wl')

consolidated = pd.DataFrame()
for filename in os.listdir(raw_data_location):
    if filename.endswith(".csv") and filename.startswith("WL-"): 
        current = pd.read_csv(raw_data_location + filename)
        del current["Unnamed: 0"]
        current["SS"] = filename.split("-")[1]
        consolidated = pd.concat([consolidated,current], ignore_index=True)
        print(filename)
    else:
        pass

print('consolidated shape: ' + str(consolidated.shape))

print('dropping duplicated referrals')

ssmso = ssmso.dropna(subset=["FECHANACIMIENTO",'INGRESO','SOSPECHA_DIAGNOSTICA'])

print('ssmso shape: ' + str(ssmso.shape))

consolidated = consolidated.dropna(subset=["FECHA_NAC",'F_ENTRADA','SOSPECHA_DIAG'])

print('consolidated shape: ' + str(consolidated.shape))

print('consolidating corpus')

consolidated=consolidated[["SOSPECHA_DIAG"]]
ssmso = ssmso[['SOSPECHA_DIAGNOSTICA']]
ssmso.columns = ['SOSPECHA_DIAG']
consolidated = consolidated.dropna()
ssmso = ssmso.dropna()

corpus = pd.concat([ssmso,consolidated])

print('corpus shape: ' + str(corpus.shape))

print('normalizing text')

corpus["text"] = corpus.SOSPECHA_DIAG.apply(normalizer)

corpus = corpus.dropna()

corpus["text"] = np.where(corpus.text.str.len() < 2, np.nan,corpus.text)

corpus = corpus.dropna()

print('tokenizing')

corpus["tokenized"] = corpus.text.apply(nltk.tokenize.word_tokenize)

sentences = corpus.tokenized.tolist()

corpus = []
for sentence in sentences:
    if len(sentence) > 1:
        line = ' '.join(sentence)
        corpus.append(line)

with open(processed_data_location + 'waiting_list_corpus.txt', 'w', encoding='utf-8') as f:
    for item in corpus:
        f.write("%s\n" % item)