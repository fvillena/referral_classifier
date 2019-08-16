import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import nltk
from utils import normalizer
import sklearn.preprocessing
import sklearn.model_selection

print('loading dataset')

data = pd.read_csv(r"../../data/raw/waiting_list_corpus_raw/Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv", 
                   low_memory=False, 
                   error_bad_lines=False,
                   sep=",")


print('normalizing dataset')

data.INGRESO = pd.to_datetime(data.INGRESO, errors="coerce")
data.FECHANACIMIENTO = pd.to_datetime(data.FECHANACIMIENTO, errors="coerce", format="%d/%m/%y")
from datetime import timedelta, date
future = data['FECHANACIMIENTO'] > '2018-01-01'
data.loc[future, 'FECHANACIMIENTO'] -= dt.timedelta(days=365.25*100)
data["edad"] = (data.INGRESO-data.FECHANACIMIENTO).astype('timedelta64[Y]')
data=data[["GES","SOSPECHA_DIAGNOSTICA","edad"]]
data['SOSPECHA_DIAGNOSTICA'] = data.SOSPECHA_DIAGNOSTICA.astype(str)
data['SOSPECHA_DIAGNOSTICA'] = data.SOSPECHA_DIAGNOSTICA.apply(normalizer)
data = data.dropna()

data = data.drop_duplicates(subset=['SOSPECHA_DIAGNOSTICA'])

print('tokenizing dataset')

data['SOSPECHA_DIAGNOSTICA'] = data.SOSPECHA_DIAGNOSTICA.apply(nltk.tokenize.word_tokenize)

print(str(data.GES.count()) + ' data points')

print('train-test splitting')

data["edad"] = np.where(data.edad < 0, np.nan, data.edad)
data = data.dropna()
scaler = sklearn.preprocessing.MinMaxScaler()
data["edad"] = scaler.fit_transform(data[["edad"]])
data['GES'] = np.where(data['GES'] == 'SI', True, False)
data = data[data.SOSPECHA_DIAGNOSTICA.str.len() > 1]


ages = data['edad']
texts = data['SOSPECHA_DIAGNOSTICA'].tolist()
texts = [' '.join(sentence) for sentence in texts]
labels = data['GES']

ages.to_csv(r'../../data/processed/' + 'ages.txt_unique', index=False)
labels.to_csv(r'../../data/processed/' + 'labels_unique.txt', index=False)
with open(r'../../data/interim/' + 'texts_unique.txt', 'w', encoding='utf-8') as file:
    for sentence in texts:
        file.write(sentence)
        file.write('\n')