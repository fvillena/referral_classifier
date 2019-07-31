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

# data = data.sample(100000)

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
train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(data[['SOSPECHA_DIAGNOSTICA','edad']], data['GES'], test_size = 0.1, random_state = 42)

print(str(train_labels.count()) + ' train data points')
print(str(test_labels.count()) + ' test data points')

print(train_labels.value_counts())

print('balancing training subset')

train = pd.concat([train_features,train_labels], axis=1)
count_class_0, count_class_1 = train['GES'].value_counts()
df_class_0 = train[train['GES'] == False]
df_class_1 = train[train['GES'] == True]
df_class_0_under = df_class_0.sample(count_class_1)
train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
train_features = train_under[['SOSPECHA_DIAGNOSTICA','edad']]
train_labels = train_under['GES']

print(train_labels.value_counts())

print('saving subsets')

train_age = train_features['edad']
train_text = train_features['SOSPECHA_DIAGNOSTICA'].tolist()
train_text = [' '.join(sentence) for sentence in train_text]

test_age = test_features['edad']
test_text = test_features['SOSPECHA_DIAGNOSTICA'].tolist()
test_text = [' '.join(sentence) for sentence in test_text]

train_age.to_csv(r'../../data/processed/' + 'train_age.txt', index=False)
train_labels.to_csv(r'../../data/processed/' + 'train_labels.txt', index=False)
with open(r'../../data/interim/' + 'train_text.txt', 'w', encoding='utf-8') as file:
    for sentence in train_text:
        file.write(sentence)
        file.write('\n')

test_age.to_csv(r'../../data/processed/' + 'test_age.txt', index=False)
test_labels.to_csv(r'../../data/processed/' + 'test_labels.txt', index=False)
with open(r'../../data/interim/' + 'test_text.txt', 'w', encoding='utf-8') as file:
    for sentence in test_text:
        file.write(sentence)
        file.write('\n')
