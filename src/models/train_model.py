import sklearn.neural_network
import sklearn.metrics
import joblib
import numpy as np
import json

serialized_model_file = '../../models/serialized_models/model_waiting_list.joblib'
performance_file = '../../models/model_performances/model_waiting_list.json'

train_texts_location = '../../data/processed/train_text_waiting_list.npy'
train_ages_location = '../../data/processed/train_age.txt'
train_labels_location = '../../data/processed/train_labels.txt'
test_texts_location = '../../data/processed/test_text_waiting_list.npy'
test_ages_location = '../../data/processed/test_age.txt'
test_labels_location = '../../data/processed/test_labels.txt'

print('loading data')

train_texts = np.load(train_texts_location)

train_ages = []
with open(train_ages_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        train_ages.append(float(line))
train_ages = np.asarray([train_ages]).T


train_labels = []
with open(train_labels_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        if line == 'True':
            train_labels.append(True)
        else:
            train_labels.append(False)
train_labels = np.asarray([train_labels]).T

train = np.concatenate([train_texts, train_ages, train_labels], axis=1)

train = train[~np.isnan(train).any(axis=1)]


test_texts = np.load(test_texts_location)

test_ages = []
with open(test_ages_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        test_ages.append(float(line))
test_ages = np.asarray([test_ages]).T


test_labels = []
with open(test_labels_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        if line == 'True':
            test_labels.append(True)
        else:
            test_labels.append(False)
test_labels = np.asarray([test_labels]).T

test = np.concatenate([test_texts, test_ages, test_labels], axis=1)
test = test[~np.isnan(test).any(axis=1)]

print('training model')

classifier = sklearn.neural_network.MLPClassifier(
    verbose=True,
    solver='adam',
    activation='relu',
    alpha=1e-6,
    hidden_layer_sizes=16,
    max_iter=1000
)

classifier.fit(
    train[:,:301],
    train[:,301]
)

print('measuring performance')

performance = sklearn.metrics.classification_report(
    test[:,301],
    classifier.predict(
        test[:,:301]
    ),
    output_dict = True
)

print('saving files')

with open(performance_file,'w') as file:
    json.dump(performance, file, indent=1)

joblib.dump(classifier, serialized_model_file) 
