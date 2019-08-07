# import sklearn.neural_network
import sklearn.linear_model
# import sklearn.metrics
# import joblib
import numpy as np
import json

np.random.seed(11)

# serialized_model_file = '../../models/serialized_models/lr_corpus_words_no_punct_multi_line.joblib'
performance_file = '../../models/model_performances/lr_SBW-vectors-300-min5.json'

# train_texts_location = '../../data/processed/train_text_biomedical_corpus.npy'
# train_ages_location = '../../data/processed/train_age.txt'
# train_labels_location = '../../data/processed/train_labels.txt'
# test_texts_location = '../../data/processed/test_text_biomedical_corpus.npy'
# test_ages_location = '../../data/processed/test_age.txt'
# test_labels_location = '../../data/processed/test_labels.txt'

texts_location = '../../data/processed/texts_SBW-vectors-300-min5.npy'
ages_location = '../../data/processed/ages.txt'
labels_location = '../../data/processed/labels.txt'
# train_labels_location = '../../data/processed/train_labels.txt'

print('loading data')

# train_texts = np.load(train_texts_location)
texts = np.load(texts_location)

# train_ages = []
# with open(train_ages_location, encoding='utf-8') as file:
#     for line in file:
#         line = line.rstrip()
#         train_ages.append(float(line))
# train_ages = np.asarray([train_ages]).T

ages = []
with open(ages_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        ages.append(float(line))
ages = np.asarray([ages]).T


# train_labels = []
# with open(train_labels_location, encoding='utf-8') as file:
#     for line in file:
#         line = line.rstrip()
#         if line == 'True':
#             train_labels.append(True)
#         else:
#             train_labels.append(False)
# train_labels = np.asarray([train_labels]).T

labels = []
with open(labels_location, encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        if line == 'True':
            labels.append(True)
        else:
            labels.append(False)
labels = np.asarray([labels]).T

# train = np.concatenate([train_texts, train_ages, train_labels], axis=1)

# train = train[~np.isnan(train).any(axis=1)]

data = np.concatenate([texts, ages, labels], axis=1)


# test_texts = np.load(test_texts_location)

# test_ages = []
# with open(test_ages_location, encoding='utf-8') as file:
#     for line in file:
#         line = line.rstrip()
#         test_ages.append(float(line))
# test_ages = np.asarray([test_ages]).T


# test_labels = []
# with open(test_labels_location, encoding='utf-8') as file:
#     for line in file:
#         line = line.rstrip()
#         if line == 'True':
#             test_labels.append(True)
#         else:
#             test_labels.append(False)
# test_labels = np.asarray([test_labels]).T

# test = np.concatenate([test_texts, test_ages, test_labels], axis=1)
# test = test[~np.isnan(test).any(axis=1)]

print('training model')

# classifier = sklearn.neural_network.MLPClassifier(
#     verbose=True,
#     solver='adam',
#     activation='relu',
#     alpha=1e-6,
#     hidden_layer_sizes=16,
#     max_iter=1000
# )

classifier = sklearn.linear_model.LogisticRegression(solver='liblinear')

cv = sklearn.model_selection.cross_validate(
    classifier,
    data[:,:301],
    data[:,301],
    cv=10,
    pre_dispatch = 3,
    n_jobs=-1,
    scoring=[
        'accuracy',
        'precision_micro',
        'precision_macro',
        'precision_weighted',
        'recall_micro',
        'recall_macro',
        'recall_weighted',
        'f1_micro',
        'f1_macro',
        'f1_weighted'
        ],
    verbose=2
)

# classifier.fit(
#     data[:,:301],
#     data[:,301]
# )

# print('measuring performance')

# performance = sklearn.metrics.classification_report(
#     test[:,301],
#     classifier.predict(
#         test[:,:301]
#     ),
#     output_dict = True
# )

print('saving files')

# with open(performance_file,'w') as file:
#     json.dump(performance, file, indent=1)

print(cv)

for key,val in cv.items():
    cv[key] = list(val)

with open(performance_file,'w') as file:
    json.dump(cv, file, indent=1)

# joblib.dump(classifier, serialized_model_file) 
