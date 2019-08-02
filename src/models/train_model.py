import sklearn.neural_network
import sklearn.metrics
import joblib

serialized_model_file = '../../models/serialized_models/model.joblib'
performance_file = '../../models/model_performances/model.txt'

train_texts = []
train_ages = []
train_features = []
train_labels = []

test_texts = []
test_ages = []
test_features = []
test_labels = []

classifier = sklearn.neural_network.MLPClassifier(
    verbose=True,
    solver='adam',
    activation='relu',
    alpha=1e-6,
    hidden_layer_sizes=16,
    max_iter=1000
)

classifier.fit(
    train_features,
    train_labels
)

performance = sklearn.metrics.classification_report(
    train_labels,
    classifier.predict(
        test_features
    ),
    output_dict = True
)

with open(performance_file,'w') as file:
    file.write(performance)

joblib.dump(classifier, serialized_model_file) 
