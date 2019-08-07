import utils
import gensim
import json
embedding_location = r'../../models/embeddings/SBW-vectors-300-min5.txt'
analogies_dataset_location = r'../../data/raw/medical_analogy_test_set_es.txt'
performance_location = r'../../models/model_performances/SBW-vectors-300-min5_analogies_performance.json'
embedding = gensim.models.KeyedVectors.load_word2vec_format( # loading word embeddings
    embedding_location, # using the spanish billion words embeddings
    binary=False # the model is in binary format
)
evaluator = utils.AnalogyEvaluator(5)
evaluator.set_model(embedding)
evaluator.set_dataset(analogies_dataset_location)
evaluator.evaluate()
subsets = evaluator.performance.keys()
performance = {}
for subset in subsets:
    performance[subset] = {}
    performance[subset]['n_questions'] = len(evaluator.questions[subset])
    performance[subset]['n_correct'] = len(evaluator.correct[subset])
    performance[subset]['performance'] = evaluator.performance[subset]
    performance[subset]['questions'] = evaluator.questions[subset]
    performance[subset]['correct'] = evaluator.correct[subset]
with open(performance_location,'w') as file:
    json.dump(performance, file, indent=1, ensure_ascii=False)