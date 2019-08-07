import re
def mreplace(s, chararray, newchararray):
    for a, b in zip(chararray, newchararray):
        s = s.replace(a, b)
    return s
def normalizer(word):
    word = re.sub(r'[^a-zA-Záéíóúñ]', '', word.lower())
    return mreplace(word,'áéíóú','aeiou')
def solveAnalogy(model, k, w1, w2, w3):
    return model.wv.most_similar(positive=[w3, w2], negative=[w1], topn = k)
def evaluator(model, k, w1, w2, w3, correct):
    responses = [normalizer(response[0]) for response in solveAnalogy(model, k, w1, w2, w3)]
    return normalizer(correct) in responses
class AnalogyEvaluator:
    def __init__(self, k=5):
        self.k = k
    def set_model(self, model):
        self.model = model
        self.vocabulary = list(model.wv.vocab.keys())
    def set_dataset(self, dataset):
        self.questions = {}
        with open(dataset, encoding='utf-8') as file:
            current_group = ""
            for line in file:
                line = line.rstrip()
                if line.startswith(':'):
                    current_group = line
                    self.questions[current_group] = []
                else:
                    line = line.split(' ')
                    if set(line).issubset(self.vocabulary):
                        self.questions[current_group].append(line)
    def evaluate(self):
        self.correct = {}
        for group, questions in self.questions.items():
            correct = []
            for question in questions:
                if evaluator(self.model, self.k, question[0], question[1], question[2], question[3]):
                    correct.append(question)
            self.correct[group] = correct
        self.performance = {}
        for group,questions in self.questions.items():
            accuracy = len(self.correct[group])/len(questions)
            self.performance[group] = accuracy