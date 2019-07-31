import nltk
import os
from utils import normalizer

raw_data_location = r'../../data/raw/biomedical_corpus_raw/'
processed_data_location = r'../../data/processed/'

print('loading raw corpus')

lines = []
for file in os.listdir(raw_data_location):
    if file.endswith('.txt'):
        with open(raw_data_location + file, encoding='utf-8') as current_file:
            for line in current_file:
                if not line.startswith('http'):
                    if len(line) > 3:
                        lines.append(line.rstrip())

raw = '\n'.join(lines)

print('sentence tokenization')
sentences = nltk.tokenize.sent_tokenize(raw)

print(str(len(sentences)) + ' sentences')

print('word tokenization')

sentences_tokenized = []

for sentence in sentences:
    normalized = normalizer(sentence)
    tokenized = nltk.word_tokenize(normalized)
    if len(tokenized) > 1:
        sentences_tokenized.append(tokenized)

print('saving corpus')

with open(processed_data_location + 'biomedical_corpus.txt', 'w', encoding='utf-8') as file:
    for sentence in sentences_tokenized:
        line = " ".join(sentence)
        file.write(line)
        file.write('\n')

