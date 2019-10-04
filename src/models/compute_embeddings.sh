#!/usr/bin/env bash

word2vec_binary_location="/home/fvillena/word2vec/word2vec"
corpus_location="/home/fvillena/code/referral_classifier/data/processed/corpus.txt"
embedding_file="/home/fvillena/code/referral_classifier/models/embeddings.vec"
threads=4

$word2vec_binary_location -train $corpus_location -output $embedding_file -debug 2 -cbow 0 -size 300 -threads $threads -binary 0