#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    line = line.rstrip('\n')
    word_index_dict[line] = i

# =========== Unigram model ===========
with open("brown_100.txt") as f:
    counts = np.zeros(len(word_index_dict))
    for line in f:
        sentence = line.rstrip('\n').split()
        for word in sentence:
            if word.lower() in word_index_dict:
                counts[word_index_dict[word.lower()]] += 1

probs = counts / np.sum(counts)
word_prob_map = {}
for word, prob in zip(word_index_dict.keys(), probs):
    word_prob_map[word] = prob

perplexities = []
with open("toy_corpus.txt") as corpus:
    # Calculate probability for each line in the corpus
    for line in corpus:
        sentence = line.rstrip('\n').split()
        sentence_prob = 1.0
        for word in sentence:
            if word.lower() in word_prob_map:
                sentence_prob *= word_prob_map[word.lower()]
        sent_len = len(sentence)
        perplexity = 1 / (pow(sentence_prob, 1.0 / sent_len))
        perplexities.append(perplexity)

with codecs.open("unigram_eval.txt", "w", encoding="utf-8") as out_file:
    out_file.write(f'\n'.join([str(p) for p in perplexities]))       

# ========== Bigram model ==========
with open("brown_100.txt") as f:
    counts = np.zeros((len(word_index_dict), len(word_index_dict)))
    for line in f:
        words = line.rstrip('\n').split()
        previous_word = '<s>'  # start symbol
        for word in words:
            if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
                counts[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]] += 1
            previous_word = word

    probabilities = normalize(counts, norm='l1', axis=1)

perplexities = []
with open("toy_corpus.txt") as f:
    for line in f:
        words = line.rstrip().split()
        previous_word = '<s>'
        sentprob = 1
        valid_transitions = 0
        for word in words:
            if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
                sentprob *= probabilities[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]]
                valid_transitions += 1
            previous_word = word

        perplexity = 1/pow(sentprob, 1.0/valid_transitions)
        perplexities.append(perplexity)

with codecs.open("bigram_eval.txt", "w", encoding="utf-8") as out_file:
    out_file.write(f'\n'.join([str(p) for p in perplexities]))

# ========== Bigram model with smooting ==========
counts = np.zeros((len(word_index_dict), len(word_index_dict)))

with open("brown_100.txt") as f:
    for line in f:
        words = line.rstrip().split()
        previous_word = '<s>'
        for word in words[1:]:
            if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
                counts[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]] += 1
            previous_word = word
        
alpha = 0.1
counts += alpha

probabilities = normalize(counts, norm='l1', axis=1)    

perplexities = []
with open("toy_corpus.txt") as f:
    for line in f:
        words = line.rstrip('\n').split()
        previous_word = '<s>'
        sentprob = 1
        valid_transitions = 0
        for word in words:
            if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
                sentprob *= probabilities[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]]
                valid_transitions += 1
            previous_word = word

        perplexity = 1/pow(sentprob, 1.0/valid_transitions)
        perplexities.append(perplexity)

with codecs.open("smoothed_eval.txt", "w", encoding="utf-8") as out_file:
    out_file.write(f'\n'.join([str(p) for p in perplexities]))
