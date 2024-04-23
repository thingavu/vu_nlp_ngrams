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


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.rstrip()] = i

vocab.close()

f = codecs.open("toy_corpus.txt")

counts = np.zeros((len(word_index_dict), len(word_index_dict)))

for line in f:
    words = line.rstrip().split()
    previous_word = '<s>'  # start symbol
    for word in words[1:]:
        if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
            counts[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]] += 1
        previous_word = word

probabilities = normalize(counts, norm='l1', axis=1)

f = codecs.open("toy_corpus.txt")
for line in f:
    print(line)
    print()
    words = line.rstrip().split()
    previous_word = '<s>'
    sentprob = 1
    for word in words[1:]:
        sentprob *= probabilities[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]]
        print(f'p({word} | {previous_word}) = {probabilities[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]]:.5f}')
        print(sentprob)
        previous_word = word

    perplexity = 1/pow(sentprob, 1.0/len(words))
    print(perplexity)


f.close()

