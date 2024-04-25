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
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i

vocab.close()

f = codecs.open("brown_100.txt")

vocab_size = len(word_index_dict)
counts = np.zeros((vocab_size, vocab_size, vocab_size))

# Open the corpus file
f = codecs.open("brown_100.txt", encoding='utf-8')

for line in f:
    words = line.strip().split()
    for i in range(2, len(words)):
        w1, w2, w3 = words[i-2].lower(), words[i-1].lower(), words[i].lower()
        if w1 in word_index_dict and w2 in word_index_dict and w3 in word_index_dict:
            counts[word_index_dict[w1], word_index_dict[w2], word_index_dict[w3]] += 1
f.close()

alpha = 0.1
counts_w_apla = counts + alpha

probabilities = np.zeros((vocab_size, vocab_size, vocab_size))
probabilities_w_smooth = np.zeros((vocab_size, vocab_size, vocab_size))

for i in range(vocab_size):
    for j in range(vocab_size):
        total_count = np.sum(counts[i, j, :])
        if total_count > 0:
            probabilities[i, j, :] = counts[i, j, :] / total_count

        total_count = np.sum(counts_w_apla[i, j, :])
        if total_count > 0:
            probabilities_w_smooth[i, j, :] = counts_w_apla[i, j, :] / total_count
                          
trigram_contexts = [
    ("in", "the", "past"),
    ("in", "the", "time"),
    ("the", "jury", "said"),
    ("the", "jury", "recommended"),
    ("jury", "said", "that"),
    ("agriculture", "teacher", ",")
]

with codecs.open("trigram_probs.txt", "w", encoding="utf-8") as out_file:
    for w1, w2, w3 in trigram_contexts:
        index1, index2, index3 = word_index_dict[w1], word_index_dict[w2], word_index_dict[w3]
        prob = probabilities[index1, index2, index3]
        out_file.write(f'p({w3} | {w1}, {w2}) = {prob:.5f}\n')

with codecs.open("trigram_snooth_probs.txt", "w", encoding="utf-8") as out_file:
    for w1, w2, w3 in trigram_contexts:
        index1, index2, index3 = word_index_dict[w1], word_index_dict[w2], word_index_dict[w3]
        prob = probabilities_w_smooth[index1, index2, index3]
        out_file.write(f'p({w3} | {w1}, {w2}) = {prob:.5f}\n')
