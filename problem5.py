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
from problem1 import get_word_index_dict

#load the indices dictionary
#TODO: import part 1 code to build dictionary
word_index_dict = get_word_index_dict(vocab_path="brown_vocab_100.txt")

def trigram_model(smoothing=False):
    out_file = "trigram_probs.txt"
    if smoothing:
        out_file = "trigram_smooth_probs.txt"

    vocab_size = len(word_index_dict)
    counts = np.zeros((vocab_size, vocab_size, vocab_size))

    # Open the corpus file
    with codecs.open("brown_100.txt", encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for i in range(2, len(words)):
                w1, w2, w3 = words[i-2].lower(), words[i-1].lower(), words[i].lower()
                if w1 in word_index_dict and w2 in word_index_dict and w3 in word_index_dict:
                    counts[word_index_dict[w1], word_index_dict[w2], word_index_dict[w3]] += 1

    if smoothing:
        alpha = 0.1
        counts += alpha

    probabilities = np.zeros((vocab_size, vocab_size, vocab_size))

    for i in range(vocab_size):
        for j in range(vocab_size):
            total_count = np.sum(counts[i, j, :])
            if total_count > 0:
                probabilities[i, j, :] = counts[i, j, :] / total_count

    trigram_contexts = [
        ("in", "the", "past"),
        ("in", "the", "time"),
        ("the", "jury", "said"),
        ("the", "jury", "recommended"),
        ("jury", "said", "that"),
        ("agriculture", "teacher", ",")
    ]

    with codecs.open(out_file, "w", encoding="utf-8") as out_file:
        for w1, w2, w3 in trigram_contexts:
            index1, index2, index3 = word_index_dict[w1], word_index_dict[w2], word_index_dict[w3]
            prob = probabilities[index1, index2, index3]
            out_file.write(f'{prob:.7f}\n')

    return counts, probabilities

if __name__ == '__main__':
    trigram_model()
    trigram_model(smoothing=True)