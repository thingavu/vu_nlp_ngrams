#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import codecs
import numpy as np
from sklearn.preprocessing import normalize
from problem1 import get_word_index_dict

#load the indices dictionary
#TODO: import part 1 code to build dictionary
word_index_dict = get_word_index_dict(vocab_path="brown_vocab_100.txt")

def bigram_model_w_smoothing():
    with codecs.open("brown_100.txt") as f:
        #TODO: initialize numpy 0s array
        counts = np.zeros((len(word_index_dict), len(word_index_dict)))

        #TODO: iterate through file and update counts
        for line in f:
            words = line.rstrip().split()
            previous_word = '<s>'  # start symbol
            for word in words[1:]:
                if word.lower() in word_index_dict and previous_word.lower() in word_index_dict:
                    counts[word_index_dict[previous_word.lower()], word_index_dict[word.lower()]] += 1
                previous_word = word
        
    #TODO: add-alpha smoothing
    alpha = 0.1
    counts += alpha

    #TODO: normalize counts
    probabilities = normalize(counts, norm='l1', axis=1)

    #TODO: writeout bigram probabilities
    smooth_probs = [
        ("all", "the"),
        ("the", "jury"),
        ("the", "campaign"),
        ("anonymous", "calls")
    ]
    with codecs.open("smooth_probs.txt", "w", encoding="utf-8") as out_file:
        for prev_word, word in smooth_probs:
            prob = probabilities[word_index_dict[prev_word], word_index_dict[word]]
            out_file.write(f'{prob:.7f}\n')
    return counts, probabilities

if __name__ == '__main__':
    bigram_model_w_smoothing()