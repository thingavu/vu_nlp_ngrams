#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
import numpy as np
from problem1 import get_word_index_dict

#load the indices dictionary
#TODO: import part 1 code to build dictionary
word_index_dict = get_word_index_dict(vocab_path="brown_vocab_100.txt")

def unigram_model():
    with open("brown_100.txt") as f:
        counts = np.zeros(len(word_index_dict))
        for line in f:
            sentence = line.rstrip('\n').split()
            for word in sentence:
                if word.lower() in word_index_dict:
                    counts[word_index_dict[word.lower()]] += 1
    #TODO: normalize and writeout counts.
    probs = counts / np.sum(counts)
    np.savetxt("unigram_probs.txt", probs)
    return counts,probs

if __name__ == '__main__':
    unigram_model()
