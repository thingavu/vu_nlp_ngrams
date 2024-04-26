#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
import numpy as np


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    line = line.rstrip('\n')
    word_index_dict[line] = i

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict))

for line in f:
    sentence = line.rstrip('\n').split()
    for word in sentence:
        if word.lower() in word_index_dict:
            counts[word_index_dict[word.lower()]] += 1

print(counts)   

f.close()

#TODO: normalize and writeout counts. 

probs = counts / np.sum(counts)

# print(probs)

np.savetxt("unigram_probs.txt", probs)

