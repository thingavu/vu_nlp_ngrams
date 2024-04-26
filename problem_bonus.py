import numpy as np
from problem1 import get_word_index_dict
from problem2 import unigram_model
from problem3 import bigram_model
from problem4 import bigram_model_w_smoothing


#load the indices dictionary
word_index_dict = get_word_index_dict(vocab_path="brown_vocab_100.txt")

count_uni, probs_uni = unigram_model()
count_bigram, probs_bigram = bigram_model()

# Calculate PMI for all word pairs
pmi_values = []
for i in range(len(word_index_dict)):
    for j in range(len(word_index_dict)):
        if count_uni[i] >= 10 and count_uni[j] >= 10:
            if count_bigram[i, j] != 0:
                pmi = np.log((count_bigram[i, j] * len(word_index_dict)) / (count_uni[i] * count_uni[j]))
            else:
                pmi = 0
            pmi_values.append((pmi, i, j))

# Sort the PMI values in descending order
pmi_values.sort(reverse=True)

top_20 = pmi_values[:20]
bottom_20 = pmi_values[-20:]

# Print the top 20 word pairs with highest PMI
print("================= Top 20 ==========================")
for pmi, i, j in top_20:
    word1 = list(word_index_dict.keys())[i]
    word2 = list(word_index_dict.keys())[j]
    print(f"{word1}, {word2}: {pmi}")

# Print the bottom 20 word pairs with lowest PMI
print("\n================= Bottom 20 ==========================")
for pmi, i, j in bottom_20:
    word1 = list(word_index_dict.keys())[i]
    word2 = list(word_index_dict.keys())[j]
    print(f"{word1}, {word2}: {pmi}")