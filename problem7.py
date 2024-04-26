from generate import GENERATE
from problem1 import get_word_index_dict
from problem2 import unigram_model
from problem3 import bigram_model
from problem4 import bigram_model_w_smoothing

word_index_dict = get_word_index_dict(vocab_path="brown_vocab_100.txt")
start_word = "<s>"
# ========== unigram =================
max_words = 25
model_type = "unigram"
_, probs = unigram_model()
sentences = []
for _ in range(10):
    sent = GENERATE(word_index_dict, probs, model_type, max_words, start_word)
    sentences.append(start_word + " " + sent)

with open("unigram_generation.txt", "w") as f:
    f.write("\n".join(sentences))

# ========== bigram =================
max_words = 30 # 25 was not enough - most of the time did not reach the end of the sentence </s>
model_type = "bigram"
_, probs = bigram_model()
sentences = []
for _ in range(10):
    sent = GENERATE(word_index_dict, probs, model_type, max_words, start_word)
    sentences.append(" " + sent) # do not have to add the beginning <s>

with open("bigram_generation.txt", "w") as f:
    f.write("\n".join(sentences))

# ========== bigram with smoothing =================
max_words = 40 # 30 was not enough - most of the time did not reach the end of the sentence </s>
model_type = "bigram"
_, probs = bigram_model_w_smoothing()
sentences = []
for _ in range(10):
    sent = GENERATE(word_index_dict, probs, model_type, max_words, start_word)
    sentences.append(" " + sent)

with open("smooth_generation.txt", "w") as f:
    f.write("\n".join(sentences))