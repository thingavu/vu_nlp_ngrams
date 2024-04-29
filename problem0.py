import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import brown

nltk.download('brown')
nltk.download('averaged_perceptron_tagger')

# Choose two genres
genres = ['news', 'romance']

def analyze_corpus(genre=None):
    words = brown.words(categories=genre)
    sentences = brown.sents(categories=genre)
    tagged_words = nltk.pos_tag(words)

    num_tokens = len(words)
    num_types = len(set(words))
    num_words = sum(1 for word in words if word.isalpha())
    avg_words_per_sentence = num_words / len(sentences)
    avg_word_length = sum(len(word) for word in words if word.isalpha()) / num_words
    pos_counts = Counter(tag for word, tag in tagged_words)
    most_common_pos = pos_counts.most_common(10)

    print(f'Genre: {genre if genre else "whole corpus"}')
    print(f'Number of tokens: {num_tokens}')
    print(f'Number of types: {num_types}')
    print(f'Number of words: {num_words}')
    print(f'Average number of words per sentence: {avg_words_per_sentence:.2f}')
    print(f'Average word length: {avg_word_length:.2f}')
    print(f'Ten most frequent POS tags: {most_common_pos}')

    # Write information to file
    with open('problem0_information.txt', 'a') as file:
        file.write(f'Genre: {genre if genre else "whole corpus"}\n')
        file.write(f'Number of tokens: {num_tokens}\n')
        file.write(f'Number of types: {num_types}\n')
        file.write(f'Number of words: {num_words}\n')
        file.write(f'Average number of words per sentence: {avg_words_per_sentence:.2f}\n')
        file.write(f'Average word length: {avg_word_length:.2f}\n')
        file.write(f'Ten most frequent POS tags: {most_common_pos}\n')

    # Plot frequency curves
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.values(), reverse=True)

    plt.figure()
    plt.plot(sorted_word_counts)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Frequency curve for {genre if genre else "whole corpus"} (linear scale)')

    plt.figure()
    plt.loglog(sorted_word_counts)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Frequency curve for {genre if genre else "whole corpus"} (log-log scale)')
    plt.show()

# Analyze the whole corpus
analyze_corpus()

# Analyze the chosen genres
for genre in genres:
    analyze_corpus(genre)

# Close the file
with open('problem0_information.txt', 'a') as file:
    file.close()
