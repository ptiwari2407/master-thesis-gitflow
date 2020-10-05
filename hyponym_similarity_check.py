#Load other necessary libraries
import pandas as pd
import json
import numpy as np
from functions import similarity_check_Word2Vec, percentage_accuracy


# Random_word_from files
csv1 = "data/random_single_lemma.csv"

df_random_single_lemma = pd.read_csv(csv1)
col = ['index', 'list']
df_random_single_lemma.columns = col
random_list = df_random_single_lemma['list']
random_list = list(random_list)


# Deriving from json file the hyponym_combination_pair
file = open('data/hypo_syn_pair_1.json', 'r')
data_1 = json.load(file)
file.close()

file = open('data/hypo_syn_pair_2.json', 'r')
data_2 = json.load(file)
file.close()

hypo_pair = data_1

# Let us make sure first that no point in Random list is empty

x = pd.Series(random_list)
z = x[x.isnull()].index.to_list()
for item in z:
    random_list[item] = random_list[3]          #substituting Nan with a value

# Number of random words generated on the fly for the algorithm and hence its length should be equal to number of words in hypo_pair
# np.random.seed(412)
# r_index = np.random.randint(low=0, high= len(random_list), size= len(hypo_pair))
# ohne_phrase = pd.Series(random_list)
# random_list = (list(ohne_phrase[r_index]))

# Running the analysis now for case 1
not_in_vocab, correct_pred, false_pred = similarity_check_Word2Vec(hypo_pair, random_list)
print("Case 1: When 1st word of pair is *support* and 2nd word is queried along with a random word")
# calculating the accuracy in %
print("Accuracy for correct predictions is: %.4f" %percentage_accuracy(len(correct_pred), len(false_pred)))
print("Number of correct predictions is: %d" %len(correct_pred))
print("Number of incorrect predictions is: %d" %len(false_pred))
print("Number of words in synset pair not found in word2vec model is: %d" %len(not_in_vocab))
print(" ")

# Running the analysis now for case 2
not_in_vocab, correct_pred, false_pred = similarity_check_1(hypo_pair, random_list, 1)
print("Case 2: When 2nd word of pair is *support* and 1st word is queried along with a random word")
print("Accuracy for correct predictions is: %.4f" %percentage_accuracy(len(correct_pred), len(false_pred)))
print("Number of correct predictions is: %d" %len(correct_pred))
print("Number of incorrect predictions is: %d" %len(false_pred))
print("Number of words in synset pair not found in word2vec model is: %d" %len(not_in_vocab))
print(" ")

print("Task Finished")