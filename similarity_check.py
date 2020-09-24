# Load the necessary libraries
import numpy as np
import json
import pandas as pd
from functions import similarity_check_1, percentage_accuracy

# Deriving the list of random words
csv1 = "data/random_single_lemma.csv"

df_random_single_lemma = pd.read_csv(csv1)
col = ['index', 'list']
df_random_single_lemma.columns = col
random_list = df_random_single_lemma['list']
random_list = list(random_list)

# Deriving the list of single_lemma_pair
file = open('data/single_pair.json', 'r')
single_lemma_pair = json.load(file)

# Make sure that random list has no NAN value
x = pd.Series(random_list)
z = x[x.isnull()].index.to_list()
for item in z:
    random_list[item] = random_list[3]


# Number of random words generated on the fly for the algorithm and hence its length should be equal to number of words in single_lemma_pair
np.random.seed(412)
r_index = np.random.randint(low=0, high= len(random_list), size= len(single_lemma_pair))
ohne_phrase = pd.Series(random_list)
random_list = (list(ohne_phrase[r_index]))




# def similarity_check():
#     """
#     1. This function performs a similarity check between random word and pair of words that belong to the same synset.
#     2. This similarity check is performed under the assumption that by removing the "_" or "-" or other separators from the wordnet,
#      we can formulate a vector that is equivalent to words that map meaning conveued by multi-phrasal words
#     """
#     sim_val_list = list()
#     nonsim_val_list = list()
#     counter = -1
#     for x,y in zip(single_lemma_pair, random_list):
#         counter = counter + 1
#         temp_1, temp_2 = x[0], x[1]
#         print(counter,temp_1, temp_2)
#         temp = temp_1.split("_")
#         temp.extend(temp_2.split("_"))
#         # print(temp)
#         vocab_absence = [0 for item in temp if item not in wv.vocab]
#         if not vocab_absence:
#             syn_sim = wv.n_similarity(temp_1.split('_'), temp_2.split('_'))  #synset similarity
#             r_sim_1 = wv.n_similarity(temp_1.split('_'), y.split('_'))         # random similarity with 1st word of pair and random word
#             r_sim_2 = wv.n_similarity(y.split('_'), temp_2.split('_'))      # Random similarity between 2nd pair of word and random word
#             r_sim = r_sim_1 if(r_sim_1 > r_sim_2) else r_sim_2              # select the greater of above two similarity
#             if (syn_sim > r_sim):
#                 sim_val_list.append(syn_sim)
#             else:
#                 nonsim_val_list.append(r_sim)
#         if vocab_absence:
#             nonsim_val_list.append(-100)
#     return sim_val_list, nonsim_val_list




x = 1

not_in_vocab, correct_pred, false_pred = similarity_check_1(single_lemma_pair, random_list)
print("Case 1: When 1st word of pair is *support* and 2nd word is queried along with a random word")
# calculating the accuracy in %
print("Accuracy for correct predictions is: %.4f" %percentage_accuracy(len(correct_pred), len(false_pred)))
print("Number of correct predictions is: %d" %len(correct_pred))
print("Number of incorrect predictions is: %d" %len(false_pred))
print("Number of words in synset pair not found in word2vec model is: %d" %len(not_in_vocab))
print(" ")

# Running the analysis now for case 2
not_in_vocab, correct_pred, false_pred = similarity_check_1(single_lemma_pair, random_list, 1)
print("Case 2: When 2nd word of pair is *support* and 1st word is queried along with a random word")
print("Accuracy for correct predictions is: %.4f" %percentage_accuracy(len(correct_pred), len(false_pred)))
print("Number of correct predictions is: %d" %len(correct_pred))
print("Number of incorrect predictions is: %d" %len(false_pred))
print("Number of words in sysnset pair not found in word2vec model is: %d" %len(not_in_vocab))
print(" ")


print("Task Finished")

# x, y = similarity_check_1()


