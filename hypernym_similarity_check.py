# Load necessary libraries
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import similarity_check_Word2Vec, percentage_accuracy, CountFrequency, similarity_check_Fasttext, similarity_check_Glove_wiki



# Random_word_from files
csv1 = "data/random_single_lemma.csv"

df_random_single_lemma = pd.read_csv(csv1)
col = ['index', 'list']
df_random_single_lemma.columns = col
random_list = df_random_single_lemma['list']
random_list = list(random_list)

file = open('data/hyper_syn_pair.json', 'r')
hyper_pair = json.load(file)


# Let us make sure first that no point in Random list is empty

x = pd.Series(random_list)
z = x[x.isnull()].index.to_list()
non_null_index = x.first_valid_index()
for item in z:
    random_list[item] = random_list[non_null_index]   #substituting Nan with a value

# Analysis for word2vec model: Model 1 pos 0
not_in_vocab_1_0, correct_pred_1_0, false_pred_1_0 = similarity_check_Word2Vec(hyper_pair, random_list)
# Analysis for word2vec model: Model 1 pos 1
not_in_vocab_1_1, correct_pred_1_1, false_pred_1_1 = similarity_check_Word2Vec(hyper_pair, random_list, 1)


# Analysis for fasttext model: Model 2 pos 0
not_in_vocab_2_0, correct_pred_2_0, false_pred_2_0 = similarity_check_Fasttext(hyper_pair, random_list)
# Analysis for fasttext model: Model 2 pos 1
not_in_vocab_2_1, correct_pred_2_1, false_pred_2_1 = similarity_check_Fasttext(hyper_pair, random_list, 1)

# Analysis for Glove-wiki model: Model 3 pos 0
not_in_vocab_3_0, correct_pred_3_0, false_pred_3_0 = similarity_check_Glove_wiki(hyper_pair, random_list)
# Analysis for Glove-wiki model: Model 3 pos 1
not_in_vocab_3_1, correct_pred_3_1, false_pred_3_1 = similarity_check_Glove_wiki(hyper_pair, random_list, 1)


# Plotting the similarity distribution graph
sns.distplot(correct_pred_1_0, color = "dodgerblue", label = "Word2vec " + str(percentage_accuracy(len(correct_pred_1_0), len(false_pred_1_0))))
sns.distplot(correct_pred_2_0, color="orange", label="Fasttext " + str(percentage_accuracy(len(correct_pred_2_0), len(false_pred_2_0))))
sns.distplot(correct_pred_3_0, color="deeppink", label="Glove " + str(percentage_accuracy(len(correct_pred_3_0), len(false_pred_3_0))))
plt.legend()
plt.savefig("3.png", dpi=300)
plt.show()
