import numpy as np
import pandas as pd
from math import factorial as f
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# Loading the word-vectors
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')



# Multi-level list flattening using Recursion
def unlist(m):
  q = []
  p = []
  for x in m:
    if (type(x) is list):
      for y in x:
        q.append(y)
      continue
    q.append(x)
  for i in range(len(q)):
    if (type(q[i]) is not list):
      if (i == (len(q)-1)):
        return q
      continue
    else:
      break
  for elem in q:
    if (type(elem) is list):
      z = unlist(elem)
      q.remove(elem)
      for item in z:
        q.append(item)
  return q

allSynsets = list(wn.all_synsets(wn.NOUN))

# Logic-1 : Access only the lemma names of the synset itself
syn_group = []
syn_dict = dict()
for elem in allSynsets:
    syn_group.append(elem.name())

for elem in syn_group:
    word = wn.synset(elem)
    syn_dict[elem] = word.lemma_names()

# Separate the syn_dict into two categories, the one which has only one lemma and the one which has more tha 1 lemma

single_lemma = dict()
multiple_lemma = dict()
for key, value in syn_dict.items():
  if(len(value)<2):
    single_lemma[key] = value
  else:
    multiple_lemma[key] = value

# from the single_lemma, generate THE list of random words, which would be completely un-correlated later when we
# would need to feed to algorithms some random words
random_list = []
a = []
for key, value in single_lemma.items():
    random_list.append(value)

random_list = unlist(random_list)


# Now we shall filter out the from random list strings that have _ in them, so that our algorithm does
# not break on account of supply of phrasal word group e.g car is ok, but rail_car is not recognized by word-vector from random_list side.

ohne_phrase =[]
for item in random_list:
    if('_' not in item and item in wv.vocab):
        ohne_phrase.append(item)

# Deal with multi-lemma items for a particular synset
multi_lemma_list = []
length_per_list = []
for key,value in multiple_lemma.items():
  multi_lemma_list.append(value)
  length_per_list.append(len(value))

def combination(n):
  """Total combination of words needed to be generated
  when feeding to algorithm, while selecting two values at a time from list.
  This way we also get to know how many comparisons are run in total"""
  return f(n) // f(2) // f(n-2)

total_runs = sum(map(combination, length_per_list))  # Total number of comparisons that would be run

def generate_combination_pair(m_list):
  """This generates the total combination pair from the multi_lemma_list"""
  out_list = []
  for elem in m_list:
    for i in range(len(elem)):
      for j in range(i+1, len(elem)):
        out_list.append([elem[i], elem[j]])
  return out_list

total_pairs = (generate_combination_pair(multi_lemma_list))

# Generate Random numbers between 1 to len(ohne_phrase) for total_runs(total number of comparisons to be run)
np.random.seed(412)
r_index = np.random.randint(low=0, high= len(ohne_phrase), size= total_runs)
ohne_phrase = pd.Series(ohne_phrase)
random_list = pd.Series(list(ohne_phrase[r_index]))

# save both the lists to a file.
random_list.to_csv('data/random_single_lemma.csv')
# total_pairs.to_csv('data/single_lemma_pair.csv')

total_pair_dict = dict()
for item in range(0, len(total_pairs)):
  total_pair_dict[item] = total_pairs[item]

import json
file = open('data/single_pair.json', "w+")
json.dump(total_pairs, file)
file.close()






print("List stored to the files")

# Now, we start by feeding one-word from the multi-lemma-list to word-vector and see the results of comparison.

# Now we can directly use word-vector to find similiarity of multi-phrasal words
# sim = wv.n_similarity(['railroad', 'car'],['car'])
# print(sim)