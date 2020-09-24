import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


#Load other necessary libraries
import pandas as pd
import json
import numpy as np
from math import factorial as f


def CountFrequency(my_list):
    """Count the frequency of elements in a list in sorted order"""
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    print(len(freq))
    freq = dict(sorted(freq.items()))

    for key, value in freq.items():
        print("% d : % d" % (key, value))



def combination(n):
  """Total combination of words needed to be generated
  when feeding to algorithm, while selecting two values at a time from list.
  This way we also get to know how many comparisons are run in total"""
  return f(n) // f(2) // f(n-2)

def generate_combination_pair(m_list):
  """This generates the total combination pair from the multi_lemma_list"""
  out_list = []
  for elem in m_list:
    for i in range(len(elem)):
      for j in range(i+1, len(elem)):
        out_list.append([elem[i], elem[j]])
  return out_list

def percentage_accuracy(x,y):
    """
    calculates the accuracy in percentage
    :param x: is a integer
    :param y: is a integer
    :return: accuracy
    """
    return ((x/(x+y))*100)
# Load both the files


def similarity_check_1(pair, rand_list, pos=0):
    """
    1. This performs cosine similarity analysis between 1st word of the pair as support and 2nd word of pair as queried : True
    2. This performs cosine similarity analysis between 1st word of the pair and random word queried as False
    3. word_not_in_vocab
    """
    word_not_in_vocab = list()
    correct_pred = list()
    false_pred = list()

    for x,y in zip(pair, rand_list):
        support = x[pos]                  #Given word
        query_T = x[1-pos]                  #True Queried
        query_F = y                     #False Queried

        if (support in wv.vocab and query_T in wv.vocab):
            syn_sim = wv.n_similarity([support], [query_T])
            rand_sim = wv.n_similarity([support], [query_F])
            if(syn_sim > rand_sim):
                correct_pred.append(syn_sim)
            else:
                false_pred.append(rand_sim)
        else:
            word_not_in_vocab.append(x)

    return (word_not_in_vocab, correct_pred, false_pred)