#
# def solution(predicted, observed):
#     import math
#     val = []
#     for x,y in zip(predicted, observed):
#         temp = (x-y)**2
#         val.append(temp)
#     net_sum = sum(val)
#     return math.sqrt((net_sum/len(val)))
#
#
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing

sentences = brown.sents()
print(type(sentences))

s =[]
for i in range(len(sentences)):
    temp = sentences[i]
    s.append(temp)

EMB_DIM = 300
w2v = Word2Vec(s, size = EMB_DIM, window=5, min_count = 5, negative=15, iter = 10, workers = multiprocessing.cpu_count())
