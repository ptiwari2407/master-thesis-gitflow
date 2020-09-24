# this script makes up data by going one level down: top-down traversal

import nltk
import json
from nltk.corpus import wordnet as wn
from functions import CountFrequency, generate_combination_pair


allSynsets = list(wn.all_synsets(wn.NOUN))
sub_allSynsets = allSynsets[0:40]

# Multi-level flattening of list using recursion
flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]

temp = list()
syn_group = list()
syn_dict = dict()
hypo = lambda s: s.hyponyms()
for elem in allSynsets:
    syn = elem.name()
    syn_dict[syn] = elem.lemma_names()
    hypo_syn_list = list(elem.closure(hypo, depth=1))
    for item in hypo_syn_list:
      temp = syn_dict[syn]
      syn_dict[syn] = temp + item.lemma_names()
    syn_group.append(flatten(syn_dict[syn]))

temp = list()
for item in syn_group:
  temp.append(len(item))

CountFrequency(temp)

syn_list = list()   # syn_list: all groups which have more than 1 word for a synset
for elem in syn_group:
    if (len(elem)>1):
        print(elem)
        syn_list.append(elem)



# Uncomment below to check the validity of the statement
# for elem in syn_list:
#     if(len(elem)<2):
#         print(elem)



# Now we shall generate combination pair from every group
hypo_syn_pair = generate_combination_pair(syn_list)

div_val = len(hypo_syn_pair)//2
hypo_syn_pair_1 = hypo_syn_pair[:div_val]
hypo_syn_pair_2 = hypo_syn_pair[div_val:]


# Next we shall save this syn_list as a json data for further analysis, for that we first convert it into a dictionary
file = open('data/hypo_syn_pair_1.json', "w+")
json.dump(hypo_syn_pair_1, file)
file.close()

file = open('data/hypo_syn_pair_2.json', "w+")
json.dump(hypo_syn_pair_2, file)
file.close()


file = open('data/hypo_syn_pair.json', 'r')
hypo_pair = json.load(file)



print("Task Finished")


