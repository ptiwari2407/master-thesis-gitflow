# Load the word-vector
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

#Load other necessary libraries
import pandas as pd
import json

# Load both the files

# Deriving the list of random words
csv1 = "data/random_single_lemma.csv"

df_random_single_lemma = pd.read_csv(csv1)
col = ['index', 'list']
df_random_single_lemma.columns = col
random_lemma = df_random_single_lemma['list']
random_lemma = list(random_lemma)

# Deriving the list of single_lemma_pair
file = open('data/single_pair.json', 'r')
single_lemma_pair = json.load(file)



# Now, we have both the list and we shall feed it to compare to the similarity index




