# The below is skip gram made using word2vec tunable parameters

# from gensim.models import Word2Vec
# from nltk.corpus import brown
# import multiprocessing
# sentences = list(brown.sents())
# EMB_DIM = 300
#
# w2v = Word2Vec(sentences, size=EMB_DIM, window=5, min_count=5,
#                negative = 15, workers= multiprocessing.cpu_count(), iter = 10, sg=1)


# Now is something different
# Load corpus
# Text Preprocessing
#
#
# docs = ['Well done!',
# 'Good work',
# 'Great effort',
# 'nice work',
# 'Excellent!']
#
# from keras.preprocessing.text import Tokenizer
#
# t = Tokenizer()
# t.fit_on_texts(docs)


# Import data
import numpy as np
from nltk.corpus import gutenberg
from string import punctuation
from keras.preprocessing import text

bible = list(gutenberg.sents('bible-kjv.txt'))

remove_terms = punctuation + '0123456789'
norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [" ".join(tok_sent) for tok_sent in norm_bible]

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)

# maintain the mapping betwee the words
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

vocab_size = len(word2id) + 1
embed_size = 100
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]

# Using the NIFTY skipgram generator in keras to get the pair of input/output

from keras.preprocessing.sequence import skipgrams

skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10)for wid in wids]

# Looking inside the hood, so that we can make changes later, uncomment the line below
pairs, labels = skip_grams[0][0],skip_grams[0][1]
for i in range(len(skip_grams[0][0])):
    print("({:s} ({:d}) , {:s} ({:d}) -> {:d})".format(
        id2word[pairs[i][0]], pairs[i][0], id2word[pairs[i][1]], pairs[i][1], labels[i]
    ))

from keras.layers import dot
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input


# *******************************SEQUENTIAL****************************************
# Build skip-gram architecture always using the functional paradigm
# They are easier to pass data from one layer to another. In sequence arises
# many problems.
# word_model = Sequential()
# word_model.add(Embedding(vocab_size, embed_size,
#                          embeddings_initializer="glorot_uniform",
#                          input_length=1))
# word_model.add(Reshape((embed_size, )))
#
# context_model = Sequential()
# context_model.add(Embedding(vocab_size, embed_size,
#                             embeddings_initializer="glorot_uniform",
#                             input_length=1))
# context_model.add(Reshape((embed_size, )))
#
# dot_product = Dot([word_model, context_model], axes=1)
# dot_product = Reshape((1,))(dot_product)
#
# output = Dense(1, activation='sigmoid')(dot_product)
#
# # *******************************SEQUENTIAL****************************************


input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, embed_size, input_length= 1,
                      embeddings_initializer="glorot_uniform")

word_embedding = embedding(input_target)
word_embedding = Reshape((embed_size, 1))(word_embedding)
context_embedding = embedding(input_context)
context_embedding = Reshape((embed_size, 1))(context_embedding)

# performing the dot product operation
dot_product = dot([word_embedding, context_embedding], axes=1)
dot_product = Reshape((1,))(dot_product)

# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
model = Model([input_target, input_context], output)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print(model.summary())

# Visualize model structure
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file="skip-gram-classic.png")
#
# SVG(model_to_dot(model, show_shapes=True, show_layer_names=False,
#                  rankdir='TB').create(prog='dot', format='svg'))



# from tensorflow.python.keras.utils.vis_utils import plot_model
# plot_model(model, to_file="skip-gram-classic.png")

# GraphViz executable not found, ask to make graphviz executable to the admin


for epoch in range(1, 2):
    loss = 0
    for i, elem in enumerate(skip_grams):
        # First ensure that length of element at position 0 is not zero
        if (len(elem[0])==0):
            continue
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 10000 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += model.train_on_batch(X,Y)

    print('Epoch:', epoch, 'Loss:', loss)



