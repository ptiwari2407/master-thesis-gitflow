# Load necessary libraries

import numpy as np
from nltk.corpus import gutenberg
from string import punctuation
from keras.preprocessing import text
import random


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


def skipgrams_modified_generator(sequence, vocabulary_size, window_size=4,
                        negative_samples=1., shuffle=True,
                        categorical=False, sampling_table=None, seed=None):

    """Generate skipgram also by keeping the structures of the sentences.
    1. This is further extension of default skipgrams provided under keras.preprocessing.sequence.skipgrams
    2. This generation is done by keeping in mind the semi-supervised output to be provided for embedding layer.
    3. It gives three output: [target, context], label, position
    4. Position tells us where the context occurs with respect to the target: whether Left or Right. it will be a softmax in model, left=1 and right=2
    5. if the target-context pair has the label 0, position will have value 0
    """
    couples = []
    labels = []
    position = []

    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)

        for j in range(window_start, window_end):
            if j!=i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)
                if(j<i):
                    position.append(1)
                else:
                    position.append(2)

    if negative_samples > 0:
        num_negative_samples = int(len(labels)*negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i % len(words)],
                         random.randint(1, vocabulary_size-1)]
                        for i in range(num_negative_samples)]
        if categorical:
            labels += [[1,0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples
        position += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 1006)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
        random.seed(seed)
        random.shuffle(position)

    return couples, labels, position








skip_grams = [skipgrams_modified_generator(wid, vocabulary_size=vocab_size, window_size=10)for wid in wids]
pairs, labels, position = skip_grams[0][0], skip_grams[0][1], skip_grams[0][2]
for i in range(len(skip_grams[0][0])):
    print("({:s} ({:d}) , {:s} ({:d}) -> {:d} -> {:d})".format(
        id2word[pairs[i][0]], pairs[i][0], id2word[pairs[i][1]], pairs[i][1], labels[i], position[i]
    ))




from keras.layers import dot
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input

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
output_1 = Dense(1, activation='sigmoid', name="labels-identifier")(dot_product)
output_2 = Dense(3, activation= 'softmax', name="position-identifier")(dot_product)
model = Model([input_target, input_context], [output_1, output_2])
model.compile(loss=['mean_squared_error', 'SparseCategoricalCrossentropy'], optimizer='rmsprop')

print(model.summary())

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
        # loss += model.train_on_batch(X,Y)

    print('Epoch:', epoch)