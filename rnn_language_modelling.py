import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime

from RNNNumpy import *
from utils import *

import matplotlib.pyplot as plt

# Download NLTK model data (you need to do this once)
# nltk.download("book")

#%% Training Data and Preprocessing
load_pickle = True
vocabulary_size = 8000 # i.e. 8000 most common words
unknown_token = "UNKNOWN_TOKEN" # Words not in vocab will be replaced with this word. 
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
if load_pickle:
    sentences = pickle.load(open( "./data/sentences.p", "rb" ))
    tokenized_sentences = pickle.load(open("./data/tokenized_sentences.p", "rb" ))
    word_freq = pickle.load(open("./data/word_freq.p", "rb" ))
    vocab = pickle.load(open("./data/vocab.p", "rb" ))
    index_to_word = pickle.load(open("./data/index_to_word.p", "rb" ))
    word_to_index = pickle.load(open("./data/word_to_index.p", "rb" ))
else:
    print "Reading CSV file..."
    fpath = './data/reddit-comments-2015-08.csv'
    sentences, tokenized_sentences, word_freq, vocab, index_to_word, word_to_index \
      = preprocess_data(fpath, vocabulary_size, unknown_token, sentence_start_token, sentence_end_token)
    # Pickle the files. Preprocessing takes a long time.   
    pickle.dump(sentences, open("./data/sentences.p", "wb"))
    pickle.dump(tokenized_sentences, open("./data/tokenized_sentences.p", "wb"))
    pickle.dump(word_freq, open("./data/word_freq.p", "wb"))
    pickle.dump(vocab, open("./data/vocab.p", "wb"))
    pickle.dump(index_to_word, open("./data/index_to_word.p", "wb"))
    pickle.dump(word_to_index, open("./data/word_to_index.p", "wb"))

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print an training data example
x_example, y_example = X_train[17], y_train[17]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print "Shape of o", o.shape
print "o is:", o

predictions = model.predict(X_train[10])
print "shape predictions", predictions.shape
print "predictions", predictions

# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
#grad_check_vocab_size = 100
#np.random.seed(10)
#model2 = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
#model2.gradient_check([0,1,2,3], [1,2,3,4])

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

# np.random.seed(10)
#model = RNNNumpy(vocabulary_size)
# model.sgd_step(X_train[10], y_train[10], 0.005)
# %timeit model.sgd_step(X_train[10], y_train[10], 0.005)

# np.random.seed(10)
# Train on a small subset of the data to see what happens
# model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:10], y_train[:10], nepoch=10, evaluate_loss_after=1)


