# Authors: Alexandre Gramfort
#          Chloe Clavel
# License: BSD Style.
# TP Cours ML Telecom ParisTech MDI343

import os.path as op
import numpy as np
import collections
import math, nltk

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB

from nltk import SnowballStemmer
from nltk import pos_tag

###############################################################################
# Load data
print("Loading dataset")

from glob import glob
filenames_neg = sorted(glob(op.join('.', 'data', 'imdb1', 'neg', '*.txt')))
filenames_pos = sorted(glob(op.join('.', 'data', 'imdb1', 'pos', '*.txt')))

texts_neg = [open(f).read() for f in filenames_neg]
texts_pos = [open(f).read() for f in filenames_pos]

texts = texts_neg + texts_pos

# tag of POS: adj, noun, adverb, verb
tagFilter = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# record real classificatioin: neg = 0, pos = 1
y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))
sw = open('data/english.stop', 'r').read().split() # stopwords reader

###############################################################################
# Start part to fill in

def count_words(texts):
    """Vectorize text : return count of each word in the text snippets

    Parameters
    ----------
    texts : list of str
        The texts

    Returns
    -------
    vocabulary : dict
        A dictionary that points to an index in counts for each word.
    counts : ndarray, shape (n_samples, n_features)
        The counts of each word in each text.
        n_samples == number of documents.
        n_features == number of words in vocabulary.
    """
    nText = len(texts)
    vocabulary = {}
    # word stemming
    snowball_stemmer = SnowballStemmer("english")
    for it, text in enumerate(texts):
        # tokenize texts and assign pos tag
        text = nltk.word_tokenize(text)
        text = nltk.pos_tag(text)
    
        for word in text: # ex. word:('good', 'JJ')
            if word[0] not in sw and word[1] in tagFilter:            
                if word[0] in vocabulary.keys():                
                    vocabulary[word[0]][it] += 1
                    
                else:
                    vocabulary[word[0]] = [0] * nText
                    vocabulary[word[0]][it] = 1

        # stemming texts      
        # for word in text.split():
            # temp = snowball_stemmer.stem(word)
            # if temp not in sw:
                #   if temp in vocabulary.keys():
                    # vocabulary[temp][it] += 1
                    # else:
                    # vocabulary[temp] = [0] * nText
                    # vocabulary[temp][it] = 1

    vocabulary = collections.OrderedDict(sorted(vocabulary.items()))
    counts = np.hstack(np.asarray(v).reshape((nText,1)) for v in vocabulary.values())

    for i, key in enumerate(vocabulary):
        vocabulary[key] = i
    return vocabulary, counts


class NB(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.nTokens = len(vocabulary.keys())
        self.prior = np.zeros(2)
        self.condprob = np.zeros((self.nTokens, 2))

    def fit(self, X, y):
        """Train : return count of each word in the text snippets
            
            c = 0 for the class neg
            c = 1 for the class pos

            nTokens: number of words in vocabulary
            nTokens_c: number of a word in vocabulary for class c
            nTexts_c: number of texts from class c
            index_c: list, c-class texts indices in matrix X
            X_c: ndarray, shape (nTexts_c, nTokens_c)
            sumTokens_c: sum of occurrences of each token from class c
        """
        
        for c in [0,1]: 
            self.nTexts_c = sum(y == c)
            self.prior[c] = 1.0*self.nTexts_c/len(y)   
            self.index_c = np.where(y == c)[0]
            self.X_c = np.zeros(self.nTokens, dtype = np.int)
            for index in self.index_c:
                self.X_c = np.vstack((self.X_c, X[index]))
            self.X_c = self.X_c[1:]
            self.sumTokens_c = self.X_c.sum()
            i = 0
            for column in self.X_c.T:
                self.nTokens_c = sum(column)
                # Laplace lissage
                self.condprob[i][c] = 1.0*(self.nTokens_c + 1)/(self.sumTokens_c + self.nTokens)     
                i = i + 1
        return self

    def predict(self, X):
        self.y_tst = []
        self.score = np.zeros(2)
        for row in X:
            for c in [0,1]:
                self.score[c] = math.log(self.prior[c])
                for idx, val in enumerate(row):
                    if val != 0:
                        self.score[c] += math.log(self.condprob[idx][c])
            self.y_tst.append(np.argmax(self.score))
        return self.y_tst

    def score(self, X, y):  
        return np.mean(self.predict(X) == y) # take the average resultat as accuracy measurement

# Count words in text
vocabulary, X = count_words(texts)

# Try to fit, predict and score
# nb = MultinomialNB()
nb = NB()

# 5Flod Cross Validation, shuffle the data before slipping it
kf = KFold(n=len(texts), n_folds = 5, shuffle=True)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

nb.fit(X_train, y_train)
print nb.score(X_test, y_test)


# half of data as training set, half as testing set
# nb.fit(X[::2], y[::2])
# print nb.score(X[1::2], y[1::2])
