# use and compare algorithms in library sklearn

import os.path as op
import numpy as np
import collections, nltk


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import KFold


###############################################################################
# Load data
print("Loading dataset")

from glob import glob
filenames_neg = sorted(glob(op.join('.', 'data', 'imdb1', 'neg', '*.txt')))
filenames_pos = sorted(glob(op.join('.', 'data', 'imdb1', 'pos', '*.txt')))

texts_neg = [open(f).read() for f in filenames_neg]
texts_pos = [open(f).read() for f in filenames_pos]

texts = texts_neg + texts_pos
y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))
sw = open('data/english.stop', 'r').read().split()

#####################################################
# use pipeline and countVectorizer to apply Multinomal NB in sklearn

# Convert a collection of text documents to a matrix of token counts
count = CountVectorizer(vocabulary=None, ngram_range=(1,2), stop_words='english', analyzer='word', token_pattern=r'\b\w+\b')
# ngram_range : tuple (min_n, max_n) The lower and upper boundary of the range of
# n-values for different n-grams to be extracted. 
# All values of n such that min_n <= n <= max_n will be used.

pipeline = Pipeline([
    ('classifier', MultinomialNB())
])

# Learn the vocabulary dictionary and return term-document matrix.
X = count.fit_transform(texts)


kf = KFold(n=len(texts), n_folds = 5)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 1/2 traing set, 1/2 testing set
# X_train = X[::2]
# y_train = y[::2]
# X_test = X[1::2]
# y_test = y[1::2]

# use LinearSVC and LogisticRegression in sklearn library
lin_clf = LinearSVC()
lr_clf = LogisticRegression()

# training
pipeline.fit(X_train, y_train)
lin_clf.fit(X_train, y_train)
lr_clf.fit(X_train, y_train)

# validation
print pipeline.score(X_test, y_test)
print np.mean(lin_clf.predict(X_test) == y_test)
print np.mean(lr_clf.predict(X_test) == y_test)


