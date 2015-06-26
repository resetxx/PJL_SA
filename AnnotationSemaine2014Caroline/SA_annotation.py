import annotation
import os.path as op
import numpy as np
import collections
import math, nltk

from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import KFold
from sklearn import cross_validation as cv

from nltk.tokenize import RegexpTokenizer
from nltk import SnowballStemmer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


###############################################################################
# Load data
print("Loading dataset")
text = annotation.getPosNegNeuTexts()

# concatenate pos(neg/neu) texts into a single list of words
texts_pos = sum(text[0],[])
texts_neg = sum(text[1],[])
texts_neu = sum(text[2],[])

# texts = texts_neg + texts_pos
texts = texts_neg + texts_neu + texts_pos 

y = np.ones(len(texts), dtype=np.int) # neu: 1 
y[:len(texts_neg)] = 0. # neg: 0
y[len(texts_neg)+len(texts_neu):] = 2. # pos: 2

print("%d documents" % len(texts))
sw = open('../data/english.stop', 'r').read().split()
# pos tag
tagFilterAdj = ['JJ', 'JJR', 'JJS']
tagFilterNoun = ['NN', 'NNS', 'NNP', 'NNPS']
tagFilterAdv = ['RB', 'RBR', 'RBS']
tagFilterVerb =['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
tagFilter = tagFilterAdj + tagFilterNoun + tagFilterAdv + tagFilterVerb

tokenizer = RegexpTokenizer('\s+', gaps=True)
lmtzr = WordNetLemmatizer() # lemma
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
    # snowball_stemmer = SnowballStemmer("english")
     
    for it, text in enumerate(texts):
        text = tokenizer.tokenize(text)
        text = nltk.pos_tag(text)
        # for word in text.split():
        for word in text:        
            # if word not in sw:
            if word[0] not in sw and word[1] in tagFilter:
                # temp = snowball_stemmer.stem(word[0])
                if word[1] in tagFilterVerb:
                    temp = lmtzr.lemmatize(word[0], 'v')
                elif word[1] in tagFilterAdv:
                    temp = lmtzr.lemmatize(word[0], 'r')
                elif word[1] in tagFilterAdj:
                    temp = lmtzr.lemmatize(word[0], 'a')
                elif word[1] in tagFilterNoun:
                    temp = lmtzr.lemmatize(word[0], 'n')                

                # temp = snowball_stemmer.stem(word) 
                if temp in vocabulary.keys():
                    vocabulary[temp][it] += 1 
                else:
                    vocabulary[temp] = [0] * nText
                    vocabulary[temp][it] = 1
    
    vocabulary = collections.OrderedDict(sorted(vocabulary.items()))
    counts = np.hstack(np.asarray(v).reshape((nText,1)) for v in vocabulary.values())
 
    for i, key in enumerate(vocabulary):
        vocabulary[key] = i
    return vocabulary, counts


class NB(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.nTokens = len(vocabulary.keys())
        self.prior = np.zeros(3)
        self.condprob = np.zeros((self.nTokens, 3))
        # self.prior = np.zeros(2)
        # self.condprob = np.zeros((self.nTokens, 2))

    def fit(self, X, y):
        """Train : return count of each word in the text snippets
            
            c = 0 for the class neg
            c = 1 for the class neu
            c = 2 for the class pos

            nTokens: number of words in vocabulary
            nTokens_c: number of a word in vocabulary for class c
            nTexts_c: number of texts from class c
            index_c: list, c-class texts indices in matrix X
            X_c: ndarray, shape (nTexts_c, nTokens_c)
            sumTokens_c: sum of occurrences of each token from class c
        """
        
        for c in [0, 1, 2]:
        # for c in [0, 1]: 
            self.nTexts_c = sum(y == c)
            self.prior[c] = 1.0*self.nTexts_c/len(y)    
            self.index_c = np.where(y == c)[0]
            self.X_c = np.zeros(self.nTokens, dtype = np.int)
            for index in self.index_c:
                self.X_c = np.vstack((self.X_c, X[index]))
            self.X_c = self.X_c[1:]

            self.sumTokens_c = self.X_c.sum()
            # len(self.X_c) : each class's texts number ;
            # len(self.X_c[0]): tokens number of class neg

            i = 0
            for column in self.X_c.T:
                self.nTokens_c = sum(column)
                self.condprob[i][c] = 1.0*(self.nTokens_c + 1)/(self.sumTokens_c + self.nTokens)     
                i = i + 1
        return self

    def predict(self, X):
        self.y_tst = []
        self.score = np.zeros(3)
        # self.score = np.zeros(2)
        for row in X:
            for c in [0, 1, 2]:
            # for c in [0, 1]:
                self.score[c] = math.log(self.prior[c])
                for idx, val in enumerate(row):
                    if val != 0:
                        self.score[c] += math.log(self.condprob[idx][c])
            self.y_tst.append(np.argmax(self.score))
        return self.y_tst

    def score(self, X, y):  
        # hypMatrix = np.zeros((2,2))
        hypMatrix = np.zeros((3,3))

        result = self.predict(X)
        a = (result == y)

        for i, pol in enumerate(result):
            if pol == 0 and y[i] == 0:
                    hypMatrix[0][0] += 1
            elif pol == 0 and y[i] == 1:
                    hypMatrix[1][0] += 1
            elif pol == 0 and y[i] == 2:
                    hypMatrix[2][0] += 1
            elif pol == 1 and y[i] == 0:
                    hypMatrix[0][1] += 1
            elif pol == 1 and y[i] == 1:
                    hypMatrix[1][1] += 1
            elif pol == 1 and y[i] == 2:
                    hypMatrix[2][1] += 1
            elif pol == 2 and y[i] == 0:
                    hypMatrix[0][2] += 1
            elif pol == 2 and y[i] == 1:
                    hypMatrix[1][2] += 1
            elif pol == 2 and y[i] == 2:
                    hypMatrix[2][2] += 1
        # for i, pol in enumerate(result):
        #     if pol == 0 and y[i] == 0:
        #             hypMatrix[0][0] += 1
        #     elif pol == 0 and y[i] == 1:
        #             hypMatrix[1][0] += 1
        #     elif pol == 1 and y[i] == 0:
        #             hypMatrix[0][1] += 1
        #     elif pol == 1 and y[i] == 1:
        #             hypMatrix[1][1] += 1

        print hypMatrix
        return np.sum(a)/float(len(y))

# Count words in text
vocabulary, X = count_words(texts)

# Try to fit, predict and score
nb = NB()
nbL = MultinomialNB()


kf = KFold(n=len(y), n_folds = 5, shuffle=True)

for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# X_train = X[::2]
# y_train = y[::2]
# X_test = X[1::2]
# y_test = y[1::2]

lin_clf = LinearSVC()
lr_clf = LogisticRegression()

nb.fit(X_train, y_train)
nbL.fit(X_train, y_train)
lin_clf.fit(X_train, y_train)
lr_clf.fit(X_train, y_train)

print nb.score(X_test, y_test)
print nbL.score(X_test, y_test)
print np.mean(lin_clf.predict(X_test) == y_test)
print np.mean(lr_clf.predict(X_test) == y_test)

# Random permutation cross-validation iterator.Yields indices to split data into training and test sets.
# rs = cv.ShuffleSplit(len(y), n_iter = 15, test_size = .1, random_state = 0)
# for train_index, test_index in rs:
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
# nb.fit(X_train, y_train)
# print score(nb.predict(X_test), y_test)
# print nb.score(X_test, y_test)
