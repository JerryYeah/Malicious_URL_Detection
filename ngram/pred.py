from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
from sklearn.linear_model import LogisticRegression
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix


def getTokens(input):
    temp = str(input)
    ngrams = set()
    for i in range(len(temp)-2):
        ngrams.add(temp[i:i+3])
    ngrams = list(ngrams) 
    return ngrams

def TL():
    allurls = '../data/data.csv' #path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False) #reading file
    allurlsdata = pd.DataFrame(allurlscsv)      #converting to a dataframe

    allurlsdata = np.array(allurlsdata) #converting it into an array
    random.shuffle(allurlsdata) #shuffling

    #test = allurlsdata[:500]
    #with open('test_set', 'wb') as input:
    #    pickle.dump(test, input)

    y = [d[1] for d in allurlsdata]     #all labels 
    corpus = [d[0] for d in allurlsdata]        #all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)   #get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)        #get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #split into training and testing set 80/20 ratio

    lgs = LogisticRegression(max_iter=10000)    #using logistic regression
    lgs.fit(X_train, y_train)
    print(lgs.score(X_test, y_test))    #pring the score. It comes out to be 98%
    return vectorizer, lgs

'''
vectorizer, lgs = TL()
with open('lgs.pickle', 'wb') as model:
        pickle.dump(lgs, model)
with open('vct', 'wb') as vct:
        pickle.dump(vectorizer, vct)
print('Finished writing models')
'''
with open('lgs.pickle', 'rb') as model:
    lgs = pickle.load(model)
with open('vct', 'rb') as vct:
    vectorizer = pickle.load(vct)

testurls = '../data/test.csv'
testurlscsv = pd.read_csv(testurls,',',error_bad_lines=False) #reading file
testurlsdata = pd.DataFrame(testurlscsv)      #converting to a dataframe
testurlsdata = np.array(testurlsdata) #converting it into an array

y = [d[1] for d in testurlsdata]     #all labels
x = [d[0] for d in testurlsdata]

x = vectorizer.transform(x)
y_pred = lgs.predict(x)
#labels = ['good', 'bad']
matrix = confusion_matrix(y, y_pred, labels=["good", "bad"])
print(matrix)

