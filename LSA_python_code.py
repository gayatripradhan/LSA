# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:18:42 2017

@author: Gayatri Pradhan
"""

import nltk
from nltk.corpus import stopwords
import re
from string import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pandas as pd
import string
import numpy as np
 
stemmer = PorterStemmer()
 
dataFile = open("D:/RBL/data/test.txt")
data = dataFile.read()
data = data.split("\n")  
 
texts=[]
for i in range(len(data)):
    texts.append(data[i])
    texts[i] = texts[i].translate(string.punctuation).lower()
    texts[i] = nltk.word_tokenize(texts[i])
    texts[i] = [stemmer.stem(word) for word in texts[i] if not word in stopwords.words('english')]
#    texts[i] = join(texts[i]," ")
    texts[i] = "".join(texts[i])
 
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(texts)

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
lsa = svd.fit_transform(tfidf.T)

def getClosestTerm(term,transformer,model):
 
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]      
 
    model = np.dot(model,model.T)
    searchSpace =np.concatenate( (model[index][:index] , model[index][(index+1):]) )  
 
    out = np.argmax(searchSpace)
 
    if out<index:
        return transformer.get_feature_names()[out]
    else:
        return transformer.get_feature_names()[(out+1)]
 
def kClosestTerms(k,term,transformer,model):
 
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]
 
    model = np.dot(model,model.T)
 
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[transformer.get_feature_names()[i]] = model[index][i]
 
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
 
    return sortedList[::-1][0:k]
