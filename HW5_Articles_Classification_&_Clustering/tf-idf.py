#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:23:47 2017

@author: sitibanc
"""
import pandas as pd
from jieba import analyse
from sklearn.cluster import KMeans
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# Data Preprocessing
# =============================================================================
# Read Data
data = pd.read_excel('FDATA.xlsx')
corpus = [0] * len(data)
label = data['mainTag']
analyse.set_stop_words('stop_words.txt')
# Cutting Articles
for i in range(len(data)):
    seg_list = analyse.extract_tags(data['postContent'][i], topK = 30)
    datastr = ''
    for t in seg_list:
        datastr += t + ' '
    corpus[i] = datastr
# Vectorize & Calculate tf-idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# =============================================================================
# Clustering (K-Means, k = 5)
# =============================================================================
kmeans = KMeans(5).fit(X)
# Print Out Result
print('K-Means Clustering Result:')
print(kmeans.labels_)
# =============================================================================
# Classification (using Decision Tree)
# =============================================================================
# Spliting Training and Teating Data
trainX, testX, trainY, testY = train_test_split(X, label, test_size = 0.25)
# Build decision tree
clf = tree.DecisionTreeClassifier(max_depth = 10)
clf = clf.fit(trainX, trainY)
# Calculate the accuracy
testY_predicted = clf.predict(testX)
accuracy = metrics.accuracy_score(testY, testY_predicted)
# Print Out Result
print('\nDecision Tree Classification Result:')
print('Accuracy:', accuracy * 100, '%')