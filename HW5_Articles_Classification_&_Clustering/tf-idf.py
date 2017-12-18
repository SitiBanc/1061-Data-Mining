#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:23:47 2017

@author: sitibanc
"""
import pandas as pd
from jieba import analyse
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
    seg_list = analyse.extract_tags(data['postContent'][i])
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

# =============================================================================
# Classification (using Decision Tree)
# =============================================================================
