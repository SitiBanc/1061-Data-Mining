#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:53:23 2017

@author: sitibanc
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier


def K_fold_CV(k, data):
    subset_size = len(data) // k
    accuracy = 0
    for i in range(k):
        start = subset_size * i
        stop = subset_size * (i + 1)
        testing = data[start:stop]
        training = data[0:start]
        training = training.append(data[stop:])
        # split features & target
        testY, trainY = testing['income'], training['income']
        testX = testing.drop('income', axis = 1)
        trainX = training.drop('income', axis = 1)
        # gradiant boosting
        clf = GradientBoostingClassifier().fit(trainX, trainY)
        # calculate accuracy
        accuracy += clf.score(testX, testY)
        print('i =', i, ', accuracy =', clf.score(testX, testY))
    return accuracy / k


# =============================================================================
# Data Pre-Processing
# =============================================================================
# Read CSV
raw = pd.read_csv('data.csv')
# Drop Duplicate Columns
data = raw.drop('education', axis = 1)
# Replace '?' with np.nan
data = data.replace(to_replace = ' ?', value = np.NaN)
# Filling Missing Value
values = {
        'age':data['age'].mean(),
        'workclass':data['workclass'].mode()[0],
        'fnlwgt':data['fnlwgt'].mean(),
        'education_num':data['education_num'].mode()[0],
        'marital_status':data['marital_status'].mode()[0],
        'occupation':data['occupation'].mode()[0],
        'relationship':data['relationship'].mode()[0],
        'race':data['race'].mode()[0],
        'sex':data['sex'].mode()[0],
        'capital_gain':data['capital_gain'].mode()[0],
        'capital_loss':data['capital_loss'].mode()[0],
        'hours_per_week':data['hours_per_week'].mean(),
        'native_country':data['native_country'].mode()[0]
        }
data = data.fillna(value = values, inplace = True)
# Transfer Categorical Features
le = LabelEncoder()
cols = [
        'workclass',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'native_country',
        'income'
        ]
for i in range(len(cols)):
    idx = cols[i]
    tmp = pd.Series(le.fit_transform(data[idx]), name = idx)
    data = data.drop(idx, axis = 1)
    data = data.join(tmp)
# Drop Missing Value
data.dropna(inplace = True)
# =============================================================================
# Apply K-Fold Cross-Validation
# =============================================================================
avg_accuracy = K_fold_CV(10, data)
print('Average Accuracy of 10-fold Cross Validation:', avg_accuracy)