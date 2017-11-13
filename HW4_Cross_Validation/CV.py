#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:53:23 2017

@author: sitibanc
"""
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


def K_fold_CV(k, data):
    subset_size = len(data) // k
    accuracy = 0
    for i in range(k):
        start = subset_size * k
        stop = subset_size * (k + 1)
        testing = data[start:stop]
        training = data[0:start] + data[stop:]
        # split features & target
        # gradiant boosting
        # calculate accuracy
        accuracy += metrics.accuracy_score()
    return accuracy / k


# Read CSV
raw = pd.read_csv('data.csv')
# Data Pre-Processing