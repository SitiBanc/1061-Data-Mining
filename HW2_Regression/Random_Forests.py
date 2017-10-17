#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:50:28 2017

@author: sitibanc
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Read Data
data = pd.read_csv('nyc-rolling-sales-processed.csv')
data = data.drop(data.columns[[0]], axis = 1)

# Split Features & Targets
targets = data['SALE PRICE']
features = data.drop('SALE PRICE', axis = 1)

# Split Training & Testing Data
trainX, testX, trainY, testY = train_test_split(targets, features, test_size = 0.25)