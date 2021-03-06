#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:43:23 2017

@author: sitibanc
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Read Data
raw_data = pd.read_csv('nyc-rolling-sales.csv', encoding = 'iso-8859-1')

# Data Preprocessing
# Drop unused columns
data = raw_data.drop(raw_data.columns[[0]], axis = 1)
data = data.drop('EASE-MENT', axis = 1)
data = data.drop('APARTMENT NUMBER', axis = 1)
data = data.drop('SALE DATE', axis = 1)
data = data.drop('ADDRESS', axis = 1)

# Drop missing target value & outliers
data = data[data['SALE PRICE'] != ' -  ']
tmp = pd.to_numeric(data['SALE PRICE'], downcast = 'signed')
for i in tmp.index:
    if tmp[i] <= 10:
        data = data.drop(i)
        
# Filling missing value
# Fill 'BUILDING CLASS AT PRESENT' with mode
tmp = data['BUILDING CLASS AT PRESENT']
tmp = tmp.replace(to_replace = ' ', value = np.nan)
tmp = tmp.fillna(data['BUILDING CLASS AT PRESENT'].mode())
data = data.drop('BUILDING CLASS AT PRESENT', axis = 1)
data = data.join(tmp)

# Fill 'TAX CLASS AT PRESENT' with mode
tmp = data['TAX CLASS AT PRESENT']
tmp = tmp.replace(to_replace = ' ', value = np.nan)
tmp = tmp.fillna(data['BUILDING CLASS AT PRESENT'].mode())
data = data.drop('TAX CLASS AT PRESENT', axis = 1)
data = data.join(tmp)

# Fill 'LAND SQUARE FEET' with mean
tmp  = data['LAND SQUARE FEET']
tmp = tmp.replace(to_replace = ' -  ', value = np.nan)
tmp = pd.to_numeric(tmp, downcast='signed')
tmp = tmp.fillna(tmp.mean())
data = data.drop('LAND SQUARE FEET', axis = 1)
data = data.join(tmp)

# Fill 'GROSS SQUARE FEET' with mean
tmp = data['GROSS SQUARE FEET']
tmp = tmp.replace(to_replace = ' -  ', value = np.nan)
tmp = pd.to_numeric(tmp, downcast='signed')
tmp = tmp.fillna(tmp.mean())
data = data.drop('GROSS SQUARE FEET', axis = 1)
data = data.join(tmp)

data = data.dropna()

# Transfer categorical variable
le = preprocessing.LabelEncoder()
col = ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT PRESENT']
for i in range(len(col)):
    idx = col[i]
    le.fit(data[idx])
    tmp = pd.Series(le.transform(data[idx]), name = idx)
    data = data.drop(idx, axis = 1)
    data = data.join(tmp)

# Output Data
data.to_csv('./nyc-rolling-sales-processed.csv')
