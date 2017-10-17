#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:50:28 2017

@author: sitibanc
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Read Data
rdata = pd.read_csv('nyc-rolling-sales-processed.csv')
rdata = rdata.dropna()
data = rdata.drop(rdata.columns[[0]], axis = 1)


# Improve Model By Dropping Columns
#data = data.drop('BOROUGH', axis = 1)
data = data.drop('NEIGHBORHOOD', axis = 1)
data = data.drop('BLOCK', axis = 1)
#data = data.drop('LOT', axis = 1)
data = data.drop('ZIP CODE', axis = 1)
data = data.drop('RESIDENTIAL UNITS', axis = 1)
#data = data.drop('COMMERCIAL UNITS', axis = 1)
data = data.drop('TOTAL UNITS', axis = 1)
#data = data.drop('YEAR BUILT', axis = 1)
#data = data.drop('TAX CLASS AT TIME OF SALE', axis = 1)
#data = data.drop('TAX CLASS AT PRESENT', axis = 1)
data = data.drop('BUILDING CLASS CATEGORY', axis = 1)
#data = data.drop('BUILDING CLASS AT TIME OF SALE', axis = 1)
#data = data.drop('BUILDING CLASS AT PRESENT', axis = 1)
data = data.drop('LAND SQUARE FEET', axis = 1)
#data = data.drop('GROSS SQUARE FEET', axis = 1)


# Split Training & Testing Data
train = data[:int(len(data) * 0.95)]
test = data[int(len(data) * 0.95):]

# Split Features & Targets
trainY = train['SALE PRICE']
trainX = train.drop('SALE PRICE', axis = 1)
testY = test['SALE PRICE']
testX = test.drop('SALE PRICE', axis = 1)

# Create Random Forest Regressor object
regr = RandomForestRegressor()

# Train the model using the training sets
regr.fit(trainX, trainY)

# Make predictions using the testing set
predY = regr.predict(testX)

# Calculate Mean Absolute Error
MAE = mean_absolute_error(testY, predY)
print('MAE:', MAE)