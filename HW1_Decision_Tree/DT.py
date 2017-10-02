#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:49:39 2017

@author: sitibanc
"""
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

# Read Data
# 1.Read CSV File
raw_data = pd.read_csv('./character-deaths.csv')

# Data Preprocessing
# 2-1.Replace NaN with 0
filled_data = raw_data.fillna(0)
# 2-2.If 'Death Year' has a value set it to 1 and rename the column to 'is_Dead'
mask = filled_data['Death Year'] > 0
is_Dead = 1 * mask
is_Dead.name = 'is_Dead'
# 2-3.Transform 'Allegiances' to dummy variables
final_data = filled_data.join(pd.get_dummies(filled_data['Allegiances'], prefix = 'is'))
# Cleaning Redundant or Unused Attributes
final_data = final_data.drop('Name', 1)
final_data = final_data.drop('Book of Death', 1)
final_data = final_data.drop('Death Chapter', 1)
final_data = final_data.drop('Death Year', 1)
final_data = final_data.drop('Allegiances', 1)
# Spliting Training and Teating Data
trainX, testX, trainY, testY = train_test_split(final_data, is_Dead, test_size = 0.25)

# Build Model
# 3.Build decision tree
clf = tree.DecisionTreeClassifier(max_depth = 8)
clf = clf.fit(trainX, trainY)

# Calculate Scores
# 4.Calculate the precision rate, recall rate, accuracy
testY_predicted = clf.predict(testX)
precision = metrics.precision_score(testY, testY_predicted)
recall = metrics.recall_score(testY, testY_predicted)
accuracy = metrics.accuracy_score(testY, testY_predicted)
print('Precision Rate:', precision * 100, '%')
print('Recall Rate:', recall * 100, '%')
print('Accuracy:', accuracy * 100, '%')

# Data Visualization
# 5.Generate Tree Image
target_names = ['Alive', 'Dead']
result = tree.export_graphviz(clf, out_file = None, feature_names = testX.columns, class_names = target_names, filled = True, rounded = True, special_characters = True)  
graph = graphviz.Source(result)  
graph.render('DT_result', view = True)