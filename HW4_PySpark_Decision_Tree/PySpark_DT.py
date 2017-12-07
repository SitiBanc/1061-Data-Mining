#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:59:54 2017

@author: sitibanc
"""
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql import functions as F
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics


def convertColumn(df, colNames, newType):
    for name in colNames:
        df = df.withColumn(name, df[name].astype(newType))
    return df


def parseDF(df):
    parsedData = []
    for row in df.collect():
        parsedData.append(LabeledPoint(row[-1], list(row[:-1])))
    return parsedData


# Read Data
df = spark.read.csv('1061-Data-Mining/HW1_Decision_Tree/character-deaths.csv', header=True)

# Drop Unneeded Columns
df = df.drop('Name', 'Book of Death', 'Death Chapter')

# Fill null value with 0
df = df.fillna({'Death Year': 0, 'Book Intro Chapter': 0, 'Gender': 0, 'Nobility': 0, 'GoT': 0, 'CoK': 0, 'SoS': 0, 'FfC': 0, 'DwD': 0})

# Generate is_Dead columns
isDead_expr = [F.when(F.col('Death Year') > 0, 1).otherwise(0).alias('is_Dead')]
df = df.select('Allegiances', 'Book Intro Chapter', 'Gender', 'Nobility', 'GoT', 'CoK', 'SoS', 'FfC', 'DwD', * isDead_expr)

# Convert columns types
df = convertColumn(df, df.columns[2:], IntegerType())
df = df.withColumn(df.columns[1], df[df.columns[1]].astype(FloatType()))

# Get Dummies
allegiances = df.select('Allegiances').distinct().rdd.flatMap(lambda x: x).collect()
allegiances_expr = [F.when(F.col('Allegiances') == a, 1).otherwise(0).alias('is_' + a) for a in allegiances]
df1 = df.select('Book Intro Chapter', 'Gender', 'Nobility', 'GoT', 'CoK', 'SoS', 'FfC', 'DwD', * allegiances_expr, 'is_Dead')

# Split Training & Testing datasets
trainData, testData = df1.randomSplit(weights=[0.75, 0.25])
df1.count() == trainData.count() + testData.count()

# Data Parsing
train = parseDF(trainData)
test = parseDF(testData)

# Training
DTmodel = DecisionTree.trainClassifier(data = sc.parallelize(train), numClasses = 2, categoricalFeaturesInfo = {}, maxDepth = 8)

# Get Predictions
testRDD = sc.parallelize(test)
predictions = DTmodel.predict(testRDD.map(lambda x: x.features))
labels = sc.parallelize(testRDD.map(lambda y: y.label).collect())

# Constructing Confusion Matrix
predictionAndLabels = predictions.zip(labels)

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)
# Calculate accuracy, recall & precision
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
accuracy = metrics.accuracy

# Print Out Results
print('Precision:\t', precision * 100, '%')
print('Recall:\t\t', recall * 100, '%')
print('Accuracy:\t', accuracy * 100, '%')