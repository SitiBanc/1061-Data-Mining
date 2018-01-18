# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:34:02 2017

@author: user
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# 1 read data
df_2015 = pd.DataFrame(pd.read_csv('the-counted-2015.csv'))
df_2016 = pd.DataFrame(pd.read_csv('the-counted-2016.csv'))
df = pd.concat([df_2015, df_2016], ignore_index = True)

# 2 處理資料
df = df.drop(['uid', 'name', 'day', 'year', 'streetaddress'], 1)
df = df.drop(df.index[np.where((df['raceethnicity'] == 'Unknown') | (df['age'] == 'Unknown') | (df['armed'] == 'Unknown') | (df['lawenforcementagency'] == 'Unknown'))])

# replace data to int
df = df.replace(['Male', 'Female', 'Non-conforming', '40s'], [0, 1, 0, 40])
df['month'] = df['month'].replace(list(Counter(df['month'])), np.arange(len(Counter(df['month']))) + 1)
df['city'] = df['city'].replace(list(Counter(df['city'])), np.arange(len(Counter(df['city']))))
df['state'] = df['state'].replace(list(Counter(df['state'])), np.arange(len(Counter(df['state']))))
df['classification'] = df['classification'].replace(list(Counter(df['classification'])), np.arange(len(Counter(df['classification']))))
df['lawenforcementagency'] = df['lawenforcementagency'].replace(list(Counter(df['lawenforcementagency'])), np.arange(len(Counter(df['lawenforcementagency']))))
df['armed'] = df['armed'].replace(list(Counter(df['armed'])), np.arange(len(Counter(df['armed']))))

# dummy variables
#dum_city = pd.get_dummies(df['city']) # 1178
#dum_state = pd.get_dummies(df['state']) # 51
#dum_agency = pd.get_dummies(df['lawenforcementagency']) # 1208
#dum_armed = pd.get_dummies(df['armed']) # 7
#df = pd.concat([df, dum_city, dum_state, dum_agency, dum_armed], axis = 1)
#df = df.drop(['city', 'state','lawenforcementagency', 'armed'], 1)

# target value
cls = df['raceethnicity']
df = df.drop(['raceethnicity'], 1)
df.insert(loc = 0, column = 'raceethnicity', value = cls)

# save as csv
df.to_csv('out_police.csv')

# 3 split train, test data
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size = int(len(df)*0.25))

# 4 train a model
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5), n_estimators = 100)
clf.fit(x_train, y_train)
importances = clf.feature_importances_
predict = clf.predict(x_test)

# 5 calaulate accuracy, precision, recall
accuracy = metrics.accuracy_score(y_test, predict)
print('Accuracy:', accuracy)

# 6 plot 20 important features
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
feature = []
for i in range(len(indices)):
    feature.append(list(df)[indices[i]+1])
plt.bar(range(len(indices)), importances[indices], yerr = std[indices], align = 'center')
plt.title('Feature importances')
plt.xticks(range(len(indices)), feature, rotation = 'vertical')
plt.xlim([-1, len(indices)])
plt.show()

#plt.title('Feature importances')
#plt.bar(range(20), importances[indices[1:21]], yerr=std[indices[1:21]], align = 'center')
#plt.xticks(range(20), feature, rotation = 'vertical')
#plt.xlim([-1, 20])