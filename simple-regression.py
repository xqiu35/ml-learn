# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:38:13 2019

@author: xqiu3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# prepare data
df = pd.read_csv('Salary_Data.csv')
df.dropna(inplace=True)
X = np.array(df['YearsExperience']).reshape(-1,1)
y = np.array(df['Salary'])

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# training
clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print(confidence)

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test);


# Visualising Traning set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, train_pred, color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, test_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()