# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:05:26 2019

@author: xqiu3
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

# pre-processing
X = df['Level'].values
y = df['Salary'].values
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X.reshape(-1,1))


clf = LinearRegression()
clf.fit(X_poly,y)


# Visualize
plt.scatter(X,clf.predict(X_poly),color='blue')
plt.plot(X,y, color='red')
