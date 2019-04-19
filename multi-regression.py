
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('50_Startups.csv')

# pre-processing
le = LabelEncoder();
le.fit(df['State'])
df['State'] = le.transform(df['State'])

X = df.drop(['Profit'],axis=1).values
dd = dim(X)
y = df['Profit'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# training
clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test,y_test)
print(confidence)

