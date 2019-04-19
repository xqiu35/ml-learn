import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


quandl.ApiConfig.api_key = "CQVM78g1Qisb7QCsmHgk"
df = quandl.get('WIKI/GOOGL', start_date="2018-12-31", end_date="2019-03-01")
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.0026*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
confidence = clf.score(X_test,y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set)


# Visualising Traning set results
#plt.scatter(X_train, y_train, color = 'red')
#plt.show()





