#import sys
#print(sys.version)
import pandas as pd
import quandl
import math
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
#pd.options.mode.chained_assignment = None  # default='warn'

df = quandl.get('WIKI/GOOGL')
df.head()

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100.0
df['PCT_daily_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0

df1 = df[['Adj. Close', 'HL_PCT', 'PCT_daily_change', 'Adj. Volume']]
df1.head()

forecast_col = 'Adj. Close'
df1.fillna(-99999, inplace = True)

x1 = 0.01*len(df1)
forecast_out = int(math.ceil(x1))
df1['label'] = df[forecast_col].shift(-forecast_out)
df1.dropna(inplace = True)
df1.head()
df1.tail()

X = np.array(df1.drop(['label'],1))
y = np.array(df1['label'])

X = preprocessing.scale(X)
y  = np.array(df1['label'])
#print(len(X), len(y)) checking if length of both arrays i.e. x and y are same

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
#clf = svm.SVR() #support vector regression
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)