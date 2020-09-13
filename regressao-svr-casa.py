# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:22:28 2019

@author: hugos
"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_treina, X_teste, y_treina, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_treina, y_treina)
score = regressor.score(X_treina, y_treina)

regressor.score(X_teste, y_teste)

previsoes = regressor.predict(X_teste)
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

