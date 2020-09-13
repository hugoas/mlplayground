# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:52:12 2019

@author: hugos
"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 5:6].values
y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treina, X_teste, y_treina, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treina, y_treina)
score = regressor.score(X_treina, y_treina)

import matplotlib.pyplot as plt
plt.scatter(X_treina, y_treina)
plt.plot(X_treina, regressor.predict(X_treina), color = 'orange')

previsoes = regressor.predict(X_teste)

resultado = y_teste - previsoes
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor. predic(X_teste), color = 'red')

regressor.score(X_teste, y_teste)