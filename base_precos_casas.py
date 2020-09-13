# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:52:18 2019

@author: hugos
"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treina, X_teste, y_treina, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_treina_poly = poly.fit_transform(X_treina)
X_teste_poly = poly.transform(X_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treina_poly, y_treina)
score = regressor.score(X_treina_poly, y_treina)

previsoes = regressor.predict(X_teste_poly)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)
