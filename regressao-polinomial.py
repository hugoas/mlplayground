# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:40:47 2019

@author: hugos
"""

import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

regressor1.predict(40)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor1.predict(X), color = 'red')
plt.title('Regressão Linear')
plt.xlabel('Idade')
plt.ylabel('Custo')

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
score2 = regressor2.score(X_poly, y)

regressor2.predict(poly.transform(40))

#import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Regressão Polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')