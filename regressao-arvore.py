# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:20:44 2019

@author: hugos
"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treina, X_teste, y_treina, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor() 
regressor.fit(X_treina, y_treina)
score = regressor.score(X_treina, y_treina)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

regressor.score(X_teste, y_teste)