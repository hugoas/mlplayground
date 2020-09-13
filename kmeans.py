# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:27:16 2019

@author: hugos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = [20, 27, 37, 50, 55]
y = [1000, 1200, 2000, 2100, 3400]
plt.scatter(X,y)

base = np.array([[20, 1000], [27, 1200], [37,2000],
                [50, 2100], [55, 3400]])

scaler = StandardScaler()
base = scaler.fit_transform(base)

kmmeans = KMeans(n_clusters=3)
kmmeans.fit(base)

centroides = kmmeans.cluster_centers_
rotulos = kmmeans.labels_

cores = ["g.", "r.", "b."]
for i in range(len(X)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 5)
plt.scatter(centroides[:,0], centroides[:,1], marker="x")