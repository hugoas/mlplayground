# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:41:15 2019

@author: hugos
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples = 200, centers = 4)
plt.scatter(x[:,0], x[:,1])

kmeans = KMeans(n_clusters = 6)
kmeans.fit(x)

previsoes = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c = previsoes)