# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:53:14 2017

@author: Akshay
"""


import pandas as pd

import numpy as num

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

chess = pd.read_csv('games.csv')

chess

x = fruits[['mass', 'width', 'height', 'color_score']]

y = fruits['fruit_label']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 0)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

knn.score(x_train,y_train)