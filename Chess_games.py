
# coding: utf-8

# In[12]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[13]:

chess = pd.read_csv('games.csv')
chess.columns


# In[14]:

mapping = {'white':1, 'black':2, 'draw':3}
chess.replace({'winner':mapping}, inplace = True)
chess.columns


# Preprocessing: change all non-numeric values to numeric 

# In[15]:

l=pd.factorize(chess['moves'])
chess['moves'] = l[0]

l=pd.factorize(chess['opening_name'])
chess['opening_name'] = l[0]

l = pd.factorize(chess['opening_eco'])
chess['opening_eco'] = l[0]

l = pd.factorize(chess['victory_status'])
chess['victory_status'] = l[0]

b = []
for a in chess['increment_code']:
    l = a.split('+')
    b.append(int(l[0])*60 + int(l[1]))
    
chess['increment_code'] = b


# In[16]:

chess['elapse'] = chess['last_move_at']-chess['created_at']
chess.columns


# In[17]:

cols = ['created_at','last_move_at','turns','white_rating','black_rating','opening_ply','opening_eco']
#y = chess['victory_status']
y = chess['winner']


# In[21]:

cols1 = ['turns', 'increment_code', 'white_rating', 'black_rating', 'moves', 'opening_eco', 'opening_name', 'opening_ply',
       'elapse']
X1 = chess[cols1]
cols2 = ['increment_code', 'white_rating', 'black_rating', 'moves', 'opening_eco', 'opening_ply', 'elapse']
X2 = chess[cols2]
cols3 = ['increment_code', 'white_rating', 'black_rating', 'elapse']
X3 = chess[cols3]
cols4 = ['white_rating', 'black_rating']
X4 = chess[cols4]
X = X4


# In[38]:

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X1,y, test_size = 0.2, random_state = 0)

test1 = []
train1 = []
test2 = []
train2 = []
for n in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    train1.append(knn.score(X_train, y_train))
    test1.append(knn.score(X_test, y_test))
    knn.fit(X2_train, y2_train)
    train2.append(knn.score(X2_train, y2_train))
    test2.append(knn.score(X2_test, y2_test))
 
plt.scatter(range(1,50), test1, c='r')
plt.scatter(range(1,50), test2, c='b')
plt.show()

