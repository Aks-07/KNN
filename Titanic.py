
# In[]: import all packages and libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[]: read data

titanic = pd.read_csv('train.csv')


# In[] drop the Name and Cabin columns since we won't be using them

titanic.drop('Cabin',inplace = True,axis = 1)
titanic.drop('Name',inplace = True,axis = 1)

# In[] check the number of records and columns in the data 

titanic.columns
titanic.shape
titanic.describe()

# In[]: categorizing the sex variable into numeric

mapping = {'male':1, 'female':2}
titanic.replace({'Sex':mapping}, inplace = True)
titanic.columns

# In[]: making the variables numeric
    
l=pd.factorize(titanic['Ticket'])
titanic['Ticket'] = l[0]

l = pd.factorize(titanic['Embarked'])
titanic['Embarked'] = l[0]


# In[] make a copy of the data where NA values of age are replaced by mean of the entire data # Model 1

titanic_mean = titanic.copy()
titanic_mean.describe()

#mean substitution

titanic_mean.fillna(titanic_mean.mean(),inplace = True)

titanic_mean.shape

titanic_mean.describe()

# In[] make a copy of the data where NA values are dropped #  Model 2

titanic_drop = titanic.copy()
titanic_drop.describe()

# drop rows with missing values

titanic_drop.dropna(inplace=True)

print(titanic_drop.shape)


# In[] make a copy of the data where NA values are replaced by mean depending upon class and sex # Model 3

titanic_mean_age = titanic.copy()

titanic_mean_age.fillna(titanic_mean_age.groupby(['Pclass', 'Sex'], as_index=False).mean())

titanic_mean_age.describe()


# In[]: defining columns for x and y

cols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

y = titanic_mean['Survived']


# In[]: check different models with different combination of features 

columns1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
X1 = titanic_mean[columns1]

columns2 = ['Pclass', 'Sex','Age', 'SibSp', 'Fare', 'Embarked']
X2 = titanic_mean[columns2]

columns3 = ['Pclass','Sex', 'Age','SibSp', 'Ticket', 'Fare']
X3 = titanic_mean[columns3]

columns4 = ['Pclass', 'Sex', 'Age', 'Parch', 'Ticket','Embarked']
X4 = titanic_mean[columns4]

columns5 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X5 = titanic_mean[columns5]


# In[]: scale the data and check the accuracy of the models via scatter matrix

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y, test_size = 0.2, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y, test_size = 0.2, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y, test_size = 0.2, random_state = 0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4,y, test_size = 0.2, random_state = 0)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5,y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)
X3_train = scaler.fit_transform(X3_train)
X3_test = scaler.transform(X3_test)
X4_train = scaler.fit_transform(X4_train)
X4_test = scaler.transform(X4_test)
X5_train = scaler.fit_transform(X5_train)
X5_test = scaler.transform(X5_test)


test1 = []
train1 = []
test2 = []
train2 = []
test3 = []
train3 = []
test4 = []
train4 = []
test5 = []
train5 = []

for n in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X1_train, y1_train)
    train1.append(knn.score(X1_train, y1_train))
    test1.append(knn.score(X1_test, y1_test))
    knn.fit(X2_train, y2_train)
    train2.append(knn.score(X2_train, y2_train))
    test2.append(knn.score(X2_test, y2_test))
    knn.fit(X3_train, y3_train)
    train3.append(knn.score(X3_train, y3_train))
    test3.append(knn.score(X3_test, y3_test))
    knn.fit(X4_train, y4_train)
    train4.append(knn.score(X4_train, y4_train))
    test4.append(knn.score(X4_test, y4_test))
    knn.fit(X5_train, y5_train)
    train5.append(knn.score(X5_train, y5_train))
    test5.append(knn.score(X5_test, y5_test))
 
plt.scatter(range(1,50), test1, c='r')
plt.scatter(range(1,50), test2, c='b')
plt.scatter(range(1,50), test3, c='g')
plt.scatter(range(1,50), test4, c='y')
plt.scatter(range(1,50), test5, c='p')
plt.show()


# In[ ]: once you have selected the model, perform the same preprocessing on test dataset

titanic_test = pd.read_csv('test.csv')

titanic_test.drop('Cabin',inplace = True,axis = 1)
titanic_test.drop('Name',inplace = True,axis = 1)

titanic_test.fillna(titanic_mean.mean(),inplace = True)

titanic_test.shape

titanic_test.describe()

mapping = {'male':1, 'female':2}
titanic_test.replace({'Sex':mapping}, inplace = True)
titanic_test.columns

l=pd.factorize(titanic_test['Ticket'])
titanic_test['Ticket'] = l[0]

l = pd.factorize(titanic_test['Embarked'])
titanic_test['Embarked'] = l[0]


# In[] we select the apt model because of its consistency in accuracy and select the features corresponding to it ( in this case, the blue one)
  
columns3 = ['Pclass', 'Sex','Age', 'SibSp', 'Ticket', 'Fare']
X3 = titanic_test[columns3]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X3_scaled = scaler.fit_transform(X3)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X3_train, y3_train)
pred = knn.predict(X3_scaled)

print(pred)

# In[] Check the output

titanic_test['Survived'] = pred

print(titanic_test[['PassengerId','Survived']]) 

# In[] Create a data frame with two columns and write to a csv file
    
PassengerId = np.array(titanic_test["PassengerId"]).astype(int)
Project1 = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
print(Project1)

Project1.to_csv("Project1.csv", index_label = ["PassengerId"])
