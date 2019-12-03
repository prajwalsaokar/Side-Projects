# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 23:01:30 2018

@author: saoka
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm



# Importing the dataset
dataset= pd.read_table('iris.txt')
features=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
X = dataset[features]
y = dataset['Class']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# classifier here
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
classifier= SVC()

# Fitting and Predicting
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluating the dataset
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print ('Accuracy = %.3f' %(accuracy_score(y_test, y_pred)))
print(cm)

