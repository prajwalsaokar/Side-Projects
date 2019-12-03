# -*- coding: utf-8 -*-

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
features=['Dependents','Education','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History','ApplicantIncome','CoapplicantIncome','Property_Area']
# Importing the training set
training_set = pd.read_csv('train.csv')
X_train= training_set[features]
y_train=training_set['Loan_Status']
#Importing the test set
test_set = pd.read_csv('test.csv')  
X_test = test_set[features]
# Splitting the dataset into the Training set and Test set

'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)'''
#Data Exploration
training_set.head()
training_set.tail()
print(training_set.describe())
print(training_set.groupby('Gender').size())
test_set.isnull().sum()
training_set.isnull().sum()
print(training_set.groupby('Dependents').size())
print(training_set.groupby('Self_Employed').size())
print(test_set.groupby('Self_Employed').size())
print(training_set.groupby('LoanAmount').size())
print(training_set.groupby('Loan_Amount_Term').size())
print(training_set.groupby('Credit_History').size())


#Cleaning up the data
training_set['Dependents'].fillna('0', inplace=True)
test_set['Dependents'].fillna('0', inplace=True)
training_set['Self_Employed'].fillna('0', inplace=True)
test_set['Self_Employed'].fillna('0', inplace=True)
training_set['LoanAmount'].fillna('136', inplace=True)
test_set['LoanAmount'].fillna('136', inplace=True)
training_set['Loan_Amount_Term'].fillna('360', inplace=True)
test_set['Loan_Amount_Term'].fillna('360', inplace=True)
training_set['Credit_History'].fillna('1', inplace=True)
test_set['Credit_History'].fillna('1', inplace=True)
# Encoding the Categorical Variables
cat_features=['Gender','Married','Dependents','Education','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train = le.fit_transform(X_train)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit_transform(X_train[cat_features])
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Creating and fitting the classifier, possible with K Cross Fold
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
#Pickling the Estimators
import _pickle as Pickle
import joblib
joblib.dump(classifier, 'estimator.pkl', compress=0)
with open("estimator.pkl", "wb") as model: Pickle.dump(classifier, model)
with open("estimator.pkl", "rb") as model: thenewmodel = Pickle.load(model)