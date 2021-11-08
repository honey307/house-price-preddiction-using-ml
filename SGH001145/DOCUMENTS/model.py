#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python38-32\python.exe
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:34:24 2020

@author: Honey
"""
# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import cgitb
cgitb.enable()
import cgi
fs = cgi.FieldStorage()
print ("Content-Type: text/plain;charset=utf-8")
print ()

# Importing the dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1 ].values

#ENcoding train categorical Data

from sklearn.preprocessing import LabelEncoder
labelencoder_x= LabelEncoder()
X[:,1]=labelencoder_x.fit_transform(X[:,1])
labelencoder_y= LabelEncoder()
X[:,2]=labelencoder_y.fit_transform(X[:,2])
labelencoder_z= LabelEncoder()
X[:,3]=labelencoder_z.fit_transform(X[:,3])
labelencoder_x1= LabelEncoder()
X[:,5]=labelencoder_x1.fit_transform(X[:,5])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 43)
regressor.fit(X, y)
#print(regressor.score(X_train,y_train))
#print(regressor.score(X_test,y_test))

# input
#a=input() #BHK
#b=input() #TYPE
#c=input() #AREA
#d=input() #C_STATUS
#e=input() #SQ_FT
#f=input() #CITY
a = fs['bhk'].value
b = fs['type'].value
c = fs['area'].value
d = fs['c_status'].value
e = fs['sq_ft'].value
f = fs['city'].value

# Transform into array in integer form
b1=labelencoder_x.transform([b])
c1=labelencoder_y.transform([c])
d1=labelencoder_z.transform([d])
f1=labelencoder_x1.transform([f])


new_input=[[a,b1[-1],c1[-1],d1[-1],e,f1[-1]]]
y_pred = regressor.predict(new_input)

print(y_pred[-1])