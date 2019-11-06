# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:44:04 2019

@author: ROHIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Importing the dataset
dataset = pd.read_csv('test.csv')
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
def N_est(X_train, X_test, y_train, y_test):
    N_est=[10,20,50,100,150,200,300]
    rms_test=[]
    rms_train=[]
    
    for i in N_est:
        regressor = RandomForestRegressor(n_estimators = 25,random_state = 0,max_depth=20,max_features=4,min_samples_leaf=10,min_samples_split=20)
        regressor.fit(X_train, y_train)
        
        tabel_test=np.zeros((len(X_test),2))
        # Predicting with testdata
        for i in range(len(X_test)):
            y_pred = regressor.predict(X_test[i].reshape(1,-1))
            tabel_test[i,0]=y_pred
            tabel_test[i,1]=y_test[i]
             
        rmstest = sqrt(mean_squared_error(tabel_test[:,1], tabel_test[:,0]))
        rms_test.append(rmstest)
        # Predicting with testdata
        tabel_train=np.zeros((len(X_test),2))
        for i in range(len(X_test)):
            y_pred = regressor.predict(X_train[i].reshape(1,-1))
            tabel_train[i,0]=y_pred
            tabel_train[i,1]=y_train[i]
             
        rmstrain = sqrt(mean_squared_error(tabel_train[:,1], tabel_train[:,0]))
        rms_train.append(rmstrain)
    
    
    plt.plot(N_est,rms_test, color = 'blue',label="test")
    plt.plot(N_est,rms_train, color = 'red',label="train")
    plt.title('N_estimator (Random Forest Regression)')
    plt.xlabel('n estimator')
    plt.ylabel('RMS error')
    plt.show()