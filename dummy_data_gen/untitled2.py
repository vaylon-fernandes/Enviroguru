# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 02:42:24 2019

@author: ROHIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('nitrogen_dioxide.csv')
X = dataset.iloc[:,:].values

long=np.zeros((100,1))
lat=np.zeros((100,1))

for i in range(100):
        x=X[i,-2]
        y=x.split(',')
        long[i]=float(y[0])
        lat[i]=float(y[1])

X_mod=np.concatenate((X,long),axis=1)
X_mod=np.concatenate((X_mod,lat),axis=1)

X_mod=np.delete(X_mod,6,1)

#remove row and col index
X_mod=np.delete(X_mod,0,1)
X_mod=np.delete(X_mod,0,0)

df = pd.DataFrame(X_mod) 
  
# saving the dataframe 
df.to_csv('file1.csv')