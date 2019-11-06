# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:23:14 2019

@author: ROHIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n=100
x=np.zeros((n,2))

for i in range(n):
    x[i,1]=np.random.normal(5,2)
    x[i,0]=i
    
plt.plot(x[:,0],x[:,1])
plt.show()