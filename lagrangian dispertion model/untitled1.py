# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:31:01 2019

@author: ROHIT
"""
import numpy as np

def traj(x,y,z):
    Xp=Xp0+deltaT*(u_mean+u_eddi)
    Yp=Yp0+deltaT*(v_mean+v_eddi)
    Zp=Zp0+deltaT*(w_mean+w_eddi)
    
    return [Xp,Yp,Zp]

u_mean=v_mean=w_mean=2;


arr=np.zeros((1000,3),dtype='int32')

for i in range(1000):
    u_mean=np.random.normal()
    v_mean=np.random.normal()
    w_mean=np.random.normal()
    arr[i]=traj()    
    
