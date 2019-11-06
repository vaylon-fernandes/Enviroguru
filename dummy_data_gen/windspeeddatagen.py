# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:54:12 2019

@author: ROHIT
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 04:02:45 2019

@author: ROHIT
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#const-def
date_day=[1,30]
date_month=[1,12]
date_yr=[2017,2018,2019,2020]
wind_speed=[3,6]
#wind_dir=["NE","NW","SE","SW","N","S","E","W"]-mapped as...
wind_dir=[0,1,2,3,4,5,6,7]

location=[[1,2],[2,4],[3,8],[4,6],[5,3],[6,9],[7,2],[8,4],[9,1]]

#rows and cols
features=10
n=5000
X=np.zeros((n,features))

#date stuff
day=1
month=1
yr=2017
hr=0

season=['rain','summer','winter']

temp_min=[28,30,37,45,52,59,64,64,55,43,32,28]
temp_max=[41,46,57,66,75,84,86,86,82,72,59,48]

temp=np.zeros((12))

for i in range(12):
    temp[i]=((temp_min[i]+temp_max[i])/2)
temp.sort()


#cal slope for maping
maxT=28561
minT=283.3
maxQ=13.5
minQ=8.2
m=(maxQ-minQ)/(maxT-minT)
# output = output_start + m * (input - input_start)


for i in range(n):
    X[i,5]=day                    #day
    
    X[i,6]=month                  #month
    
    X[i,7]=yr                   #year
    day+=1
    if(day > 31):
        day=1
        month+=1
        if(month > 12):
            month=1
            yr+=1
    
    X[i,8]=hr                   #hr
    hr=hr+1
    if(hr>24):
        hr=0
            
    
    if(month>=3 and month <=5):
        season_temp='summer'
        token1=5
    elif(month>=6 and month <=9):
        season_temp='rain'
        token1=10
    else:
        season_temp='winter'
        token1=13
        
    X[i,4]=season.index(season_temp)
    

    if(day>=3 and day <=10):
        token2=5
    elif(day>=11 and day <=19):
        token2=10
    else:
        token2=13
        
        
    if(hr>=3 and hr <=10):
        token3=5
    elif(hr>=11 and hr <=19):
        token3=10
    else:
        token3=13
        
    
        
    index=np.random.randint(9,size=1)[0]       #long
    X[i,2]=location[index][0]
           
    X[i,3]=location[index][1]                   #lat
    
    
    if(index>=0 and index<=3):
        token4=13
    elif(index>=4 and index<=6):
        token4=10
    elif(index>=7 and index<=8):
        token4=5
        

    X[i,9]=temp[np.random.randint(12)]
    
    token5=X[i,9]/75
        
    final_token=token1*token2*token3*token4*token5
    
    mean= minQ + (m*(final_token-minT)) #mapping mean
    
    X[i,0]=np.random.normal(mean,1.5)      #wind speed
   
    index=np.random.randint(8,size=1)[0]       #wind dir
    X[i,1]=wind_dir[index]
            
cols=['wind speed','Wind dir','long','lat','season','day','month','year','time','temp']
pd.DataFrame(X,columns=cols).to_csv("wind_dir_dummy_data_file.csv")
    