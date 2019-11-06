# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 04:02:45 2019

@author: ROHIT
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def conc(x,y,z,hs,z0,Q,U):
    deltah=10
    H=hs+deltah
    iy=0.88/(math.log((hs/z0),math.e)-1)
    iz = 0.50 / (math.log((hs / z0), math.e) - 1)

    sigmay=x*iy
    sigmaz=x*iz

    temp1=(Q/U)*(1/(2*math.pi*sigmay*sigmaz))
    temp2=math.exp(-math.pow(y,2)/(2*(math.pow(sigmay,2))))
    temp3=(math.exp((-math.pow((z-H),2))/(2*math.pow(sigmaz,2))) + math.exp((-math.pow((z+H),2)/(2*math.pow(sigmaz,2)))))
    final=temp1*temp2*temp3

    return final;



#const-def
poll=[50,70] #mean=60 sd=10
date_day=[1,30]
date_month=[1,12]
date_yr=[2017,2018,2019,2020]
stack_speed=[3,6]
wind_speed=[3,6]
#wind_dir=["NE","NW","SE","SW","N","S","E","W"]-mapped as...
wind_dir=[0,1,2,3,4,5,6,7]
stack_height=[230,260]
Q=[230,270]
output=[60-90]

location=[[1,2],[2,4],[3,8],[4,6],[5,3],[6,9],[7,2],[8,4],[9,1]]
Z=[0.0002,0.005,0.03,0.10,0.25,0.50,1.0,2,4]

#rows and cols
features=18
n=5000
X=np.zeros((n,features))

#date stuff
day=1
month=1
yr=2017

season=['rain','summer','winter']


#cal slope for maping
maxT=2535
minT=50
maxQ=300
minQ=150
m=(maxQ-minQ)/(maxT-minT)
# output = output_start + m * (input - input_start)


for i in range(n):
    X[i,13]=np.random.normal(70,10)             #output of the factory
    output_temp=X[i,13]
    
    X[i,14]=day                    #day
    
    X[i,15]=month                  #month
    
    X[i,16]=yr                   #year
    day+=1
    if(day > 31):
        day=1
        month+=1
        if(month > 12):
            month=1
            yr+=1
            
            
    #cal token for Q
    if(month>=3 and month <=5):
        season_temp='summer'
        token1=5
    elif(month>=6 and month <=9):
        season_temp='rain'
        token1=10
    else:
        season_temp='winter'
        token1=13
            
    X[i,17]=token1
    
    index=np.random.randint(9,size=1)[0]       #long
    X[i,6]=location[index][0]
           
    X[i,7]=location[index][1]                   #lat
    
    X[i,8]=Z[index]                             #Z0
    
    #cal token for Q
    if(index>=0 and index<=3):
        token2=13
    elif(index>=4 and index<=6):
        token2=10
    elif(index>=7 and index<=8):
        token2=5
    
            
    if(output_temp>40 and output_temp<60):
        token3=5
    elif(output_temp>60 and output_temp<70):
        token3=10
    elif(output_temp>70 and output_temp<80):
        token3=13
    elif(output_temp>80):
        token3=15
    else:
        token3=2
        
    final_token=token1*token2*token3
    
    Q_mean= minQ + (m*(final_token-minT)) #mapping Qmean
      
    X[i,5]=np.random.normal(Q_mean,70) #final Q
        
    X[i,0]=np.random.normal(60,10)        #poll conc
    
    X[i,1]=np.random.normal(3,1.5)    #stack speed
    
    X[i,2]=np.random.normal(4.5,1.5)      #wind speed
   
    x=np.random.randint(1000,size=1)[0]+1       
    X[i,9]=x                                    #x
    
    y=np.random.randint(1000,size=1)[0]+1
    X[i,10]=y                                    #y
    
    z=np.random.randint(1000,size=1)[0]+1
    X[i,11]=z                                   #z
    
    X[i,12]= conc(x,y,z,120,Z[index],X[i,0],X[i,2])     #conc at a point
    
    index=np.random.randint(8,size=1)[0]       #wind dir
    X[i,3]=wind_dir[index]
    
    X[i,4]=np.random.normal(250,25)       #stack height
    
    

    
            
cols=['avg poll conc.','stack speed','wind speed','Wind dir','Stack height','Q','long','lat','Z0','X','Y','Z','conc. at a point','output','day','month','year','season']
pd.DataFrame(X,columns=cols).to_csv("dummy_data_file1_150_300_withseason.csv")
    