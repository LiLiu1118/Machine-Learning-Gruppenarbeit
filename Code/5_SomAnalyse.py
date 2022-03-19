# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:43:21 2020
This Script analyses the Data by an Self-Organizing-Map
"""
#%%Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from minisom import MiniSom
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
#%%Laden des Datensatzes
park=2
if park==1:
    file=r'E:\02_Python\04_MlApplications\dataPark1Cleaned.pickle'
    tStart='2014-07-01' #Zeitpunkt ab dem 
elif park==2:
    file=r'E:\02_Python\04_MlApplications\dataPark2Cleaned.pickle'

data=pd.read_pickle(file)
if park==1:
    data=data[tStart:]
elif park==2:  
    data=data[(data['Amb_Temp_Avg']<=np.inf)&(data['Amb_Temp_Avg']>=-20)]
    data=data[(data['Amb_WindSpeed_Min']<40)&(data['Amb_WindSpeed_Min']>=0)]
    data=data[(data['Amb_WindSpeed_Max']<40)&(data['Amb_WindSpeed_Max']>=0)]
#%%Feature Engineering
data['Amb_TurbIntensity']=data['Amb_WindSpeed_Std']/data['Amb_WindSpeed_Avg'] #Neues Feature Turbulenz
data['Amb_Density']=(1*10^5)/(287*(data['Amb_Temp_Avg']+273.15)) #Annahme Luft 1bar und R 287
data['Season']=((data.index.month % 12 + 3) // 3).map({1:4, 2:1, 3:2, 4:3})
#%%Select Important Parameters effecting the Performance
pars=[
'Season',
'Amb_Density',
'Amb_WindDir_Abs_Avg',
'Amb_WindSpeed_Avg',
'Amb_TurbIntensity',
'Blds_PitchAngle_Avg',
'HCnt_Avg_AlarmAct',
'HCnt_Avg_AmbOk',
'HCnt_Avg_SrvOn',
'HCnt_Avg_TrbOk',
'HCnt_Avg_WindOk',
'HCnt_Avg_Yaw',
'Rtr_RPM_Avg',
'Grd_Prod_Pwr_Std',
'Grd_Prod_Pwr_Avg']
#%%Do some Data Preparation
#Resample
data=data.resample('H').mean()
#Drop Na
data=data.dropna()
#Shuffle Data
data = shuffle(data)
X=data[pars]
#Scale Data
scaler = MinMaxScaler()
X=scaler.fit_transform(X)

n=int(np.sqrt(5*np.sqrt(len(X))))

#Rule of Thumb for choosing number of neurons. 5*srt(N)
som = MiniSom(n,n, len(pars), neighborhood_function = 'gaussian', sigma=1,learning_rate= 0.5, random_seed = 10)
som.random_weights_init(data = X)
som.train_batch(data = X, num_iteration = 10000, verbose = True)
#%%Visualisation
#ClusterPlot
fig=plt.figure()
ax=fig.subplots()
ax.pcolor(som.distance_map(), cmap='inferno')
#ParameterPlot
neuron_weights = som.get_weights()
fig=plt.figure(figsize=(12,12))
for i in range(len(pars)):
    ax=fig.add_subplot(int(np.sqrt(len(pars)))+2,int(np.sqrt(len(pars))),i+1)
    ax.pcolor(neuron_weights[:,:,i],cmap='inferno')
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(pars[i])
#fig.tight_layout()