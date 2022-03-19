# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:52:38 2020
This Script clusters the data by an Hierachical CLustering and Kmeans
"""
#%%Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
#%%Deklaration zum Plot einer Spalte
def plotParameter(data,col):
    if isinstance(col,list) == False:
        col=[col]        
    fig=plt.figure()
    ax=fig.subplots(len(col),sharex=True)
    for i, frame in data.groupby('Unit'):
        for j,axes in enumerate(fig.axes):
            axes.plot(frame[col[j]],label='Unit '+str(i))
    for j,axes in enumerate(fig.axes):
        axes.legend()
        axes.set_title(col[j])
#%%Laden des Datensatzes
#Wahl des Parks
park=1
#Wahl der Anzahl an Cluster -> Parameter aus Ellbow-Kurve
k=5 
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
#%%Feature Engineering
data['Season']=((data.index.month % 12 + 3) // 3).map({1:4, 2:1, 3:2, 4:3}) #1:Frühling 2:Sommer usw.
data['Amb_TurbIntensity']=data['Amb_WindSpeed_Std']/data['Amb_WindSpeed_Avg'] #Neues Feature Turbulenz
data['Amb_Density']=(1*10^5)/(287*(data['Amb_Temp_Avg']+273.15)) #Annahme Luft 1bar und R 287
#%%Zwischenschritt Visualisierung der Power
power=data.groupby('Unit')['Grd_Prod_Pwr_Avg'].mean().sort_values()
power.index = power.index.map(str)
fig=plt.figure()
ax=fig.subplots()
ax.plot(power,'X')
#%%Ambient Clustering based on Features
#Select Important Parameters
pars=['Amb_WindSpeed_Avg','Amb_WindDir_Abs_Avg','Amb_TurbIntensity','Amb_Density'] #Select those Parameters 
features=[]
for i,frame in data.groupby('Unit'):
    stats=[]
    for column in pars:
        temp=[frame[column].mean(),frame[column].std(),frame[column].skew(),frame[column].kurt()]
        temp=pd.DataFrame([temp],columns=[column+'_mean',column+'_std',column+'_skew',column+'_kurt'])
        stats.append(temp)
    temp=pd.concat(stats,axis=1)
    temp['Unit']=i
    features.append(temp)
features=pd.concat(features).reset_index(drop=True)

####Hierachical Clustering
units=features['Unit']
features=features.iloc[:,:-1]

scaler=MinMaxScaler()
features=scaler.fit_transform(features)

Z = linkage(features, 'ward')
dn = dendrogram(Z,labels=units.values)

#####KMeans Clustering
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,len(features)))
visualizer.fit(features)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
#%%Visualize Kmeans
kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
power=data.groupby('Unit')['Grd_Prod_Pwr_Avg'].mean().sort_values()
power.index = power.index.map(str)
power=power.to_frame()
power['Label']=kmeans.labels_

fig=plt.figure()
ax=fig.subplots()
ax.plot(power[power.columns[0]],'X')
for i,frame in power.groupby('Label'):
    ax.plot(frame['Grd_Prod_Pwr_Avg'],'X',label='Cluster '+str(i))

