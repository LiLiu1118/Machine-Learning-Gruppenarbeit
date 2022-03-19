# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:12:35 2020
This Script creates all the Pictures for the report
Attention: This File is not an Executable
Execute only Blocks starting with #%%
"""
#%%Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import gridspec
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from windrose import WindroseAxes
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import rc
from sklearn.utils import shuffle
from minisom import MiniSom
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
#%%Lade des Datensatzes
file=r'E:\02_Python\04_MlApplications\dataPark1Cleaned.pickle'
#file=r'E:\02_Python\04_MlApplications\preprocessed_data\preprocessed_data\Windpark_1.pickle'
data=pd.read_pickle(file)
#data=data.fillna(0)
#data=data[data['Prod_LatestAvg_TotActPwr']<500000] #For Park1
#data=data[(data['Grd_Prod_Pwr_Avg']<=2100)&(data['Grd_Prod_Pwr_Avg']>=-50)]
#data=data.interpolate(method='time')
data=data.sort_index()
data['Amb_TurbIntensity']=data['Amb_WindSpeed_Std']/data['Amb_WindSpeed_Avg'] #Neues Feature Turbulenz
data['Amb_Density']=(1*10^5)/(287*(data['Amb_Temp_Avg']+273.15)) #Annahme Luft 1bar und R 287
#data=data[data['Sys_Stats_TrbStat']==1] #Only Data for active Turbinestate
#%%Matplotlib Setup
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('lines', linewidth=1,markersize=2)
rc('figure',figsize=(10,6),dpi=100)
rc('font',size=12)
folder=r'E:\02_Python\04_MlApplications\Abbildungen'
#%%Schaubild für Ausreißer
name=r'\\SchaubildAusreisser.png'
temp=data[data['Unit']==4]

ticks=matplotlib.dates.MonthLocator(interval=6, tz=None)
formatter=matplotlib.dates.AutoDateFormatter(ticks)

fig=plt.figure()
ax=fig.subplots()
ax.xaxis.set_major_locator(ticks)
ax.xaxis.set_major_formatter(formatter)
ax.plot(temp['Prod_LatestAvg_TotActPwr']/1000) #Durch 1000 für Maß in kWh
ax.grid()
ax.set_xlabel(r'\textbf{Zeit}')
ax.set_ylabel(r'\textbf{Elektrische Energie in kWh}')
ax.set_title(r'\textbf{Produzierte elektrische Energie Unit-4 Park-1}')
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#ax.xaxis.set_major_formatter(formatter)
#ax.xaxis.set_major_locator(ticks)
#%%Veranschaulichung der verschiedenen TurbineStates
name=r'\\SchaubildTurbineStates.png'
temp=data[data['Unit']==1]
temp=temp.fillna(0)
scaler = MinMaxScaler()
#temp['Prod_LatestAvg_TotActPwr']=scaler.fit_transform(temp['Prod_LatestAvg_TotActPwr'].values.reshape(-1,1))

fig=plt.figure(figsize=(10,10))
#ax=fig.subplots()

gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
axMain=plt.subplot(gs[0,1])
axDensX=plt.subplot(gs[1,1])
for i,frame in temp.groupby('Sys_Stats_TrbStat'):
    axMain.plot(frame['Amb_WindSpeed_Avg'],frame['Prod_LatestAvg_TotActPwr'],'o',label='State '+str(int(i)),alpha=0.5) 
    print(frame['Prod_LatestAvg_TotActPwr'].mean())
axMain.grid()
axMain.legend()
axMain.set_xlabel(r'\textbf{Windgeschwindigkeit in $\mathbf{\frac{m}{s}}$')
axMain.set_ylabel(r'\textbf{Energie in Wh}')
'#%%3D Schaubild Windspeed/Winddir/Performance
name=r'\\SchaubildSpeedDirPower3D.png'
temp=data[data['Unit']==1]
temp=temp.fillna(0)
#temp=temp[temp['Sys_Stats_TrbStat']==1]
scaler = MinMaxScaler()
temp['Prod_LatestAvg_TotActPwr']=scaler.fit_transform(temp['Prod_LatestAvg_TotActPwr'].values.reshape(-1,1))
temp['WindScaled']=scaler.fit_transform(temp['Amb_WindSpeed_Avg'].values.reshape(-1,1))

fig=plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

plot=ax.scatter(temp['Amb_WindSpeed_Avg'],temp['Amb_WindDir_Abs_Avg'],temp['Prod_LatestAvg_TotActPwr'],c=temp['Amb_WindSpeed_Avg'],cmap = plt.get_cmap('gist_ncar'),s=1)
cbar=plt.colorbar(plot,fraction=0.046, pad=0.04)

ax.set_title(r'\textbf{Charakteristik Unit-4 Park-1}')
ax.set_xlabel(r'\textbf{Windgeschwindigkeit in $\mathbf{\frac{m}{s}}$')
ax.set_ylabel(r'\textbf{Absolute Windrichtung in $^\circ$}')
ax.set_zlabel(r'\textbf{Normierte Leistung}')
cbar.set_label(r'\textbf{Windgeschwindigkeit in $\mathbf{\frac{m}{s}}$')
ax.view_init(30,235)
fig.tight_layout()

#plt.savefig(folder+name,dpi=1000)
#%%Curve Fitting of Poly n=6
rc('lines', linewidth=3,markersize=2)
name=r'\\Polyfit.png'
temp=data[data['Unit']==4]
temp=temp.fillna(0)
temp=temp[temp['Sys_Stats_TrbStat']==1]
#scaler = MinMaxScaler()
#temp['Prod_LatestAvg_TotActPwr']=scaler.fit_transform(temp['Prod_LatestAvg_TotActPwr'].values.reshape(-1,1))

x=temp['Amb_WindSpeed_Avg']
y=temp['Prod_LatestAvg_TotActPwr']

rss=[]
for i in range(15):
    z = np.polyfit(x, y, i,full=True)
    p = np.poly1d(z[0])
    rss.append(z[1])
rss=np.concatenate(rss)

fig=plt.figure(figsize=(5,3.5))
ax=fig.subplots()
ax.plot(np.log10(rss))
ax.axvline(6, ls='--', color='r')

ax.set_xlabel(r'\textbf{Ordnung des Polynoms}')
ax.set_ylabel(r'$\mathbf{\log _{10}RSS}$')
ax.set_title(r'\textbf{Polynomfitting}')
ax.grid()
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Example of fitted Curve
name=r'\\PolyVis.png'
#rc('lines', linewidth=3,markersize=2)

temp=data[data['Unit']==4]
temp=temp.fillna(0)
temp=temp[temp['Sys_Stats_TrbStat']==1]

x=temp['Amb_WindSpeed_Avg']
y=temp['Grd_Prod_Pwr_Avg']

z = np.polyfit(x, y, 6)
p = np.poly1d(z)

xx=np.linspace(3,14,1000)

fig=plt.figure(figsize=(5,3.5))
ax=fig.subplots()
ax.scatter(x,y/1000,label='Datenpunkte')
ax.plot(xx,p(xx)/1000,label='Polynom 6. Ordnung',color='orange')

ax.grid()
ax.legend()
ax.set_title(r'\textbf{Polynomfitting 6.Ordnung}')
ax.set_xlabel(r'\textbf{Windgeschwindigkeit in $\mathbf{\frac{m}{s}}$')
ax.set_ylabel(r'\textbf{Energie in kWh}')
fig.tight_layout()
#plt.savefig(folder+name,dpi=1000)
#%%Wahl eines Auschnitts
name=r'\\SegmentPark1.png'

temp=data
temp=temp.fillna(0)
#temp=temp[temp['Sys_Stats_TrbStat']==1]
ticks=matplotlib.dates.YearLocator()
formatter=matplotlib.dates.AutoDateFormatter(ticks)

fig=plt.figure()
ax=fig.subplots()

ax.xaxis.set_major_locator(ticks)
ax.xaxis.set_major_formatter(formatter)

for i,frame in temp.groupby('Unit'):
    ax.plot(frame['Amb_WindSpeed_Avg'],label='Unit '+str(i),alpha=0.5)
ax.axvline(pd.to_datetime('2014-07-01'),color='r',label='Segment',ls='--')
ax.legend()
ax.set_title(r'\textbf{Wahl des Zeitabschnitts}')
ax.set_xlabel(r'\textbf{Zeitraum}')
ax.set_ylabel(r'\textbf{Windgeschwindigkeit in $\mathbf{\frac{m}{s}}$')
fig.tight_layout()
plt.savefig(folder+name,dpi=800)
#%%Schaubild Leistung Park1/Park2
name=r'\\LeistungPark1.png'
rc('lines', linewidth=3,markersize=20)


power=data.groupby('Unit')['Grd_Prod_Pwr_Avg'].mean().sort_values()
power.index = power.index.map(str)

fig=plt.figure(figsize=(6,5))
ax=fig.subplots()
ax.plot(power,'X')
ax.grid()
ax.set_xlabel(r'\textbf{Windkraftanlage}')
ax.set_ylabel(r'\textbf{Leistung in kW}')
ax.set_title(r'\textbf{Durchschnittliche Leistung Windkraftanlagen Park-1}')

ax.grid()
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%WindRose Schaubild
name=r'\\WindRosePark1.png'
bins_range = np.arange(1,14,2) # this sets the legend scale

fig = plt.figure(figsize=(7,5))
for i,frame in data.groupby('Unit'):
    ax=fig.add_subplot(2,3,int(i),projection='windrose')
    ax.bar(frame['Amb_WindDir_Abs_Avg'],frame['Amb_WindSpeed_Avg'],normed=True,opening=0.9,edgecolor='white',bins=bins_range,cmap=plt.get_cmap('viridis'))
    ax.set_yticks(np.arange(0, 15, step=3))
    ax.set_yticklabels(np.arange(0, 15, step=3))
    ax.set_xlabel(r'\textbf{Unit '+str(i)+r'}')
  
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Schaubild Clustering on Features Dendogramm
name=r'\\AmbientHirachicalPark1.png'

#tStart='2014-07-01'
#temp=data[tStart:].copy()
temp=data
pars=['Amb_WindSpeed_Avg','Amb_WindDir_Abs_Avg','Amb_TurbIntensity','Amb_Density']
scaler=MinMaxScaler()
temp[temp.columns.difference(['Unit'])]=scaler.fit_transform(temp[temp.columns.difference(['Unit'])].values)
####Feature Generation
features=[]
for i,frame in temp.groupby('Unit'):
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
Z = linkage(features.values, 'ward')

rc('lines', linewidth=3,markersize=2)
fig=plt.figure(figsize=(5,5))
ax=fig.subplots()
dn = dendrogram(Z,ax=ax,labels=units.values)

ax.set_xlabel(r'\textbf{Windkraftanlage}')
ax.set_ylabel(r'\textbf{Euklidischer Abstand}')
ax.set_title(r'\textbf{Hierarchische Clusteranalyse Park-1}')

fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Schaubild Clustering on Features Dendogramm PARK2
name=r'\\AmbientHirachicalPark2.png'
pars=['Amb_WindSpeed_Avg','Amb_WindDir_Abs_Avg','Amb_TurbIntensity','Amb_Density']
scaler=MinMaxScaler()
data[data.columns.difference(['Unit'])]=scaler.fit_transform(data[data.columns.difference(['Unit'])].values)
####Feature Generation
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
Z = linkage(features.values, 'ward')
rc('lines', linewidth=3,markersize=2)
fig=plt.figure(figsize=(5,5))
ax=fig.subplots()
dn = dendrogram(Z,ax=ax,labels=units.values)

ax.set_xlabel(r'\textbf{Windkraftanlage}')
ax.set_ylabel(r'\textbf{Euklidischer Abstand}')
ax.set_title(r'\textbf{Hierarchische Clusteranalyse Park-2}')

fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Schaubild KMeans Clustering
name=r'\\AmbientKmeansPark1.png'

tStart='2014-07-01'
temp=data[tStart:].copy()
pars=['Amb_WindSpeed_Avg','Amb_WindDir_Abs_Avg','Amb_TurbIntensity','Amb_Density']
scaler=MinMaxScaler()
temp[temp.columns.difference(['Unit'])]=scaler.fit_transform(temp[temp.columns.difference(['Unit'])].values)
####Feature Generation
features=[]
for i,frame in temp.groupby('Unit'):
    stats=[]
    for column in pars:
        temp=[frame[column].mean(),frame[column].std(),frame[column].skew(),frame[column].kurt()]
        temp=pd.DataFrame([temp],columns=[column+'_mean',column+'_std',column+'_skew',column+'_kurt'])
        stats.append(temp)
    temp=pd.concat(stats,axis=1)
    temp['Unit']=i
    features.append(temp)
features=pd.concat(features).reset_index(drop=True)
####KMeans Clustering
units=features['Unit']
features=features.iloc[:,:-1]

model = KMeans()
fig=plt.figure(figsize=(5,5))

visualizer = KElbowVisualizer(model, k=(2,len(features.values)),locate_elbow=False)
visualizer.fit(features.values)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
ax=fig.axes
fig.tight_layout()

ax[0].set_xlabel(r'\textbf{Anzahl der Cluster}')
ax[0].set_ylabel(r'\textbf{Summe der Abweichungsquadrate}')
ax[0].set_title(r'\textbf{Kmeans Clusteranalyse Park-1}')
ax[1].set_ylabel(r'\textbf{Rechenzeit in Sekunden}')
ax[0].set_xticks(np.arange(2, 6, step=1))
ax[0].set_xticklabels(np.arange(2,6, step=1))

fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Schaubild KMeans Clustering
name=r'\\AmbientKmeansPark2.png'

pars=['Amb_WindSpeed_Avg','Amb_WindDir_Abs_Avg','Amb_TurbIntensity','Amb_Density']
scaler=MinMaxScaler()
temp[temp.columns.difference(['Unit'])]=scaler.fit_transform(temp[temp.columns.difference(['Unit'])].values)
####Feature Generation
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
####KMeans Clustering
units=features['Unit']
features=features.iloc[:,:-1]

rc('lines', linewidth=2,markersize=5)
model = KMeans()
fig=plt.figure(figsize=(5,5))

visualizer = KElbowVisualizer(model, k=(2,len(features.values)),locate_elbow=False)
visualizer.fit(features.values)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
ax=fig.axes
fig.tight_layout()

ax[0].set_xlabel(r'\textbf{Anzahl der Cluster}')
ax[0].set_ylabel(r'\textbf{Summe der Abweichungsquadrate}')
ax[0].set_title(r'\textbf{Kmeans Clusteranalyse Park-2}')
ax[1].set_ylabel(r'\textbf{Rechenzeit in Sekunden}')
ax[0].set_xticks(np.arange(2, 18, step=2))
ax[0].set_xticklabels(np.arange(2,18, step=2))

fig.tight_layout()
plt.savefig(folder+name,dpi=1000)

#%%Schaubild SOM Clustering
name=r'\\SomPark1.png'
pars=[
'Amb_Density',
'Amb_Temp_Avg',
'Amb_WindDir_Abs_Avg',
'Amb_WindSpeed_Avg',
'Amb_WindSpeed_Max',
'Amb_WindSpeed_Min',
'Amb_WindSpeed_Std',
'Blds_PitchAngle_Avg',
'HCnt_Avg_AlarmAct',
'HCnt_Avg_AmbOk',
'HCnt_Avg_GrdOk',
'HCnt_Avg_Run',
'HCnt_Avg_SrvOn',
'HCnt_Avg_TrbOk',
'HCnt_Avg_WindOk',
'HCnt_Avg_Yaw',
'Nac_Direction_Avg',
'Rtr_RPM_Avg',
'Grd_Prod_Pwr_Std',
'Grd_Prod_Pwr_Avg']
tStart='2014-07-01'

temp=data[tStart:].copy()
temp=temp.resample('H').mean()

temp=temp.dropna()
temp = shuffle(temp)
X=temp[pars]

scaler = MinMaxScaler()
X=scaler.fit_transform(X)

n=int(np.sqrt(5*np.sqrt(len(X))))
som = MiniSom(n,n, len(pars), neighborhood_function = 'gaussian', sigma=1,learning_rate= 0.9, random_seed = 10)
som.random_weights_init(data = X)
som.train_batch(data = X, num_iteration = 10000, verbose = True)
neuron_weights = som.get_weights()
fig=plt.figure(figsize=(12,12))
for i in range(len(pars)):
    ax=fig.add_subplot(int(np.sqrt(len(pars))),int(np.sqrt(len(pars)))+1,i+1)
    ax.pcolor(neuron_weights[:,:,i],cmap='inferno')
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(pars[i])
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)
#%%Schaubild Kmeans Park1/2
rc('lines', linewidth=3,markersize=20)
park=1
name=r'\\KmeansPark'+str(park)+r'.png'
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
data['Amb_TurbIntensity']=data['Amb_WindSpeed_Std']/data['Amb_WindSpeed_Avg'] #Neues Feature Turbulenz
data['Amb_Density']=(1*10^5)/(287*(data['Amb_Temp_Avg']+273.15)) #Annahme Luft 1bar und R 287
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
units=features['Unit']
features=features.iloc[:,:-1]
scaler=MinMaxScaler()
features=scaler.fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
power=data.groupby('Unit')['Grd_Prod_Pwr_Avg'].mean().sort_values()
power.index = power.index.map(str)
power=power.to_frame()
power['Label']=kmeans.labels_

fig=plt.figure(figsize=(5,5))
ax=fig.subplots()
ax.plot(power[power.columns[0]],'X')
for i,frame in power.groupby('Label'):
    ax.plot(frame['Grd_Prod_Pwr_Avg'],'X',label='Cluster '+str(i))
ax.set_xlabel(r'\textbf{Windkraftanlage}')
ax.set_ylabel(r'\textbf{Leistung in kW}')
ax.set_title(r'\textbf{Kmeans Clustering Park-2}')
ax.legend()
fig.tight_layout()
plt.savefig(folder+name,dpi=1000)