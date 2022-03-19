# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:02:28 2020
This Script reads the single WindPark files and combines them
Furthermore it cleans outliers based on the min-max-Values in the json
"""
#Here are the Imports
import glob
import pandas as pd
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import json

#Define the folder where the data is stored
files=glob.glob(r'E:\02_Python\04_MlApplications\Windpark*.pickle')
savePark1=r'E:\02_Python\04_MlApplications\dataPark1.pickle'
savePark2=r'E:\02_Python\04_MlApplications\dataPark2.pickle'
descFile=r'E:\02_Python\04_MlApplications\ENBW Data Signals Description.json'

#Function for Thresholfbased Outlier-Detection
def getMinMaxForSignal(dicList,signal,park=1,unit=1):
    if park==1: #Tausche Park, da in Json-File vertauscht
        park=2
    else:
        park=1
    for entry in dicList:
        parkEntry=int(entry['powerPlantUnit']['park'][-2:])
        unitEntry=int(entry['powerPlantUnit']['stationId'])
        if ((parkEntry==park) and (unitEntry==unit)):
            parName=entry['signal']['name']
            try:
                parName=parName.split('%10')[0]
            except:
                pass
            if parName == signal:
                try:
                    minVal=entry['dataInfo']['normalRange']['min']
                except:
                    minVal=-np.inf
                try:
                    maxVal=entry['dataInfo']['normalRange']['max']
                except:
                    maxVal=np.inf

                return (minVal,maxVal)
    #print('Kein Eintrag für Park und Unit vorhanden')
    #print('Suche nach Eintrag bei Unit 1')
    if ((parkEntry==park) and (unitEntry==1)):
            parName=entry['signal']['name']
            try:
                parName=parName.split('%10')[0]
            except:
                pass
            if parName == signal:
                try:
                    minVal=entry['dataInfo']['normalRange']['min']
                except:
                    minVal=-np.inf
                try:
                    maxVal=entry['dataInfo']['normalRange']['max']
                except:
                    maxVal=np.inf

    #print('Kein Eintrag hierfür gefunden:',signal)
    return (-np.inf,np.inf)
#Define two Storagevars
dataPark1=[]
dataPark2=[]
with open(descFile,'r') as f:
    desc=json.load(f)

#Load whole Dataset
for file in files:
    #Extract the park and unit from filename -> Change the last Bracket if Filename != "Windpark_1 - WTG_01.pickle'
    park=int(file[-17:-16])
    unit=int(file[-9:-7])
    #Do some more preprocessing: Handle Nan and Outliers
    outlierTresh=0.998 #Erlaube List um 2% zu redzuzieren
    temp=pd.read_pickle(file)
    #temp=temp.interpolate(method='time')
    temp=temp.fillna(0)
    
    for column in temp.columns:
        minVal,maxVal=getMinMaxForSignal(desc,column,park=park,unit=unit)
        kopie=temp.copy()
        vorher=len(temp)
        temp=temp[(temp[column]<=maxVal)&(temp[column]>=minVal)]
        nachher=len(temp)
        if ((nachher/vorher) < outlierTresh):
            temp=kopie
    #Create two new columns to store park und unit information
    temp['Windpark']=park
    temp['Unit']=unit
    #Save into storagevars for further processing
    if park==1:
        dataPark1.append(temp)
    else:
        dataPark2.append(temp)
#%%Merge the the single DataFrames and save to new file
dataPark1=pd.concat(dataPark1,join='inner')
dataPark2=pd.concat(dataPark2,join='inner')
#Save data for park1
with open(savePark1,'wb') as f:
    pickle.dump(dataPark1,f)
    
#Save data for park2
with open(savePark2,'wb') as f:
    pickle.dump(dataPark2,f,protocol=4)

