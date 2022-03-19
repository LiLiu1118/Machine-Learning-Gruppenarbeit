# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:59:56 2020
This Script cleans Outlierts with regard to the Json-Data-DescriptionFile
@author: MLUTTME
"""
import json
import pandas as pd
import numpy as np
import pickle
#Select the WIndpark
parkNum=2
descFile=r'E:\02_Python\04_MlApplications\ENBW Data Signals Description.json'
if parkNum==1:
    file=r'E:\02_Python\04_MlApplications\dataPark1.pickle'
    cleanedFile=r'E:\02_Python\04_MlApplications\dataPark1Cleaned.pickle'
elif parkNum==2:
    file=r'E:\02_Python\04_MlApplications\dataPark2.pickle'
    cleanedFile=r'E:\02_Python\04_MlApplications\dataPark2Cleaned.pickle'
    
#Read the Json-File with min-max-Values
with open(descFile,'r') as f:
    desc=json.load(f)
#Load the Dataset
data=pd.read_pickle(file)

#The following Function reads out the MinMaxValues from the Json-File
def getMinMaxForSignal(dicList,signal,park=1,unit=1):
    if signal == 'Amb_WindDir_Abs_Avg':
        return (-5,365)
    if 'HCnt' in signal:
        return (0,605)
    if 'Amb_WindSpeed' in signal:
        return (0,40)
    
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

#The following Function reduces the Data with regard to MinMaxValues
def cleanDataByMinMax(dicList,column,data,park=1,unit=1):
    minVal,maxVal=getMinMaxForSignal(dicList,column,park=park,unit=unit)
    return data[(data[column]<=maxVal)&(data[column]>=minVal)]

#%%Reduce the Data for the following Parameters
pars=['Amb_WindDir_Abs_Avg',
'Amb_WindDir_Relative_Avg',
'Amb_WindSpeed_Avg',
'Blds_PitchAngle_Avg',
'HCnt_Avg_AlarmAct',
'HCnt_Avg_AmbOk',
'HCnt_Avg_GrdOk',
'HCnt_Avg_GrdOn',
'HCnt_Avg_Run',
'HCnt_Avg_SrvOn',
'HCnt_Avg_Tot',
'HCnt_Avg_TrbOk',
'HCnt_Avg_WindOk',
'HCnt_Avg_Yaw',
'Grd_Prod_Pwr_Std',
'Grd_Prod_Pwr_Avg']

for col in pars:
    data=cleanDataByMinMax(desc,col,data,park=parkNum)

#Save the cleaned DataSet
with open(cleanedFile,'wb') as f:
    pickle.dump(data,f)