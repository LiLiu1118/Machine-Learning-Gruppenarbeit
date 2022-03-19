# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 06:25:09 2020
This Script will read the raw-Data, merge and cleans it and finally saves it as a "Python-readable"-File
"""
#Here are the Imports
import glob
import os
import pandas as pd
import numpy as np
import pickle

#Edit the WorkingDirectory if necessarry
#The Structur of the files-folder should look like:
#....fileFolder
#....fileFolder/Windpark_1 - WTG_01
#.............................................
#....fileFolder/Windpark_1 - WTG_01/Trust.txt
#....fileFolder/Windpark_1 - WTG_01/WTG01.txt
#............................................
#....fileFolder/Windpark_1 - WTG_02
#....fileFolder/Windpark_1 - WTG_03
files=glob.glob(r'E:\02_Python\04_MlApplications\Windpark*')

#Define a Function that reads and merges all files
def importFilesfromFolder(folder):
    #IMPORT FILES
    trustFiles=glob.glob(folder+r'\\Trust*.txt')    #List with the name of all Trust-Files in folder
    WtgFiles=glob.glob(folder+r'\\WTG*.txt')        #List with the name of all Wtg-Files in folder
    
    #Create empty list to store the raw DATA txt-Files
    data=[]
    for file in WtgFiles:
        print('Read '+file)
        data.append(pd.read_csv(file,sep=';',decimal=',',index_col=0,parse_dates=True,dayfirst=True).sort_index())
    print('Imported '+str(len(data))+' Data-Files')
    
    #Create empty list to store the raw TRUST txt-Files
    trust=[]
    for file in trustFiles:
        print('Read '+file)
        trust.append(pd.read_csv(file,sep=';',decimal=',',index_col=0,parse_dates=True,dayfirst=True).sort_index())
    print('Imported '+str(len(trust))+' Trust-Files') 
    return (data,trust)

def appendFrameList(frameList):
    #This Function converts a list with several DataFrames in a single Dataframe
    for i in range(len(frameList)):
        if i == 0:
            frame=frameList[i]
        else:
            frame=frame.append(frameList[i],sort=False)
            
    print('Status Duplicates in Index:',frame.index.duplicated().any())
    print('Status Duplicates in Columns:',frame.columns.duplicated().any())
    return frame

def cleanDuplicatedIndex(frame):
    #Check for Duplicated Indexes
    if frame.index.duplicated().any():
        #Get the smalles Value of the Entries
        #For a double appearing index with value [1] and [NaN] the min is 1
        anzahl=frame[frame.index.duplicated(keep=False)].shape[0]
        print(str(anzahl)+' doppelte Index gefunden')
        frame=frame.groupby(level=0).min()
        print('Status Duplicates in Index:',frame.index.duplicated().any())
        print('Status Duplicates in Columns:',frame.columns.duplicated().any())
        return frame         
    else:
        return frame
    
def checkTrustValues(frame):
    return None
    #Check if Value unequal from 0 or 1
    #If so -> replace with 0
    #if 

#Define a Function that checks some Data-Properties
def checkData(data,trust):
    dShape=0
    tShape=0
    counter=0
    
    for file in zip(data,trust):
        dShape+=np.array(file[0].shape)
        tShape+=np.array(file[1].shape)
        print('Anzahl Duplicates in Datei-File '+str(counter),np.count_nonzero(data[counter].index.duplicated()))
        print('Anzahl Duplicates in Trust-File '+str(counter),np.count_nonzero(trust[counter].index.duplicated()))
        tempDupData=file[0][file[0].index.duplicated(keep=False)]
        tempDupTrust=file[1][file[1].index.duplicated(keep=False)]
        nanCountData=tempDupData.isnull().sum(axis=1)
        nanCountTrust=tempDupTrust.isnull().sum(axis=1)
        print(nanCountData.unique())
        print(tempDupData.shape)
        print(nanCountTrust.unique())
        counter+=1
    print('Shape Data:',dShape)
    print('Shape Trust:',tShape)
    
    if dShape[0]!=tShape[0]:
        print('ACHTUNG ANZAHL ZEILEN UNTERSCHEIDET SICH')
    if dShape[1]!=tShape[1]:
        print('ACHTUNG ANZAHL SPALTEN UNTERSCHEIDET SICH')
    return (dShape,tShape)

def cleanAllNaN(frame):
    counterRow=0
    counterColumn=0
    nanRowIndex=[]
    nanColumnIndex=[]
    for i,row in frame.iterrows():
        if row.isnull().all():
            counterRow+=1
            nanRowIndex.append(i)
            
    for i,column in frame.iteritems():
        if column.isnull().all():
            counterColumn+=1
            nanColumnIndex.append(i)
            
    print('Anzahl Rows mit allNaN:',counterRow)
    print('Anzahl Columns mit allNaN:',counterColumn)
    
    #Lösche Zeilen
    frame=frame.drop(nanRowIndex,axis=0)
    #Lösche Spalten
    frame=frame.drop(nanColumnIndex,axis=1)
    
    return frame

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
    print('Kein Eintrag für Park und Unit vorhanden')
    print('Suche nach Eintrag bei Unit 1')
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

    print('Kein Eintrag hierfür gefunden:',signal)
    return (-np.inf,np.inf)
#%%    
#Starting the Work
for i,file in enumerate(files):
    #Import Raw-Data
    data,trust=importFilesfromFolder(file)
    
    #Merge Raw-Data to Dataframe
    data=appendFrameList(data)
    trust=appendFrameList(trust)
    
    #Clean Dataframe values
    data=cleanDuplicatedIndex(data)
    trust=cleanDuplicatedIndex(trust)
    
    #Regard Trustfiles and select only valid Values
    frame=data[trust==1]
    
    #Clean Rows/Columns with all NaN
    frame=cleanAllNaN(frame)
    
    with open(file+'.pickle','wb') as f:
        pickle.dump(frame,f)

