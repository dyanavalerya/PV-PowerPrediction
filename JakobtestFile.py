#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import os, sys
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import numpy as np
import seaborn as sns
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Import functions
from PV_PredictLib import fileLoader 
from PV_PredictLib import plotFunctions as pf




"""
def loadFile(file_name, path=None):
    if path == None:
        #print(f"Path of current program:\n", os.path.abspath(os.path.dirname(__file__)))
        datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/'))
    else:
        datafolder_path = path
    
 
    
    file_path = os.path.join(datafolder_path, file_name)
    file_data=None
    # assign data in file
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path,header=0)
        file_data.index = pd.DatetimeIndex(file_data["date_time"])
    else:
        #print("File name does not exist. Remember to include file type in file name")
        sys.exit()

    #print("\n*** File succesfully loaded ***")
    #print("\nFile preview:")
    #print(file_data.head())
    return file_data



station00=loadFile("station00.csv")
station01=loadFile("station01.csv")
station02=loadFile("station02.csv")
station03=loadFile("station03.csv")
station04=loadFile("station04.csv")
station05=loadFile("station05.csv")
station06=loadFile("station06.csv")
station07=loadFile("station07.csv")
station08=loadFile("station08.csv")
station09=loadFile("station09.csv")

station_data = [station00, station01, station02, station03, station04, station05, station06, station07, station08, station09]
"""
meta = fl.loadFile("metadata.csv")
for i in range(8):
    tempStr =  "dataset/station0" + str(i) + ".pkl"
    file = open(tempStr, 'rb')
    station = pickle.load(file)
    station["power"] = station["power"] * meta["Capacity"][i]
    station.to_pickle(f"station0{i}.pkl")

"""
station = fl.sliceData(station,"2018-07-21 00:00:00","2018-07-30 23:59:59")
file.close()
fig,ax=plt.subplots(1,1,figsize=(8,6))
pf.plotTimeSeries(ax,station,"power","power")
"""
plt.show()

