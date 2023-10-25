#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Import functions
import fileLoader
import plotFunctions as pf





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


#print(station00.iloc[1,5:15])


#sns.set(font_scale=1.3)

#pf.plotPowCorr(station_data)



temp1 = fileLoader.sliceData(station01,"2018-08-15 23:00:00","2018-08-16 10:00:00")
temp2 = fileLoader.sliceData(station01,"2018-08-16 23:00:00","2018-08-17 10:00:00")
temp3= fileLoader.sliceData(station01,"2018-08-17 23:00:00","2018-8-18 10:00:00")
temp4= fileLoader.sliceData(station01,"2018-08-18 23:00:00","2018-8-19 10:00:00")
temp5= fileLoader.sliceData(station01,"2018-08-19 23:00:00","2018-8-20 10:00:00")
temp6 = fileLoader.sliceData(station01,"2018-08-20 23:00:00","2018-8-21 10:00:00")
temp7= fileLoader.sliceData(station01,"2018-08-21 23:00:00","2018-8-22 10:00:00")
temp8= fileLoader.sliceData(station01,"2018-08-22 23:00:00","2018-8-23 10:00:00")
temp9= fileLoader.sliceData(station01,"2018-08-23 23:00:00","2018-8-24 10:00:00")
temp10= fileLoader.sliceData(station01,"2018-08-24 23:00:00","2018-8-25 10:00:00")
print(temp1.head)
print(temp2.head)
station01 = pd.concat((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10), axis=0)
print(station01)

pf.circle3dScatterPlot(station01,"average","Station01")


#station00 = station00.loc["nwp_globalirrad"]
#print(station00[0])

#X = station00.drop(columns=["power"])
 
#VIF = variance_inflation_factor(X, station00["power"])

#print(VIF)




#fig = plt.figure()
#ax2 = fig.add_subplot(1, 1, 1)
#pf.correlationMatrixPlotter(ax2,station02)
#plt.tight_layout()

plt.show()

