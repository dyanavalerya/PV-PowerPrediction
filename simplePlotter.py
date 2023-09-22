#Simple example of how to read and plot data from the PVO dataset
import sys, os
import pandas as pd
import matplotlib.pyplot as plt


#Start by listing data files in PVO dataset
#Use path relative to this file
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/'))

# list files in
files = os.listdir(path)
for file_name in files:
    print(file_name)

#Read one file
file = "station01.csv"
file_path = os.path.join(path, file)
# check that the file exists
if not (os.path.isfile(file_path)):
    print("File does not exist")
    sys.exit()
data = pd.read_csv(file_path)

# station1=loadData(... , station1)
# station2=loadData(... , station2)

# plt.plot(station1['date_time'], station1['power'])
# plt.plot(station2['date_time'], station2['power'])


print(data.head())
    
# Do a simple plot of date_time vs power
# Slice the data to a specific time period from the date_time column

import plotFunctions as pf
[fig,ax]=plt.subplots(2,1)
#pf.plotTimeSeries(ax[0],data,"power","power")
pf.plotColumnScatter(ax[1],data,"power","lmd_windspeed","power vs windspeed")
pf.plotHistogram(ax[0],data,"power","power")
plt.show()  

