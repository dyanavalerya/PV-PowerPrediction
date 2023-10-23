#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf
import numpy as np

station00=fileLoader.loadFile("station00.csv")
station01=fileLoader.loadFile("station01.csv")
station02=fileLoader.loadFile("station02.csv")
station03=fileLoader.loadFile("station03.csv")
station04=fileLoader.loadFile("station04.csv")
station05=fileLoader.loadFile("station05.csv")
station06=fileLoader.loadFile("station06.csv")
station07=fileLoader.loadFile("station07.csv")
station08=fileLoader.loadFile("station08.csv")
station09=fileLoader.loadFile("station09.csv")
station_data = [station00, station01, station02, station03, station04,station05, station06, station07, station08, station09]

pf.nwpError()

# print(station02.head())
#station02_sliced= fileLoader.sliceData(station02,"2018-07-22 16:00:00","2018-07-22 19:00:00")
# Create a figure to plot on, 2 axes in 1 column
#[fig,ax]=plt.subplots(2,2,figsize=(10,10))
# access the first axis by ax[0] and the second by ax[1]
#pf.plotTimeSeries(ax[0][0],station02_sliced,"power","power")
#pf.plotColumnScatter2Y(ax[0][1],station02,"power","lmd_windspeed","nwp_temperature","power vs windspeed")
#pf.plotHistogram(ax[1][0],station02,"power","power")
#plt.tight_layout()

#fig = plt.figure() 
#ax2 = fig.add_subplot(1, 1, 1)
#pf.correlationMatrixPlotter(ax2,station02)
#ax2.set_title("Correlation matrix of dataset")

#plt.tight_layout()