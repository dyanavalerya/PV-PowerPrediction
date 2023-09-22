#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import file handler functions
import fileLoader
# Import plotting functions
import plotFunctions as pf

#Start by listing data files in PVO dataset


# EXAMPLE 
try:
    station02 = fileLoader.loadFile("station02.csv", r'C:\Users\andre\OneDrive - Aalborg Universitet\_Universitet\ES7\_ES7 project\literatureAndDataset\dataset')
except:
    



print(station02.head())
    
# Create a figure to plot on, 2 axes in 1 column
[fig,ax]=plt.subplots(2,1)
# access the first axis by ax[0] and the second by ax[1]

#pf.plotTimeSeries(ax[0],data,"power","power")
pf.plotColumnScatter(ax[1],data,"power","lmd_windspeed","power vs windspeed")
pf.plotHistogram(ax[0],data,"power","power")
plt.show()  

