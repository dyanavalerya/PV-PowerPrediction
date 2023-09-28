#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf

#Start by listing data files in PVO dataset
station02 = fileLoader.loadFile("station02.csv")
fileLoader.fileInfo(station02)

# EXAMPLE Andreas PC
# try:
#     station02 = fileLoader.loadFile("station02.csv", r'C:\Users\andre\OneDrive - Aalborg Universitet\_Universitet\ES7\_ES7 project\literatureAndDataset\dataset')
# except:
#     pass
# # EXAMPLE Jeppe PC
# try:
#     station02 = fileLoader.loadFile("station02.csv", r'C:\Users\jeppe\gits\PV-PowerPrediction\dataset')
# except:
#     pass



# print(station02.head())
station02_sliced= fileLoader.sliceData(station02,"2018-07-22 16:00:00","2018-07-22 19:00:00")
# Create a figure to plot on, 2 axes in 1 column
[fig,ax]=plt.subplots(3,1)
# access the first axis by ax[0] and the second by ax[1]
pf.plotTimeSeries(ax[0],station02_sliced,"power","power")
pf.plotColumnScatter(ax[1],station02,"power","lmd_windspeed","power vs windspeed")
pf.plotHistogram(ax[2],station02,"power","power")
plt.show()
