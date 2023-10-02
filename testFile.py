#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf
import numpy as np

#Start by listing data files in PVO dataset
station00 = fileLoader.loadFile("station00.csv")
station01 = fileLoader.loadFile("station01.csv")
station02 = fileLoader.loadFile("station02.csv")
station03 = fileLoader.loadFile("station03.csv")
station04 = fileLoader.loadFile("station04.csv")
station05 = fileLoader.loadFile("station05.csv")
station06 = fileLoader.loadFile("station06.csv")
station07 = fileLoader.loadFile("station07.csv")
station08 = fileLoader.loadFile("station08.csv")
station09 = fileLoader.loadFile("station09.csv")
#fileLoader.fileInfo(station02)


#t=fileLoader.sliceData(station02,"2018-07-22 16:15:00","2018-07-22 18:15:00")
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

#Check data

# print(station02.head())

#data2 = station02.select_dtypes(include=['float64'])

print(station02[1:2])

s2Power = station02["power"]
s2nwp_globalirrad = station02["nwp_globalirrad"]
s2nwp_directirrad = station02["nwp_directirrad"]
s2nwp_temperature = station02["nwp_temperature"]
s2nwp_humidity = station02["nwp_humidity"]
s2nwp_windspeed = station02["nwp_windspeed"]
s2nwp_winddirection = station02["nwp_winddirection"]
s2nwp_pressure = station02["nwp_pressure"]


def powCorr(data):
    correlation = np.zeros(7)
    dataSelec = data.select_dtypes(include=['float64'])
    correlation[0] = dataSelec["power"].corr(dataSelec["nwp_globalirrad"])
    correlation[1] = dataSelec["power"].corr(dataSelec["nwp_directirrad"])
    correlation[2] = dataSelec["power"].corr(dataSelec["nwp_temperature"])
    correlation[3] = dataSelec["power"].corr(dataSelec["nwp_humidity"])
    correlation[4] = dataSelec["power"].corr(dataSelec["nwp_windspeed"])
    correlation[5] = dataSelec["power"].corr(dataSelec["nwp_winddirection"])
    correlation[6] = dataSelec["power"].corr(dataSelec["nwp_pressure"])
    return correlation


vectors = []
vectors.append(powCorr(station00))
vectors.append(powCorr(station01))
vectors.append(powCorr(station02))
vectors.append(powCorr(station03))
vectors.append(powCorr(station04))
vectors.append(powCorr(station05))
vectors.append(powCorr(station06))
vectors.append(powCorr(station07))
vectors.append(powCorr(station08))
vectors.append(powCorr(station09))
powCorrMatrix = np.array(vectors)
import pandas as pd
x_axis_labels = ["NWP Globalirrad","NWP Directirrad","NWP Temperature","NWP Humidity","NWP Windspeed","NWP Winddirection","NWP Pressure"] # labels for x-axis
y_axis_labels = ["Station00","Station01","Station02","Station03","Station04","Station05","Station06","Station07","Station08","Station09"] # labels for y-axis
powCorrMatrix = pd.DataFrame(powCorrMatrix)





fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


import seaborn as sns
ax = sns.heatmap(powCorrMatrix, ax=ax,vmin = -1, vmax = 1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
#ax.xaxis.tick_top()
ax.set_title("Correlation matrix of power from each recorded station and the corresponding NWP data")
plt.tight_layout()




#print(station02["power"])


#print(station02.corr())


#pf.nyFunktion(ax2,station02)
#ax2.set_title("Correlation matrix of dataset")

#plt.tight_layout()




#print(station02.loc["2018-07-22 17:00:00"]["power"])


plt.show()

