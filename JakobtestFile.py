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


sns.set(font_scale=1.3)

pf.plotPowCorr(station_data)

#station06sliced = fileLoader.sliceData(station06,"2018-07-13 16:00:00","2018-10-13 16:00:00")



#station00 = station00.loc["nwp_globalirrad"]
#print(station00[0])

#X = station00.drop(columns=["power"])
 
#VIF = variance_inflation_factor(X, station00["power"])

#print(VIF)



"""

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1,projection='3d')
#pf.correlationMatrixPlotter(ax1,station06)
#plt.tight_layout()

nwp_winddirection = station00["nwp_winddirection"].to_numpy()
temperature = station00.iloc[:, 3].to_numpy()

# Initialize arrays to store sums and counts for each angle
gns_temp = np.zeros(360, dtype=float)
temp_indeks = np.zeros(360, dtype=int)

for angle in range(360):
    angle_range = (angle <= nwp_winddirection) & (nwp_winddirection <= angle + 1)

    # Calculate sums and counts for the current angle
    gns_temp[angle] = np.sum(temperature[angle_range])
    temp_indeks[angle] = np.sum(angle_range)

# Avoid division by zero and compute the final average
temp_indeks_nonzero = temp_indeks > 0
gns_temp[temp_indeks_nonzero] /= temp_indeks[temp_indeks_nonzero]



x = np.array(list(range(360)) )

W1 = [np.cos(x*np.pi/180), np.sin(x*np.pi/180)]
ax.scatter(W1[0],W1[1],gns_temp)

ax.set_xlabel('Cosinus')
ax.set_ylabel('Sinus')
ax.set_zlabel('average nwp_temperature')
ax.set_title('Station 00 average temperature compared to wind direction')

ax = fig.add_subplot(1, 2, 2, projection='3d')

nwp_winddirection = station00["nwp_winddirection"].to_numpy()
power = station00.iloc[:, 14].to_numpy()

# Initialize arrays to store sums and counts for each angle
gns_power = np.zeros(360, dtype=float)
power_indeks = np.zeros(360, dtype=int)

for angle in range(360):
    angle_range = (angle <= nwp_winddirection) & (nwp_winddirection <= angle + 1)

    # Calculate sums and counts for the current angle
    gns_power[angle] = np.sum(power[angle_range])
    power_indeks[angle] = np.sum(angle_range)

# Avoid division by zero and compute the final average
power_indeks_nonzero = power_indeks > 0
gns_temp[power_indeks_nonzero] /= power_indeks[power_indeks_nonzero]


ax.scatter(W1[0],W1[1],gns_power)

"""

#fig = plt.figure()
#ax2 = fig.add_subplot(1, 1, 1)
#pf.correlationMatrixPlotter(ax2,station02)
#plt.tight_layout()

plt.show()

