# Simple script to plot wind direction correlation 
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PV_PredictLib import plotFunctions as pf
from PV_PredictLib import fileLoader as fl
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

corr=[]
station_names = []

for i in range(10):
    station_names.append(f'station0{i}.csv')

    data=fl.loadFile(station_names[i])
    corr.append(pf.windDirectionCorrelation(data))
    print(np.shape(corr))

corr=np.transpose(corr)

plt.figure(figsize=(8,2.5))
heatmap=sb.heatmap(corr,annot=True,cbar=False,linewidths=.5)
plt.xticks(np.arange(len(station_names)) + 0.5, station_names, rotation=45)
plt.yticks([0.5, 1.5], ['lmd_winddirection', 'nwp_winddirection'], rotation=0)
plt.title('Correlations between winddirection and power for each station')
plt.tight_layout()
plt.show()
    