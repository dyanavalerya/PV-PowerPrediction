import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import clearSkyModels as cs
import numpy as np
import matplotlib.pyplot as plt
import pysolar.solar as sun
from datetime import datetime, timedelta
import sys,os

# I have only found the panel effectivity for station0

st_meta=fl.loadFile("metadata.csv")
station_data=fl.loadFile('station00.csv')

lmd_tot=[]
lmd_diff=[]
lmd_DNI=[]
power=[]
data=[]
n=500
b=2000
for i in range(n):
    new_time=station_data.iloc[i+b,0]
    data.append(cs.lmd_clear_sky(st_meta,station_data,0, new_time))
    lmd_tot.append(station_data.loc[new_time, 'lmd_totalirrad'])
    lmd_diff.append(station_data.loc[new_time, 'lmd_diffuseirrad'])
    lmd_DNI.append(station_data.loc[new_time, 'hmd_directirrad'])
    power.append(station_data.loc[new_time, 'power'])

csm_data=[row[0] for row in data]
panel_irrad=[row[1] for row in data]
watt_pr_panel=[row[2] for row in data]
station_watt_tot=[row[3] for row in data]
station_panel_megawatt_tot=[row[4]/1000000 for row in data]
station_panel_watt_tot=[row[4] for row in data]
csm_dni=[row[0] for row in csm_data]
csm_diff=[row[1] for row in csm_data]
csm_tot=[row[2] for row in csm_data]

x = list(range(1, n+1))
plt.figure()
plt.plot(x, station_panel_megawatt_tot)
plt.plot(x, power)
plt.xlabel('Time')
plt.ylabel('(W/m^2)')
plt.legend(['clear sky power','maes power'])
plt.show()