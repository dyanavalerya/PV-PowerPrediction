# Importing libraries
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PV_PredictLib import fileLoader
from PV_PredictLib import plotFunctions as pf

station01=fileLoader.loadFile("station00.csv")

"""
10 picked out days combined in station01Days.
Used as an example for the "individual" plots

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


station01Days = pd.concat((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10), axis=0)
"""

#Can both plot an average of the power for the direction
#Or all the individual data points
pf.circle3dScatterPlot(station01,"average","Station 01")
#pf.circle3dScatterPlot(station01Days,"individual","Station 01")


plt.show()
