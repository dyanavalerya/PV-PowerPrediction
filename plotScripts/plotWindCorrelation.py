# Simple script to plot wind direction correlation 
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import libraries
import matplotlib.pyplot as plt
from PV_PredictLib import fileLoader
from PV_PredictLib import plotFunctions as pf

station00=fileLoader.loadFile('station00.csv')

correlations=pf.windDirectionCorrelation(station00)


plt.tight_layout()
plt.show()