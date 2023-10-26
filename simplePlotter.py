#Simple example of how to read and plot data from the PVO dataset
# Import libraries
import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf
import numpy as np


station00=fileLoader.loadFile('station00.csv')

correlations=pf.windDirectionCorrelation(station00)
print(np.shape(correlations))
print(correlations)
