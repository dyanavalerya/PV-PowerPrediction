# Importing libraries
import os
import sys
import matplotlib.pyplot as plt
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotFunctions as pf
import fileLoader


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

pf.plotPowCorr(station_data)


plt.show()
