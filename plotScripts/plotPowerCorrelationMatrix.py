# Importing libraries
import os
import sys
import matplotlib.pyplot as plt
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf

drop = ["nwp_winddirection", "lmd_winddirection"]


station_data = fl.loadAllPkl(drop)
pf.plotPowCorr(station_data)

plt.tight_layout()
plt.show()

