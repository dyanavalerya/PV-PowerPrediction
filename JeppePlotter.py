import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf
station02 = fileLoader.loadFile("station02.csv")
fileLoader.fileInfo(station02)
station02_sliced= fileLoader.sliceData(station02,"2018-07-23 16:00:00","2018-07-24 16:00:00")


fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
plt.title("Scatter plot of Power vs NWP Pressure") 

pf.plotColumnScatter(ax3,station02,"power","nwp_pressure","Power vs NWP Pressure")

fig = plt.figure()
ax4 = fig.add_subplot(1, 1, 1)
#pf.plotTimeSeries(ax4,station02_sliced,"nwp_globalirrad","Plot of Global Irradiance",12)
pf.plotColumnScatter2Y(ax4,station02_sliced,"date_time","power","nwp_pressure","Time plot of Power vs Global Irradiance",12)
plt.title("Time plot of Power vs Global Irradiance")
plt.tight_layout()



fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
pf.correlationMatrixPlotter(ax2,station02)
ax2.set_title("Correlation matrix of station02 dataset")
plt.tight_layout()
plt.show()
