import os,sys
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf
import matplotlib.pyplot as plt

def plotIrradiances(data,date1,date2): 
    
    st2=fl.sliceData(data,date1,date2)
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    pf.plotTimeSeries(ax,st2,"nwp_directirrad","nwp_directirrad",24)
    pf.plotTimeSeries(ax,st2,"hmd_diffuseirrad","hmd_diffuseirrad",24)
    pf.plotTimeSeries(ax,st2,"nwp_globalirrad","nwp_globalirrad",24)
    plt.title("Comparison of NWP irradiance ")
    plt.xlabel("Time")
    plt.ylabel("Irradiance [W/m^2]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_irradiance_comparison.png",format="png")
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    pf.plotTimeSeries(ax,st2,"hmd_directirrad","hmd_directirrad",24)
    pf.plotTimeSeries(ax,st2,"lmd_diffuseirrad","lmd_diffuseirrad",24)
    pf.plotTimeSeries(ax,st2,"lmd_totalirrad","lmd_totalirrad",24)
    
    plt.title("Comparison of LMD irradiance")
    plt.xlabel("Time")
    plt.ylabel("Irradiance [W/m^2]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\LMD_irradiance_comparison.png",format="png")
    plt.show()
def main():
    data=fl.loadPkl("station01.pkl")
    #sliced_data=fl.sliceData(data,"2018-07-21 00:00:00","2018-07-30 23:59:59")  
    plotIrradiances(data,"2018-10-21 00:00:00","2018-10-25 23:59:59")
if __name__ == "__main__":
    main()