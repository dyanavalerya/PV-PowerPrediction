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
    plt.plot(st2["nwp_directirrad"],alpha=1,  label="nwp_directirrad")
    plt.plot(st2["hmd_diffuseirrad"],alpha=1, label="hmd_diffuseirrad")
    plt.plot(st2["nwp_globalirrad"],alpha=1,  label="nwp_globalirrad")
    plt.title("Comparison of NWP irradiance ")
    plt.xlabel("Time")
    plt.ylabel("Irradiance [W/m^2]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\NWP_irradiance_comparison.png",format="png")
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    plt.plot(st2["hmd_directirrad"],alpha=1,  label="hmd_directirrad")
    plt.plot(st2["lmd_diffuseirrad"],alpha=1, label="lmd_diffuseirrad")
    plt.plot(st2["lmd_totalirrad"],alpha=1,  label="lmd_totalirrad")
    plt.title("Comparison of LMD irradiance")
    plt.xlabel("Time")
    plt.ylabel("Irradiance [W/m^2]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\LMD_irradiance_comparison.png",format="png")

def main():
    plotIrradiances(fl.loadPkl("station00.pkl"),"2018-07-21 00:00:00","2018-07-30 23:59:59")
if __name__ == "__main__":
    main()