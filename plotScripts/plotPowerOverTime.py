import os,sys
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import PV_PredictLib.fileLoader as fl
import PV_PredictLib.plotFunctions as pf    
def timeplot(data,name=""):
    fig,ax=plt.subplots(1,1,figsize=(8,6))

    data=fl.sliceData(data,"2018-07-23 16:00:00","2018-07-25 16:00:00")
    pf.plotTimeSeries(ax,data,"power","Power",12)
    # ax right side
    # make points orange
    ax.get_lines()[0].set_color("C0")
    # set label color
    ax.tick_params(axis='y', labelcolor="C0")
    # set label
    ax.set_ylabel("Power [MW]",color="C0")
    axr=ax.twinx()
    pf.plotTimeSeries(axr,data,"nwp_globalirrad","Global irradiance",12*2)
    # make points orange
    axr.get_lines()[0].set_color("C1")
    # set label color
    axr.tick_params(axis='y', labelcolor="C1")
    # set label
    axr.set_ylabel("Global irradiance [W/m^2]",color="C1")
    ax.get_legend().remove()
    axr.get_legend().remove()
    plt.title("Time plot of Power and Global irradiance",size=16)
    plt.tight_layout()
    plt.savefig(r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\ "+name+"Timeplot.png",format="png")
    
def main():
    data=fl.loadPkl("station02.pkl")
    timeplot(data,"Nighttime")
    plt.title("Time plot of Power and Global irradiance excluding nighttime hours",size=16)
    data=fl.loadFile("station02.csv",PKL=False)
    timeplot(data)

    plt.show()
if __name__ == "__main__":
    main()