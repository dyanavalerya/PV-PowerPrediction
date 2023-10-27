"Script for ploting the feature vs power for all features in the dataset"

# Importing libraries
import os
import sys
# set syspath to include the base folder
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import home made functions
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf
import pandas as pd
import matplotlib.pyplot as plt

# Defining functions 
def load_all_datasets():
    """
    Load all datasets into one. Add a column with the station number.

    Returns:
    all_data (pandas.DataFrame): A pandas dataframe containing all datasets.
    """
    meta=fl.loadFile(f"metadata.csv")
   
    for i in range(0,9):
        name=f"station0{i}"
        loaded_data=fl.loadFile(f"station0{i}.csv")
        loaded_data["station"] = i
        for row in meta.iterrows():
            if row[1]["Station_ID"]==name:
                loaded_data["power"]=loaded_data["power"]/meta["Capacity"][row[0]]
        if i == 0:
            all_data = loaded_data
            
        else:
            all_data = pd.concat([all_data, loaded_data])
    
    
    return all_data
def plot_feature_vs_power(all_data,save_path,feature_name):
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    pf.plotColumnScatter(ax,all_data,feature_name,"power",f"{feature_name} vs power")
    plt.title(f"Scatterplot of {feature_name} vs power")
    if not os.path.exists(save_path+"/feature_vs_power"):
        os.makedirs(save_path+"/feature_vs_power")
    plt.savefig(f"{save_path}/feature_vs_power/{feature_name}_vs_power.png",format="png")
# Main function
def main():
    # Importing data
    all_data = fl.load_all_datasets()
    print(all_data.head())
    
    all_data=all_data.resample('1H').first()
    # nwp features
    features = ["nwp_globalirrad", "nwp_directirrad", "nwp_temperature", "nwp_humidity", "nwp_windspeed", "nwp_winddirection", "nwp_pressure"]
    # Plotting
    for feature in features:
        pathForFigures=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\Appendix DatasetFeatures"
        plot_feature_vs_power(all_data,pathForFigures,feature)
    
    
# Executing main function
if __name__ == "__main__":
    main()

