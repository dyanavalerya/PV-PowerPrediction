"Script for ploting the correlation matrix of the station02 dataset"

# Importing libraries
import os
import sys
# set syspath to include the base folder
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import home made functions
import fileLoader as fl
import plotFunctions as pf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb 


def main():
    # Importing dataset 02
    data = fl.loadFile(f"station02.csv")
    #drop the date_time column
    data = data.drop(columns=["date_time"])
    
    # calculating the correlation matrix
    corrMatrix = data.corr()
    # plotting the correlation matrix
    plt.figure(figsize=(9,6))
    sb.heatmap(data.corr(), annot=True,fmt=".2f", linewidths=.5,  cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation matrix of station02 dataset",size=16)
    plt.tight_layout()
    # save the figure
    save_path=r"C:\Users\jeppe\OneDrive - Aalborg Universitet\7. Semester Shared Work\Project\Figures\Appendix DatasetFeatures/correlation"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    plt.savefig(f"{save_path}/CorrelationMatrixStation2.png",format="png")


    
if __name__ == "__main__":
    main()