"Script for ploting the correlation matrix of the station02 dataset"

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
import seaborn as sb 
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def main():
    # Importing results
    file_path = "diffModelResult.pkl"
    temp = open(file_path, 'rb')
    results_df = pickle.load(temp)
    temp.close()  
    
    # Extract the columns
    num_layers = results_df['Num Layers']
    num_neurons = results_df['Num Neurons']
    mse = results_df['MSE']
    
    df_sorted = results_df.sort_values(by='MSE')
    
    mse_dict = {}

    # Loop through unique values and create vectors
    layer1Mean = np.mean(results_df.loc[results_df['Num Layers'] == 1, 'MSE'].values)
    layer2Mean = np.mean(results_df.loc[results_df['Num Layers'] == 2, 'MSE'].values)
    layer3Mean = np.mean(results_df.loc[results_df['Num Layers'] == 3, 'MSE'].values)
    
    neuron100Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 100, 'MSE'].values)
    neuron200Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 200, 'MSE'].values)
    neuron400Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 400, 'MSE'].values)
    neuron800Mean = np.mean(results_df.loc[results_df['Num Neurons'] == 800, 'MSE'].values)
    
    batch2Mean = np.mean(results_df.loc[results_df['Batch Size'] == 2, 'MSE'].values)
    batch4Mean = np.mean(results_df.loc[results_df['Batch Size'] == 4, 'MSE'].values)
    batch8Mean = np.mean(results_df.loc[results_df['Batch Size'] == 8, 'MSE'].values)
    batch16Mean = np.mean(results_df.loc[results_df['Batch Size'] == 16, 'MSE'].values)
    
    val00Mean = np.mean(results_df.loc[results_df['Validation Split'] == 0.0, 'MSE'].values)
    val01Mean = np.mean(results_df.loc[results_df['Validation Split'] == 0.1, 'MSE'].values)
    val02Mean = np.mean(results_df.loc[results_df['Validation Split'] == 0.2, 'MSE'].values)
    

    # Create a scatter plot
    plt.scatter(num_layers, mse, marker='o', color='blue')
    plt.scatter(1, layer1Mean, marker='o',s=100 , color='red', label='Layer 1 Mean')
    plt.scatter(2, layer2Mean, marker='o',s=100, color='red', label='Layer 1 Mean')
    plt.scatter(3, layer3Mean, marker='o',s=100, color='red', label='Layer 1 Mean')

    # Set labels and title
    plt.xlabel('Number of Neurons')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Scatter Plot of Number of Neurons vs MSE')
    



    
if __name__ == "__main__":
    main()
    plt.show()