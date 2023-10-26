import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import numpy as np


def plotBase(ax,x,y,label):
    ax.plot(x,y,".",label=label)
    ax.legend()
    
    

def plotTimeSeries(ax :plt.axes ,data : pd.DataFrame,colloumName : str,label: str ,scaleTicks : float= 2):
    x = data["date_time"]
    y = data[colloumName]
    plotBase(ax,x,y,label)
    ax.set_xlabel("Time")
    ax.set_ylabel(colloumName)
    #set ticks 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=scaleTicks))
    ax.xaxis.set_tick_params(rotation=90)
    #set ticks to be 90 rotated
    #ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    return ax    

def plotColumnScatter(ax :plt.axes ,data : pd.DataFrame,colloum1Name : str,colloum2Name : str,label: str):
    x = data[colloum1Name]
    y = data[colloum2Name]
    plotBase(ax,x,y,label)
    ax.set_xlabel(colloum1Name)
    ax.set_ylabel(colloum2Name)
    #set legend location to upper right
    ax.legend(loc='upper right')
    return ax
# plot columnScatter with two y axis's
def plotColumnScatter2Y(ax :plt.axes ,data : pd.DataFrame,colloumXName : str,colloumY1Name : str,colloumY2Name : str,label: str ,scaleTicks : float= 2):
    x = data[colloumXName]
    y1 = data[colloumY1Name]
    y2 = data[colloumY2Name]
    ax.spines['right'].set_color(c='C0')
    ax.tick_params(axis='y', colors='C0')
    ax.set_ylabel(colloumY2Name,color='C0')
    ax.plot(x,y1,".",label=label,color='C0')
    ax.set_xlabel(colloumXName)
    ax.set_ylabel(colloumY1Name)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=scaleTicks))
    ax.xaxis.set_tick_params(rotation=90)
    ax2 = ax.twinx()
    # ensure another color for the second y axis
    ax2.spines['right'].set_color('C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.set_ylabel(colloumY2Name,color='C1')
    ax2.plot(x,y2,".",label=label,color='C1')
    ax2.set_ylabel(colloumY2Name)
    return ax
def plotHistogram(ax :plt.axes ,data : pd.DataFrame,colloumName : str,label: str,binCount=20):
    x = data[colloumName]
    #normalize density for better histogram
    x.plot.hist(ax=ax,label=label+" histogram",density=True,bins=binCount)
    x.plot.kde(ax=ax,legend=True,label=label+" kde")
    ax.set_xlabel(colloumName)
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax

def correlationMatrixPlotter(ax :plt.axes ,data : pd.DataFrame):
    import seaborn as sns
    #drop non float colloums
    data2 = data.select_dtypes(include=['float64'])
    ax = sns.heatmap(data2.corr(), ax=ax,annot=True, fmt=".1f")
    return ax    

def plot_means_and_variances(stats):
    """
    Plot the means and variances for each corresponding column.
    
    Parameters:
        stats (pd.DataFrame): DataFrame containing 'DataFrame', 'Column', 'Average', and 'Variance'.
    """
    unique_columns = stats['Column'].unique()

    for col in unique_columns:
        col_stats = stats[stats['Column'] == col]
        data_frames = col_stats['DataFrame']
        averages = col_stats['Average']
        variances = col_stats['Variance']

        plt.figure(figsize=(10, 5))
        
        # Scatter plot with DataFrame numbers as legends
        for i, (variance, average, dataframe) in enumerate(zip(variances, averages, data_frames), 1):
            plt.scatter(variance, average, label=f'DF {dataframe}', c=f'C{i}', cmap='viridis')
        
        plt.xlabel('Variance')
        plt.ylabel('Mean []')
        plt.title(f'Means vs. Variances for Column {col}')
        plt.legend(title='DataFrame Number')
        plt.show()


def nyFunktion(ax :plt.axes ,data : pd.DataFrame):
    import seaborn as sns
    data2 = data.select_dtypes(include=['float64'])
    print(data2.corr())

def load_all_datasets():
    import fileLoader as fl
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

def nwpError():
    from sklearn.metrics import mean_squared_error

    data=load_all_datasets()

    MSE_windspeed=mean_squared_error(data.lmd_windspeed,data.nwp_windspeed)
    RMSE_windspeed=np.sqrt(MSE_windspeed)
    NRMSE_windspeed=RMSE_windspeed/np.mean(data.lmd_windspeed)
    print('windspeed NRMSE: ',NRMSE_windspeed)

    MSE_pressure=mean_squared_error(data.lmd_pressure,data.nwp_pressure)
    RMSE_pressure=np.sqrt(MSE_pressure)
    NRMSE_pressure=RMSE_pressure/np.mean(data.lmd_pressure)
    print('pressure NRMSE: ',NRMSE_pressure)

    MSE_temperature=mean_squared_error(data.lmd_temperature,data.nwp_temperature)
    RMSE_temperature=np.sqrt(MSE_temperature)
    NRMSE_temperature=RMSE_temperature/np.mean(data.lmd_temperature)
    print('temperature NRMSE: ',NRMSE_temperature)

    MSE_globalirrad=mean_squared_error(data.lmd_totalirrad,data.nwp_globalirrad)
    RMSE_globalirrad=np.sqrt(MSE_globalirrad)
    NRMSE_globalirrad=RMSE_globalirrad/np.mean(data.lmd_totalirrad)
    print('globalirrad NRMSE: ',NRMSE_globalirrad)


    labels = ['Temperature', 'Pressure', 'Wind speed', 'Global Irradiance']  # Updated labels
    nrmse_values = [NRMSE_temperature, NRMSE_pressure, NRMSE_windspeed, NRMSE_globalirrad]  # Updated NRMSE values

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, nrmse_values, color=['blue', 'red', 'purple', 'green'])  # Added color for Global Irradiance

    for bar, nrmse_value in zip(bars, nrmse_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(nrmse_value, 3), ha='center', va='bottom', color='black', fontsize=10)

    plt.xlabel('Variables')
    plt.ylabel('NRMSE')
    plt.title('NRMSE for Different Variables')
    plt.show()

    
    return


def windDirectionCorrelation(data):
    # Extract columns
    power = data['power']
    wind_direction_lmd = data['lmd_winddirection']
    wind_direction_nwp = data['nwp_winddirection']

    # Convert wind_direction to radians
    wind_direction_rad_lmd = np.radians(wind_direction_lmd)
    wind_direction_rad_nwp = np.radians(wind_direction_nwp)

    # Create matrix A
    A_lmd = np.vstack((np.sin(wind_direction_rad_lmd), np.cos(wind_direction_rad_lmd))).T
    A_nwp = np.vstack((np.sin(wind_direction_rad_nwp), np.cos(wind_direction_rad_nwp))).T

    # Calculate conditioning number of A
    conditioning_number_lmd = np.linalg.cond(A_lmd)
    conditioning_number_nwp = np.linalg.cond(A_nwp)

    # Calculate coefficients
    x_n_lmd = np.linalg.inv(A_lmd.T @ A_lmd) @ A_lmd.T @ power
    x_n_nwp = np.linalg.inv(A_nwp.T @ A_nwp) @ A_nwp.T @ power

    # Calculate regression
    Reg_lmd = A_lmd @ x_n_lmd
    Reg_nwp = A_nwp @ x_n_nwp

    # Calculate correlation coefficient
    correlation_lmd = np.corrcoef(Reg_lmd, power)[0, 1]
    correlation_nwp = np.corrcoef(Reg_nwp, power)[0, 1]

    print("Conditioning number lmd:", conditioning_number_lmd,'; nwp:',conditioning_number_nwp)
    print("Coefficients (a1, a2) for lmd:", x_n_lmd,'; for nwp:',x_n_nwp)
    print("Correlation coefficient lmd:", correlation_lmd,'; for nwp:',correlation_nwp)
    return correlation_lmd,correlation_nwp
