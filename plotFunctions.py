import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import seaborn as sns
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


def powerHeatMap(ax :plt.axes ,data : pd.DataFrame):
    data2 = data.select_dtypes(include=['float64'])
    print(data2.corr())



def plotPowCorr(data):
    """
    This function plots a heatmap of the correlation between power and NWP data for each power station
    """
    correlation = np.zeros(13)
    vectors = []
    for i in range(len(data)):
        temp = data[i]
        #temp = temp.select_dtypes(include=['float64'])
        correlation[0] = temp["power"].corr(temp["nwp_globalirrad"])
        correlation[1] = temp["power"].corr(temp["nwp_directirrad"])
        correlation[2] = temp["power"].corr(temp["nwp_temperature"])
        correlation[3] = temp["power"].corr(temp["nwp_humidity"])
        correlation[4] = temp["power"].corr(temp["nwp_windspeed"])
        correlation[5] = temp["power"].corr(temp["nwp_winddirection"])
        correlation[6] = temp["power"].corr(temp["nwp_pressure"])
        correlation[7] = temp["power"].corr(temp["lmd_totalirrad"])
        correlation[8] = temp["power"].corr(temp["lmd_diffuseirrad"])
        correlation[9] = temp["power"].corr(temp["lmd_temperature"])
        correlation[10] = temp["power"].corr(temp["lmd_pressure"])
        correlation[11] = temp["power"].corr(temp["lmd_winddirection"])
        correlation[12] = temp["power"].corr(temp["lmd_windspeed"])
        vectors.append(correlation)
        correlation = np.zeros(13)
    powCorrMatrix = np.array(vectors)
    # labels for x-axis
    x_axis_labels = ["NWP Globalirrad","NWP Directirrad","NWP Temperature","NWP Humidity","NWP Windspeed","NWP Winddirection","NWP Pressure", "LMD Totalirrad", "LMD Diffuseirrad", "LMD Temperature", "LMD Pressure", "LMD Winddirection", "LMD Windspeed"] 
    # labels for y-axis
    y_axis_labels = ["Station00","Station01","Station02","Station03","Station04","Station05","Station06","Station07","Station08","Station09"] 
    powCorrMatrix = pd.DataFrame(powCorrMatrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax = sns.heatmap(powCorrMatrix, ax=ax,vmin = -1, vmax = 1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt=".2f")
    ax.set_title("Correlation matrix of power and each recorded feature from the 10 stations", fontsize=20)
    plt.tight_layout()
  
  
def circle3dScatterPlot(dataFrame,setting,namestring):
    """
    This functions makes a scatter plot in 3d of 
    the wind drection converted intro sin an cos values
    It can either be set to plot the "average" of 
    measured power in 1 degrees intervals or plot the
    "individual" points of data
    """
    #name  = globals()[dataFrame]
    if setting=="average":
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,1,1,projection='3d')


        nwp_winddirection = dataFrame["nwp_winddirection"].to_numpy()
        power = dataFrame.iloc[:, 14].to_numpy()

        # Initialize arrays to store sums and counts for each angle
        gns_power = np.zeros(360, dtype=float)
        power_indeks = np.zeros(360, dtype=int)

        for angle in range(360):
            angle_range = (angle <= nwp_winddirection) & (nwp_winddirection <= angle + 1)

            # Calculate sums and counts for the current angle
            gns_power[angle] = np.sum(power[angle_range])
            power_indeks[angle] = np.sum(angle_range)

        # Avoid division by zero and compute the final average
        power_indeks_nonzero = power_indeks > 0
        gns_power[power_indeks_nonzero] /= power_indeks[power_indeks_nonzero]


        x = np.array(list(range(360)) )

        W1 = [np.cos(x*np.pi/180), np.sin(x*np.pi/180)]
        ax.scatter(W1[0],W1[1],gns_power)

        ax.set_xlabel('Cosinus')
        ax.set_ylabel('Sinus')
        ax.set_zlabel('average power [MW]')
        ax.set_title(namestring + ' average power compared to wind direction')
        
    elif setting=="individual":
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        windDirConvert = [np.cos(nwp_winddirection*np.pi/180), np.sin(nwp_winddirection*np.pi/180)]


        ax.scatter(windDirConvert[0],windDirConvert[1],power)

        ax.set_xlabel('Cosinus')
        ax.set_ylabel('Sinus')
        ax.set_zlabel('power [MW]')
        ax.set_title(namestring + ' average powewr compared to wind direction')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
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
    NRMSE_windspeed=RMSE_windspeed/np.msean(data.lmd_windspeed)
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
