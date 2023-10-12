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
    ax.legend()
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

def nwpError(data):
    windspeed_error_squared=[]
    windspeed_error_sum=0
    windspeed_tot=0
    windspeed_N=0

    pressure_error_squared=[]
    pressure_error_sum=0
    pressure_tot=0
    pressure_N=0

    temperature_error_squared=[]
    temperature_error_sum=0
    temperature_tot=0
    temperature_N=0

    globalirrad_error_squared=[]
    globalirrad_error_sum=0
    globalirrad_tot=0
    globalirrad_N=0

    for i in range(len(data)):
        windspeed_error_squared.append((data[i].lmd_windspeed-data[i].nwp_windspeed)**2)
        windspeed_error_sum=np.sum(windspeed_error_squared[i])
        windspeed_tot=windspeed_tot+np.sum(data[i].lmd_windspeed)
        windspeed_N=windspeed_N+len(data[i].lmd_windspeed)

        pressure_error_squared.append((data[i].lmd_pressure-data[i].nwp_pressure)**2)
        pressure_error_sum=np.sum(pressure_error_squared[i])
        pressure_tot=pressure_tot+np.sum(data[i].lmd_pressure)
        pressure_N=pressure_N+len(data[i].lmd_pressure)

        temperature_error_squared.append((data[i].lmd_temperature-data[i].nwp_temperature)**2)
        temperature_error_sum=np.sum(temperature_error_squared[i])
        temperature_tot=temperature_tot+np.sum(data[i].lmd_temperature)
        temperature_N=temperature_N+len(data[i].lmd_temperature)

        globalirrad_error_squared.append((data[i].lmd_totalirrad-data[i].nwp_globalirrad)**2)
        globalirrad_error_sum=np.sum(globalirrad_error_squared[i])
        globalirrad_tot=globalirrad_tot+np.sum(data[i].lmd_totalirrad)
        globalirrad_N=globalirrad_N+len(data[i].lmd_totalirrad)

    
    globalirrad_RMSE=np.sqrt(globalirrad_error_sum/globalirrad_N)
    globalirrad_mean=(globalirrad_tot/globalirrad_N)
    globalirrad_NRMSE=globalirrad_RMSE/globalirrad_mean

    temperature_RMSE=np.sqrt(temperature_error_sum/temperature_N)
    temperature_mean=(temperature_tot/temperature_N)
    temperature_NRMSE=temperature_RMSE/temperature_mean

    pressure_RMSE=np.sqrt(pressure_error_sum/pressure_N)
    pressure_mean=(pressure_tot/pressure_N)
    pressure_NRMSE=pressure_RMSE/pressure_mean

    windspeed_RMSE=np.sqrt(windspeed_error_sum/windspeed_N)
    windspeed_mean=(windspeed_tot/windspeed_N)
    windspeed_NRMSE=windspeed_RMSE/windspeed_mean

    print('temperature RMSE: ',temperature_RMSE)
    print('globalirrad RMSE: ',globalirrad_RMSE)
    print('pressure RMSE: ',pressure_RMSE)
    print('windspeed RMSE: ',windspeed_RMSE)
    print()
    print('temperature NRMSE: ',temperature_NRMSE)
    print('globalirrad NRMSE: ',globalirrad_NRMSE)
    print('pressure NRMSE: ',pressure_NRMSE)
    print('windspeed NRMSE: ',windspeed_NRMSE)
    
    labels = ['Temperature', 'Pressure', 'Windspeed']
    nrmse_values = [temperature_NRMSE, pressure_NRMSE, windspeed_NRMSE]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, nrmse_values, color=['blue', 'red', 'purple'])

    for bar, nrmse_value in zip(bars, nrmse_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(nrmse_value, 2), ha='center', va='bottom', color='black', fontsize=10)

    plt.xlabel('Variables')
    plt.ylabel('NRMSE')
    plt.title('NRMSE for Different Variables')
    plt.show()
    return
