import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import seaborn as sns


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
    correlation = np.zeros(7)
    vectors = []
    for i in range(len(data)):
        temp = data[i]
        temp = temp.select_dtypes(include=['float64'])
        correlation[0] = temp["power"].corr(temp["nwp_globalirrad"])
        correlation[1] = temp["power"].corr(temp["nwp_directirrad"])
        correlation[2] = temp["power"].corr(temp["nwp_temperature"])
        correlation[3] = temp["power"].corr(temp["nwp_humidity"])
        correlation[4] = temp["power"].corr(temp["nwp_windspeed"])
        correlation[5] = temp["power"].corr(temp["nwp_winddirection"])
        correlation[6] = temp["power"].corr(temp["nwp_pressure"])
        vectors.append(correlation)
        correlation = np.zeros(7)
    powCorrMatrix = np.array(vectors)
    # labels for x-axis
    x_axis_labels = ["NWP Globalirrad","NWP Directirrad","NWP Temperature","NWP Humidity","NWP Windspeed","NWP Winddirection","NWP Pressure"] 
    # labels for y-axis
    y_axis_labels = ["Station00","Station01","Station02","Station03","Station04","Station05","Station06","Station07","Station08","Station09"] 
    powCorrMatrix = pd.DataFrame(powCorrMatrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax = sns.heatmap(powCorrMatrix, ax=ax,vmin = -1, vmax = 1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    ax.set_title("Correlation matrix of power from each recorded station and the corresponding NWP data")
    plt.tight_layout()
  