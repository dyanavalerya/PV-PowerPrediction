import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    ax.xaxis.set_tick_params(rotation=45)
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
def plotColumnScatter2Y(ax :plt.axes ,data : pd.DataFrame,colloumXName : str,colloumY1Name : str,colloumY2Name : str,label: str):
    x = data[colloumXName]
    y1 = data[colloumY1Name]
    y2 = data[colloumY2Name]

    ax.plot(x,y1,".",label=label)
    ax.set_xlabel(colloumXName)
    ax.set_ylabel(colloumY1Name)
    ax2 = ax.twinx()
    # ensure another color for the second y axis
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_ylabel(colloumY2Name,color='red')
    ax2.plot(x,y2,".",label=label,color='red')
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
def nyFunktion():
    print("test")