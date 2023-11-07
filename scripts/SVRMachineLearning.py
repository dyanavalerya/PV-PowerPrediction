import imp
import os, sys
from tabnanny import verbose
from ctypes.test.test_macholib import d
import matplotlib.pyplot as plt

# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split
# standardscaler
from sklearn.preprocessing import StandardScaler
# pipeline
from sklearn.pipeline import Pipeline
meta=fileLoader.loadFile("metadata.csv")
SVRModel = SVR(kernel='rbf')
scaler = StandardScaler()
Pipe=Pipeline([('scaler', scaler), ('SVR', SVRModel)])
# define the parasmeters to search
parameters = {'SVR__C':[100,], 'SVR__gamma':[0.001, 0.0001]}
# create the grid search object
Grid = Pipe

def normalizeFromMetadata(data,meta):
    """
    Normalize data using the metadata file. 
    """
    stationNum=data["station"][0]
    meta_station=meta[meta["Station_ID"]==f"station0{stationNum}"]
    # use the capacity of the station to normalize the power
    data["power"]=data["power"]/meta_station["Capacity"][stationNum]
    return data
def loadDataCorrect(data,hoursAhead=24):
    data=data.drop(columns=['station','nwp_pressure','lmd_pressure'])
    # The LMD named features is not known 24 hours in advance, so we shift the data 24 hours back
    for feature in data.columns:
        if feature.startswith("lmd"):
            # theres 4 samples per hour, so we shift 24*4=96 samples back
            # fx if we have lmd at 12:00 we move that measurement to the next day at 12:00
            data[feature]=data[feature].shift(hoursAhead*4)
    # drop the first 96 samples
    data=data.dropna()
    #print(data["lmd_totalirrad"])
    return data
   
stationNum=3
ahead=12
scorelist=[]
for i in range(24,0,-1):
    stationData = fileLoader.loadPkl(f"station0{stationNum}.pkl")

    stationData=loadDataCorrect(stationData,i)
    #stationData=normalizeFromMetadata(stationData,meta)
    features=stationData.drop(columns=['power','date_time'])
    power=stationData['power']
    #print("Max power:",power.max())
    
    X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.3, random_state=42)
    Grid.fit(X_train, y_train)
    # calculate the score of the regression model
    score = Grid.score(X_test, y_test)
    print(f"{i} Hours ahead score:{score}")
    scorelist.append(score)
plt.figure()
plt.plot(scorelist)
plt.show()


#stationNum=4
#stationData = fileLoader.loadFile(f"station0{stationNum}.csv")
#stationData=normalizeFromMetadata(stationData,meta)
#stationData=loadDataCorrect(stationData,ahead)

# features=stationData.drop(columns=['power','date_time'])
# power=stationData['power']
# print("Max power:",power.max())
# X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.3, random_state=42)
# score = Grid.score(X_test, y_test)
# y_pred = Grid.predict(X_test)

# print(score)
# plt.figure()
# plt.plot(y_test.values, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.show()