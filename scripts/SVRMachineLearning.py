from math import e
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split
# standardscaler
from sklearn.preprocessing import StandardScaler
# pipeline
from sklearn.pipeline import Pipeline
import sklearn as sk

def loadDataAndSplit(stationNum,testSize=1000):
    stationData2 = fileLoader.loadPkl(f"station0{stationNum}.pkl")
    # if ModelsSVR file exists then load it
    stationTrain=stationData2[:-testSize]
    stationTest=stationData2[-testSize:]
    return stationTrain,stationTest
def normalizeFromMetadata(data):
    """
    Normalize data using the metadata file. 
    """
    meta=fileLoader.loadFile("metadata.csv")

    stationNum=data["station"][0]
    meta_station=meta[meta["Station_ID"]==f"station0{stationNum}"]
    # use the capacity of the station to normalize the power
    data["power"]=data["power"]/meta_station["Capacity"][stationNum]
    return data
def loadDataCorrect(data,hoursAhead=24.0):
    data=data.drop(columns=['station','nwp_pressure','lmd_pressure'])
    # The LMD named features is not known 24 hours in advance, so we shift the data 24 hours back
    for feature in data.columns:
        if feature.startswith("lmd"):
            # theres 4 samples per hour, so we shift 24*4=96 samples back
            # fx if we have lmd at 12:00 we move that measurement to the next day at 12:00
            data[feature]=data[feature].shift(hoursAhead)
    # drop the first 96 samples
    data=data.dropna()
    #print(data["lmd_totalirrad"])
    return data
def splitContinuousData(data):
    features=data.drop(columns=['power','date_time'])
    power=data['power']
    # cutout 10% of the data
    featuresTest=features[-500:]
    features=features[:-500]
    powerTest=power[-500:] 
    power=power[:-500] 
    return features,featuresTest,power,powerTest
def trainModel(model,features,power):
    
    X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.3, random_state=42)
    clonedPipe=sk.clone(model).fit(X_train, y_train)
    return clonedPipe
def saveModels(Models,subfix):
    file = open(f"ModelsSVR{subfix}.pkl", "wb")
    pickle.dump(Models, file)
    file.close()
def train96ModelsLMDAhead(subfix,model,stationTrain):
    Models=[]
    if not os.path.isfile(f"ModelsSVR{subfix}.pkl"):
        for i in range(0,96,1):
            stationData=loadDataCorrect(stationTrain,i)
            [features,featuresTest,power,powerTest]=splitContinuousData(stationData)
            Models.append((trainModel(model,features, power)))
            # calculate the score of the regression model
            score = Models[i].score(featuresTest, powerTest)
            print(f"{i/4} Hours ahead validation score:{score}")
        saveModels(Models,subfix)
def predict96ModelsLMDAhead(Models_,stationTest):
    predictedValues=np.zeros((96,1000))
    scorelist=[]
    features=stationTest.drop(columns=['power','date_time','lmd_pressure','nwp_pressure','station'])
    power=stationTest['power']
    i=0
    for model in Models_:
        score =model.score(features, power)
        scorelist.append(score) 
        predictedValues[i,:]=(model.predict(features))
        i=i+1
    return predictedValues,scorelist

stationTrain,stationTest=loadDataAndSplit(3)


PipeLinear=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='linear'))])
PipeRadial=Pipeline([('scaler', StandardScaler()), ('SVR', SVR(kernel='rbf',epsilon=1))])
# Train 96 models
train96ModelsLMDAhead("linear",PipeLinear,stationTrain)
train96ModelsLMDAhead("rbf_elipson_0.2",PipeRadial,stationTrain)
# load models generated
Modelsrbf=pickle.load(open(f"ModelsSVR{'rbf'}.pkl", "rb"))
Modelslinear=pickle.load(open(f"ModelsSVR{'rbf_elipson_0.2'}.pkl", "rb"))
# predict on the 96 models
predictedValuesrbf,scorelistrbf=predict96ModelsLMDAhead(Modelsrbf,stationTest)
predictedValueslinear,scorelistlinear=predict96ModelsLMDAhead(Modelslinear,stationTest)
# Plot results
plt.figure()
plt.plot(stationTest['power'].values,label="Actual")
plt.plot(predictedValuesrbf[0,:],label="Predicted 0 hours ahead")
plt.plot(predictedValuesrbf[3,:],label="Predicted 1 hours ahead")
plt.plot(predictedValuesrbf[95,:],label="Predicted 24 hours ahead")
plt.legend()
plt.title("SVR prediction of power vs Time of known LMD features")
# save file with models

plt.figure()
X=np.arange (0, 24, 0.25)
plt.plot(X,scorelistrbf,label="Score of radial kernel")
plt.plot(X,scorelistlinear,label="Score of linear kernel")
plt.legend()
plt.xlabel("Hours ahead of prediction")
plt.ylabel("Score")
plt.title("Score of SVR model vs X hours ahead of prediction(LMD features)")
plt.show()


