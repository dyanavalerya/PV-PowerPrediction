import matplotlib.pyplot as plt
# Import functions
import fileLoader
import plotFunctions as pf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split

station02 = fileLoader.loadFile("station02.csv")
fileLoader.fileInfo(station02)
station02_sliced= fileLoader.sliceData(station02,"2018-07-23 16:00:00","2018-07-24 16:00:00")

feature_cols = [ 'nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_pressure','nwp_humidity','nwp_winddirection','nwp_windspeed']
x= station02.loc[:,feature_cols]
y=  station02.loc[:,'power']
#append history to each datapoint, so the model can see what the features was the last hour also add power produced from the last hour
for col in feature_cols:
    x[col+'-1']=x[col].shift(1)
    x[col+'-2']=x[col].shift(2)
    x[col+'-3']=x[col].shift(3)
    x[col+'-4']=x[col].shift(4)
    x[col+'-5']=x[col].shift(5)
    x[col+'-6']=x[col].shift(6)
    x[col+'-7']=x[col].shift(7)
    x[col+'-8']=x[col].shift(8)
    x[col+'-9']=x[col].shift(9)
    x[col+'-10']=x[col].shift(10)
    x[col+'-11']=x[col].shift(11)
    x[col+'-12']=x[col].shift(12)
x['power-1']=y.shift(1)
x['power-2']=y.shift(2)
x['power-3']=y.shift(3) 
x['power-4']=y.shift(4)
x['power-5']=y.shift(5)
x['power-6']=y.shift(6)
x['power-7']=y.shift(7)
x['power-8']=y.shift(8)
x['power-9']=y.shift(9)
x['power-10']=y.shift(10)
x['power-11']=y.shift(11)
x['power-12']=y.shift(12)
#add the columns from last hour to the current hour
x=x.dropna()
y=y[x.index]

    
    
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svr = SVR(kernel='rbf') 
  
# train the model on the data 
svr.fit(X_train, y_train) 
y_pred = svr.predict(X_test)
# calculate the score of the regression model
score = svr.score(X_test, y_test)
print(score)

xsliced=station02_sliced.loc[:,feature_cols]
ysliced=  station02_sliced.loc[:,'power']
#append history to each datapoint, so the model can see what the features was the last hour also add power produced from the last hour
for col in feature_cols:
    xsliced[col+'-1']=xsliced[col].shift(1)
    xsliced[col+'-2']=xsliced[col].shift(2)
    xsliced[col+'-3']=xsliced[col].shift(3)
    xsliced[col+'-4']=xsliced[col].shift(4)
    xsliced[col+'-5']=xsliced[col].shift(5)
    xsliced[col+'-6']=xsliced[col].shift(6)
    xsliced[col+'-7']=xsliced[col].shift(7)
    xsliced[col+'-8']=xsliced[col].shift(8)
    xsliced[col+'-9']=xsliced[col].shift(9)
    xsliced[col+'-10']=xsliced[col].shift(10)
    xsliced[col+'-11']=xsliced[col].shift(11)
    xsliced[col+'-12']=xsliced[col].shift(12)
xsliced['power-1']=ysliced.shift(1)
xsliced['power-2']=ysliced.shift(2)
xsliced['power-3']=ysliced.shift(3) 
xsliced['power-4']=ysliced.shift(4)
xsliced['power-5']=ysliced.shift(5)
xsliced['power-6']=ysliced.shift(6)
xsliced['power-7']=ysliced.shift(7)
xsliced['power-8']=ysliced.shift(8)
xsliced['power-9']=ysliced.shift(9)
xsliced['power-10']=ysliced.shift(10)
xsliced['power-11']=ysliced.shift(11)
xsliced['power-12']=ysliced.shift(12)
#add the columns from last hour to the current hour
xsliced=xsliced.dropna()
ysliced=ysliced[xsliced.index]



y_pred = svr.predict(xsliced)
# change the date time index to start at the same time as the sliced data
new_index=pd.date_range(ysliced.index[0],periods=len(y_pred),freq='15min')

y_pred=pd.Series(y_pred,index=new_index)

fig,ax=plt.subplots()
ax.plot(ysliced,label="Actual")
ax.plot(y_pred,label="Predicted")
plt.legend()
plt.show()  