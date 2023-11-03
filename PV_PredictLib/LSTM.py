import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd

def fit_LSTM(trainX,trainY):
    model = Sequential()
    model.add(LSTM(500, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))#lstm lag
    model.add(LSTM(100, activation='relu', return_sequences=False)) #lstm lag
    model.add(Dense(trainY.shape[1]))#NN lag
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
    return model     

def train_model(station):                                                                                                                                                 
    data=fl.loadFile(station+'.csv')
    data=remove_cols(data)

    # normalize the dataset
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    data_scaled = pd.DataFrame(scaler.transform(data))

    #split it to train- and test data
    trainX,trainY,testX,testY=split_data_LSTM(data_scaled)

    print(np.shape(trainX),np.shape(trainY),np.shape(testX),np.shape(testY))

    model=fit_LSTM(trainX,trainY)
    model.save("my_model.keras")



def remove_cols(data):
    cols_to_remove = ['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station']
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

def split_data_LSTM(data):
    n_future = 24 * 4   # Number of samples we want to look into the future based on the past days.
    n_past = 24 * 4   # Number of past samples we want to use to predict the future.

    data_train = data.iloc[:int(data.shape[0] * 0.8), :]
    data_test = data.iloc[int(data.shape[0] * 0.8):, :]
    print(data.shape)
    print(data_train.shape)
    print(data_test.shape)

    def create_sequences(dataset, n_past, n_future):
        X, Y = [], []
        for i in range(n_past, len(dataset) - n_future + 1):
            X.append(dataset.iloc[i - n_past:i, list(range(dataset.shape[1]))].values)
            Y.append(dataset.iloc[i + n_future - 1:i + n_future, -1].values)
        return np.array(X), np.array(Y)
    
    trainX, trainY = create_sequences(data_train, n_past, n_future)
    testX, testY = create_sequences(data_test, n_past, n_future)

    print(f'trainX shape == {trainX.shape}.')
    print(f'trainY shape == {trainY.shape}.')
    print(f'testX shape == {testX.shape}.')
    print(f'testY shape == {testY.shape}.')

    return trainX, trainY, testX, testY
