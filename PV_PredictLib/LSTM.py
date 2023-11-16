import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D,MaxPooling2D,Reshape,ZeroPadding2D,GlobalMaxPooling2D,GRU,Bidirectional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt


def fit_LSTM(trainX,trainY):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(0.2))

    # Add the second LSTM layer
    model.add(LSTM(50,  return_sequences=True))
    model.add(Dropout(0.2))

    # Flatten the output before the dense layer
    model.add(Flatten())

    # Add dense output layer
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.2, verbose=1)
    model.save("model.keras")

    return model



def remove_cols(data):
    cols_to_remove = ['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_humidity','nwp_hmd_diffuseirrad','lmd_hmd_directirrad']
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

def load_LSTM_data(station):
    file_path = station+'LSTM.pkl'
    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)
        print('and have been downloaded')
    else:
        print(f'The file {file_path} does not exist, so the dataset is being split up for the first time')
        data=fl.loadFile(station+'.csv',PKL=False)

        data=remove_cols(data)

        # normalize the dataset
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        data= pd.DataFrame(scaler.transform(data))

        n_future = 24 * 4   # Number of samples we want to look into the future based on the past days.
        n_past = 4 * 4   # Number of past samples we want to use to predict the future.

        data_train = data.iloc[:int(data.shape[0] * 0.8), :]
        data_test = data.iloc[int(data.shape[0] * 0.8):, :]
        print(data.shape)
        print(data_train.shape)
        print(data_test.shape)
        file_path = 'station00LSTM.pkl'

        def create_sequences(dataset, n_past, n_future):
            X, Y = [], []
            for i in range(n_past, len(dataset) - n_future+1 ):
                past=dataset.iloc[i - n_past:i, 4:-1].values
                future=dataset.iloc[i:i+n_future, :4].values
                combined_data = np.concatenate((past, future), axis=0)
                X.append(combined_data)
                Y.append(dataset.iloc[i:i+n_future, -1].values)
            return np.array(X), np.array(Y)
        
        trainX, trainY = create_sequences(data_train, n_past, n_future)
        testX, testY = create_sequences(data_test, n_past, n_future)

        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

        print(f'trainX shape == {trainX.shape}.')
        print(f'trainY shape == {trainY.shape}.')
        print(f'testX shape == {testX.shape}.')
        print(f'testY shape == {testY.shape}.')

    return trainX, trainY, testX, testY


def load_LSTM_data_for_day_only(station):
    file_path = station+'LSTM_day_only.pkl'
    if os.path.isfile(file_path):
        print(f'The file {file_path} exists ')
        with open(file_path, 'rb') as file:
            trainX = pickle.load(file)
            trainY = pickle.load(file) 
            testX = pickle.load(file) 
            testY = pickle.load(file)
            first_day_in_test= pickle.load(file)

        print('and have been downloaded')
    else:
        print(f'The file {file_path} does not exist, so the dataset is being split up for the first time')
        data_day=fl.loadFile(station+'.csv')
        data_night=fl.loadFile(station+'.csv',PKL=False)
        first_day_in_test=data_day.iloc[int(data_day.shape[0] * 0.8), 0]
        first_day_in_test = pd.to_datetime(first_day_in_test)

        data_day=remove_cols(data_day)
        data_night=remove_cols(data_night)
        data_night = data_night.loc[data_night.index >= first_day_in_test]
        data_night = data_night[data_day.columns]

        # normalize the dataset
        scaler = MinMaxScaler()
        scaler = scaler.fit(data_day)
        data_day= pd.DataFrame(scaler.transform(data_day))
        data_night= pd.DataFrame(scaler.transform(data_night))

        n_future = 24 * 4   # Number of samples we want to look into the future based on the past days.
        n_past = 1   # Number of past samples we want to use to predict the future.

        data_train = data_day.iloc[:int(data_day.shape[0] * 0.8), :]
        #data_test = data.iloc[int(data.shape[0] * 0.8):, :]
        data_test=data_night
        print(data_day.shape)
        print(data_train.shape)
        #print(data_test.shape)

        def create_sequences(dataset, n_past, n_future):
            X, Y = [], []
            for i in range(n_past, len(dataset) - n_future+1 ):
                past=dataset.iloc[i - n_past:i, 4:-1].values
                future=dataset.iloc[i:i+n_future, :4].values
                combined_data = np.concatenate((past, future), axis=0)
                X.append(combined_data)
                Y.append(dataset.iloc[i:i+n_future, -1].values)
            return np.array(X), np.array(Y)
        
        trainX, trainY = create_sequences(data_train, n_past, n_future)
        testX, testY = create_sequences(data_test, n_past, n_future)

        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)
            pickle.dump(first_day_in_test,file)

        print(f'trainX shape == {trainX.shape}.')
        print(f'trainY shape == {trainY.shape}.')
        print(f'testX shape == {testX.shape}.')
        print(f'testY shape == {testY.shape}.')

    return trainX, trainY, testX, testY,first_day_in_test

