import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt

def LSTM_code():
    trainX,trainY,testX,testY=LSTM.load_LSTM_data('station00')
    print(f'trainX shape == {trainX.shape}.')
    print(f'trainY shape == {trainY.shape}.')
    print(f'testX shape == {testX.shape}.')
    print(f'testY shape == {testY.shape}.')

    #fit_LSTM(trainX,trainY)

    reconstructed_model = keras.models.load_model("my_model.keras")
    pred=reconstructed_model.predict(trainX)

    x = list(range(testY.shape[1]))
    plt.figure()
    plt.plot(x,pred[:,0], label='predicted')
    plt.plot(x,testY[:,0], label='original')
    plt.legend()
    plt.show()

    print('done')


def fit_LSTM(trainX,trainY):
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))#lstm lag
    model.add(LSTM(100, activation='relu', return_sequences=False)) #lstm lag
    model.add(Dense(trainY.shape[1]))#NN lag
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(trainX, trainY, epochs=5, batch_size=1000, validation_split=0.1, verbose=1)
    model.save("my_model.keras")
    return model     



def remove_cols(data):
    cols_to_remove = ['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_humidity']
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
                past=dataset.iloc[i - n_past:i, :4].values
                future=dataset.iloc[i:i+n_future, 4:-1].values
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