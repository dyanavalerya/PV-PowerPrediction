import numpy as np
import os, sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}") 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
from matplotlib import pyplot as plt

def LSTM_code():
    trainX,trainY,testX,testY=load_LSTM_data('station00')
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


def fit_LSTM(trainX,trainY,save_file):
    """
    input_shape = (trainX.shape[1], trainX.shape[2])

    # Create a sequential model
    model = models.Sequential()

    # Add layers to the model
    model.add(layers.Flatten(input_shape=input_shape))  # Flatten the input
    model.add(layers.Dense(1000, activation='relu'))      # Dense layer with 128 units and ReLU activation                      # Dropout layer for regularization
    model.add(layers.Dense(1000, activation='relu'))       # Another Dense layer with 64 units and ReLU activation
    model.add(layers.Dense(trainY.shape[1], activation='relu'))    # Output layer with 10 units for classification (adjust as needed)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Display the model summary
    model.summary()
    """

    
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))#lstm lag
    model.add(LSTM(100, activation='relu', return_sequences=False)) #lstm lag
    model.add(Dense(trainY.shape[1]))#NN lag
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)
    model.save(save_file)
    return model     

def split_dataframe_columns(df):
    """
    Splits a DataFrame into three based on column prefixes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: A tuple containing three DataFrames - (lmd_df, nwp_df, power_df).
    """
    lmd_columns = [col for col in df.columns if col.startswith("lmd")]
    nwp_columns = [col for col in df.columns if col.startswith("nwp")]
    power_columns = [col for col in df.columns if col.startswith("power")]

    lmd_df = df[lmd_columns]
    nwp_df = df[nwp_columns]
    power_df = df[power_columns]

    return lmd_df, nwp_df, power_df

def normalize_dataframes(*dfs):
    """
    Normalizes a list of DataFrames using Min-Max scaling.

    Parameters:
    - dfs (pd.DataFrame): Variable number of DataFrames to normalize.

    Returns:
    tuple: A tuple containing the normalized DataFrames.
    """
    scaler = MinMaxScaler()

    normalized_dfs = tuple(scaler.fit_transform(df) for df in dfs)

    return normalized_dfs


def remove_cols(data, cols_to_remove=['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'nwp_pressure', 'date_time', 'station','nwp_humidity']):
    cols = [col for col in data.columns if col not in cols_to_remove]
    print('columns that are used: ',cols)
    data=data[cols]
    return data

def load_LSTM_data(station,cols_to_remove=None,n_future = 24 * 4,n_past = 4 * 4):
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

        data=remove_cols(data,cols_to_remove)
        
        lmd_data, nwp_data, power_data = split_dataframe_columns(data)

        # normalize the dataset
        if lmd_data.shape[1]>0:
            normalized_lmd, normalized_nwp, normalized_power = normalize_dataframes(lmd_data, nwp_data, power_data)
            normalized_lmd_train = normalized_lmd[:int(normalized_lmd.shape[0] * 0.8), :]
            normalized_lmd_test = normalized_lmd[int(normalized_lmd.shape[0] * 0.8):, :]
        else:
            normalized_nwp, normalized_power = normalize_dataframes(nwp_data, power_data)
        
        
        normalized_nwp_train = normalized_nwp[:int(normalized_nwp.shape[0] * 0.8), :]
        normalized_nwp_test = normalized_nwp[int(normalized_nwp.shape[0] * 0.8):, :]
        normalized_power_train = normalized_power[:int(normalized_power.shape[0] * 0.8), :]
        normalized_power_test = normalized_power[int(normalized_power.shape[0] * 0.8):, :]
        
        file_path = 'station00LSTM.pkl'

        def create_sequences(lmd_data=None,nwp_data=None,power_data=None, n_past=4*4, n_future=24*4):
            X, Y = [], []
            if n_past>0:
                for i in range(n_past, len(nwp_data) - n_future+1 ):
                    past=lmd_data[i - n_past:i, :lmd_data.shape[1]]
                    future=nwp_data[i:i+n_future, :nwp_data.shape[1]]
                    combined_data = np.concatenate((past, future), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i+n_future])
            else:
                for i in range(0, len(nwp_data) - n_future+1 ):
                    future=nwp_data[i:i+n_future, :nwp_data.shape[1]]
                    X.append(future)
                    Y.append(power_data[i:i+n_future])
            return np.array(X), np.array(Y)
        if lmd_data.shape[1]>0:
            trainX, trainY = create_sequences(normalized_lmd_train,normalized_nwp_train,normalized_power_train, n_past, n_future)
            testX, testY = create_sequences(normalized_lmd_test,normalized_nwp_test,normalized_power_test, n_past, n_future)            
        else:
            trainX, trainY = create_sequences(nwp_data=normalized_nwp_train,power_data=normalized_power_train, n_past=n_past, n_future=n_future)
            testX, testY = create_sequences(nwp_data=normalized_nwp_test,power_data=normalized_power_test, n_past=n_past, n_future=n_future)

        
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


