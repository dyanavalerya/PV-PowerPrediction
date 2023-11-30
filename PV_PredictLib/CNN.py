import os,sys
print(f"Setting syspath to include base folder: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Conv2D,MaxPooling2D,Reshape,ZeroPadding2D,GlobalMaxPooling2D,GRU,Bidirectional
from PV_PredictLib import fileLoader as fl
import numpy as np
import pandas as pd
import pickle
import keras
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import LSTM as LS
from sklearn.metrics import r2_score, mean_squared_error
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin
from keras.wrappers.scikit_learn import KerasRegressor

trainX, trainY, testX, testY = LS.load_LSTM_zero_padded('station02', 24 * 4, 24 * 4)

def load_CNN_data(station, cols_to_remove=None, n_future=24 * 4, n_past=1):
    station_name = os.path.splitext(station)[0]
    file_format = os.path.splitext(station)[1]
    file_path = station_name + 'LSTM.pkl'
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
        if file_format == '.csv':
            data = fl.loadFile(station_name + '.csv', PKL=False)
        else:
            data = fl.loadPkl(station_name + '.pkl')

        data = LS.remove_cols(data, cols_to_remove)

        lmd_data, nwp_data, power_data = LS.split_dataframe_columns(data)

        # normalize the dataset
        if lmd_data.shape[1] > 0:
            normalized_lmd, normalized_nwp, normalized_power = LS.normalize_dataframes(lmd_data, nwp_data, power_data)
            normalized_lmd_train = normalized_lmd[:int(normalized_lmd.shape[0] * 0.8), :]
            normalized_lmd_test = normalized_lmd[int(normalized_lmd.shape[0] * 0.8):, :]
        else:
            normalized_nwp, normalized_power = LS.normalize_dataframes(nwp_data, power_data)

        normalized_nwp_train = normalized_nwp[:int(normalized_nwp.shape[0] * 0.8), :]
        normalized_nwp_test = normalized_nwp[int(normalized_nwp.shape[0] * 0.8):, :]
        normalized_power_train = normalized_power[:int(normalized_power.shape[0] * 0.8), :]
        normalized_power_test = normalized_power[int(normalized_power.shape[0] * 0.8):, :]

        file_path = station_name + 'LSTM.pkl'

        def create_sequences(lmd_data=None, nwp_data=None, power_data=None, n_past=1, n_future=24 * 4):
            X, Y = [], []
            if n_past > 0:
                for i in range(n_past, len(nwp_data) - n_future + 1):
                    past = lmd_data[i - n_past:i, :lmd_data.shape[1]]
                    future = nwp_data[i:i + n_future, :nwp_data.shape[1]]
                    combined_data = np.concatenate((past.flatten(), future.flatten()), axis=0)
                    X.append(combined_data)
                    Y.append(power_data[i:i + n_future])
            else:
                for i in range(0, len(nwp_data) - n_future + 1):
                    future = nwp_data[i:i + n_future, :nwp_data.shape[1]]
                    X.append(future)
                    Y.append(power_data[i:i + n_future])
            return np.array(X), np.array(Y)

        if lmd_data.shape[1] > 0:
            trainX, trainY = create_sequences(normalized_lmd_train, normalized_nwp_train, normalized_power_train,
                                              n_past, n_future)
            testX, testY = create_sequences(normalized_lmd_test, normalized_nwp_test, normalized_power_test, n_past,
                                            n_future)
        else:
            trainX, trainY = create_sequences(nwp_data=normalized_nwp_train, power_data=normalized_power_train,
                                              n_past=n_past, n_future=n_future)
            testX, testY = create_sequences(nwp_data=normalized_nwp_test, power_data=normalized_power_test,
                                            n_past=n_past, n_future=n_future)

        # Open a file for writing
        with open(file_path, 'wb') as file:
            pickle.dump(trainX, file)
            pickle.dump(trainY, file)
            pickle.dump(testX, file)
            pickle.dump(testY, file)

    return trainX, trainY, testX, testY

def fit_CNN(trainX, trainY, save_file, num_neurons=300, epochs = 3, validation_split = 0.1):
    # input must be of shape [batch_size, time_steps, input_dimension]
    n_timesteps = trainX.shape[1]  # 1
    n_features = trainX.shape[2]  # 5
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    # filters is the nr. of neurons
    # kernel size here is 1D because of the type of CNN, which is Conv1D
    model.add(keras.layers.Conv1D(filters=num_neurons, kernel_size=40, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Conv1D(filters=100, kernel_size=13, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.MaxPooling1D(pool_size=1))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
    # We want to pass in 1 because this is the output layer and we only want to predict one thing which is power
    # Otherwise if we predict 24 hrs ahead for each time step, then we have 96 values that the Dense layer should output
    model.add(keras.layers.Dense(trainY.shape[1], name="Dense_2"))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])
    model.fit(trainX, trainY, epochs=epochs,
              validation_split=validation_split, verbose=1)
    model.save(save_file)
    return model

def fit_CNN_tune(hp):
    # input must be of shape [batch_size, time_steps, input_dimension]
    #trainX_reshaped = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    n_timesteps = trainX.shape[1]  # 104
    n_features = trainX.shape[2]  # 5
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    # filters is the nr. of neurons
    # kernel size here is 1D because of the type of CNN, which is Conv1D
    # model.add(keras.layers.Conv1D(hp.Int('conv1_units', min_value=300, max_value=500, step=16),
    #                               kernel_size=21, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Conv1D(filters=364, kernel_size=hp.Int('kernel1_size', min_value=13, max_value=41, step=3),
                                  dilation_rate=2, activation='relu', name="Conv1D_1"))

    model.add(keras.layers.MaxPooling1D(hp.Int('pool1_size', min_value=1, max_value=3, step=1)))

    model.add(keras.layers.Conv1D(filters=126, kernel_size=hp.Int('kernel2_size', min_value=3, max_value=21, step=3),
                                  dilation_rate=2, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.MaxPooling1D(hp.Int('pool2_size', min_value=1, max_value=3, step=1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('filter1_size', min_value=10, max_value=600, step=90), activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(hp.Int('filter2_size', min_value=10, max_value=300, step=50), activation='relu', name="Dense_2"))
    model.add(keras.layers.Dense(hp.Int('filter3_size', min_value=10, max_value=600, step=80), activation='relu', name="Dense_3"))
    # We want to pass in 1 because this is the output layer, and we only want to predict one thing which is power
    # Otherwise if we predict 24 hrs ahead for each time step, then we have 96 values that the Dense layer should output
    #model.add(keras.layers.Dense(trainY.shape[1], name="Dense_4"))
    model.add(keras.layers.Dense(1, name="Dense_4"))

    optimizer = tf.keras.optimizers.RMSprop(hp.Float('learning_rate', min_value=0.00005, max_value=0.01, step=0.0001))

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])

    # model.fit(trainX, trainY, epochs=epochs,
    #           validation_split=validation_split, verbose=1)
    model.save('CNN_test.keras')
    return model

def fit_CNN_bayes_optimization(hyperparameters):
    # Extract hyperparameters
    learning_rate = hyperparameters['learning_rate']
    dropout_rate = hyperparameters['dropout_rate']
    filters = hyperparameters['filters']
    kernels = hyperparameters['kernels']
    pooling_size = hyperparameters['pooling_size']
    dense_units = hyperparameters['dense_units']

    # input must be of shape [batch_size, time_steps, input_dimension]
    n_timesteps = trainX.shape[1]  # 104
    n_features = trainX.shape[2]  # 5
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    # filters is the nr. of neurons
    # kernel size here is 1D because of the type of CNN, which is Conv1D
    # model.add(keras.layers.Conv1D(hp.Int('conv1_units', min_value=300, max_value=500, step=16),
    #                               kernel_size=21, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Conv1D(filters, kernels, activation='relu', name="Conv1D_1"))

    model.add(keras.layers.MaxPooling1D(pooling_size))

    model.add(keras.layers.Conv1D(filters, kernels, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.MaxPooling1D(pooling_size))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense_units, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(dense_units, activation='relu', name="Dense_2"))
    model.add(keras.layers.Dense(dense_units, activation='relu', name="Dense_3"))
    # We want to pass in 1 because this is the output layer, and we only want to predict one thing which is power
    # Otherwise if we predict 24 hrs ahead for each time step, then we have 96 values that the Dense layer should output
    model.add(keras.layers.Dense(trainY.shape[1], name="Dense_4"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])
    model.save('CNN_test.keras')
    return model


# Define the model-building class
class CNNModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters
        self.model = None

    def fit(self, X, y):
        # Build and compile the LSTM model using the provided hyperparameters
        model = KerasRegressor(build_fn=lambda: fit_CNN_bayes_optimization(
            self.hyperparameters),
            epochs=10,
            batch_size=64,
            verbose=0
        )

        # Train the model on the provided data
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mae']),
           label='Train')
  plt.plot(history.epoch, np.array(history.history['val_mae']),
           label='Val')
  plt.legend()
  plt.ylim([0, max(history.history['val_mae'])])
  plt.savefig('hist.png')


def main():
    datafile_path = 'station02.pkl'

    # load data
    cols_to_remove = ['nwp_winddirection', 'lmd_winddirection', 'lmd_pressure', 'station',
                      'lmd_hmd_directirrad', 'nwp_hmd_diffuseirrad', 'nwp_pressure']

    # From Y = G(X), where Y is the true data, and X the input. In this case we have X containing NWP data used to make
    # the prediction and Y is the true power data, taken from LMD
    # 24 hrs x 4 timesteps (i.e. sampled at each 15 min, so 4 times per hr)
    # 2 x 4, that is for the past two hrs, 4 samples per hr
    # Each sample takes values for 96 nwp from a particular timestep
    #trainX, trainY, testX, testY = load_CNN_data(datafile_path, cols_to_remove, 24*4, 2*4)
    #trainX, trainY, testX, testY = LS.load_LSTM_zero_padded('station02', 24 * 4, 24 * 4)

    # ________________________This uses Keras build in random search hyperparameter optimizer_______________________
    # Set up the tuner
    tuner = RandomSearch(
        fit_CNN_tune,  # Model-building function
        objective='val_accuracy',  # Metric to optimize
        max_trials=3,  # Number of different hyperparameter combinations to try
        directory='Tuner',  # Directory to save results
        project_name='CNN_hp')  # Project name

    # Search for the best hyperparameters
    #tuner.search(trainX, trainY[0], epochs=3, validation_split=0.2)
    tuner.search(trainX, trainY[0].reshape(trainY[0].shape[1], trainY[0].shape[0]), epochs=2, validation_split=0.2)

    # Get the best parameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train the final model
    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(trainX, trainY[0], batch_size=64, epochs=10, validation_split=0.2, verbose=1)

    # Plot the history of training and validation error
    # The further they are from each other, the more the model is over fitting the train data
    plot_history(history)

    # Evaluate the model
    eval_result = final_model.evaluate(testX, testY)

    # _________________________This uses Bayesian optimization for hyperparameter estimation________________________
    # Define the search space
    # param_space = {
    #     'learning_rate': (1e-6, 1e-2, 'log-uniform'),
    #     'dropout_rate': (0.0, 0.5),
    #     'filters': (16, 256),
    #     'kernels': (3, 21),
    #     'pooling_size': (1, 5),
    #     'dense_units': (16, 350)
    # }
    #
    # cnn_model_wrapper = CNNModelWrapper()
    # # Perform Bayesian Optimization
    # bayes_search = BayesSearchCV(
    #     estimator=cnn_model_wrapper,
    #     search_spaces=param_space,
    #     n_iter=10,  # Number of iterations (adjust as needed)
    #     cv=TimeSeriesSplit(n_splits=3),
    #     n_jobs=-1,  # Use all available CPUs
    #     verbose=1
    # )
    #
    # bayes_search.fit(trainX, trainY)
    #
    # # Get the best hyperparameters
    # best_params = bayes_search.best_params_
    # print("Best Hyperparameters:", best_params)
    #
    # # Train the final model
    # final_model = fit_CNN_bayes_optimization(**best_params)
    # final_model.fit(trainX, trainY, epochs=10, validation_split=0.2)
    #
    # # Evaluate the final model
    # eval_result = final_model.evaluate(testX, testY)

    print("Test accuracy:", eval_result[1])
    print("Test mae:", eval_result[0])

    power_predictions = final_model.predict(testX)
    r2 = r2_score(testY[0], power_predictions)
    mse = mean_squared_error(testY[0], power_predictions)
    #r2 = r2_score(testY.reshape(testY.shape[0], testY.shape[1]), power_predictions)
    #mse = mean_squared_error(testY.reshape(testY.shape[0], testY.shape[1]), power_predictions)

    print('R2 score: ', r2)
    print('MSE: ', mse)
    #cnn_model = fit_CNN(trainX, trainY, 'CNN_test.keras')
    #cnn_model = keras.models.load_model('./CNN_test.keras')

    # plt.figure()
    # plt.scatter(testY, power_predictions.flatten())
    # plt.ylabel('Predicted')
    # plt.xlabel('True values')
    # plt.ylim(-0.2, 1)
    # plt.xlim(-0.2, 1)
    # plt.plot([0, 1], [0, 1], color='r')
    # plt.show()

    # n = 0 is predicting 15 min ahead
    # n = 95 is predicting 24 hrs ahead
    n = 0
    plt.figure()
    plt.plot(testY[0], label='true value')
    plt.plot(power_predictions, label='predicted')
    # plt.plot(testY.reshape(testY.shape[0], testY.shape[1])[:, n], label='true value')
    # plt.plot(power_predictions[:, n], label='predicted')
    plt.legend()
    plt.savefig('Pred_vs_true.png')
    plt.show()

    print('test')

if __name__ == '__main__':
        main()