from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from sklearn.preprocessing import MinMaxScaler
from ConfigParser import ConfigParser
import deepclean.preprocessing as ppr
import argparse
import os
import numpy as np
import time


def data():

    #####################################
    # YOU MAY NEED TO SET THIS MANUALLY #
    #####################################
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    ini_file = '{}/deepclean_ve/configs/tuning.ini'.format(os.environ['HOME'])

    # Read config file
    settings = ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)

    # Collect config params
    section  = 'Tuning'
    run_params = {}
    run_params['datafile']  = settings.get(section, 'datafile')
    run_params['data_type'] = settings.get(section, 'data_type')
    run_params['fs']        = settings.getint(section, 'fs')
    run_params['highcut']   = settings.getfloat(section, 'highcut')
    run_params['lowcut']    = settings.getfloat(section, 'lowcut')
    run_params['N_bp']      = settings.getint(section, 'N_bp')
    run_params['preFilter'] = settings.getboolean(section, 'preFilter')
    run_params['save_mat']  = settings.getboolean(section, 'save_mat')

    subsystems = settings.get(section, 'subsystems')
    if ',' in subsystems:
        subsystems = [s.strip() for s in subsystems.split(',')]
    else:
        subsystems = [subsystems]

    run_params['subsystems'] = subsystems
    run_params['tfrac']      = settings.getfloat(section, 'tfrac')
    run_params['ts']         = settings.getint(section, 'ts')

    # load dataset and scale
    if run_params['save_mat']:
        if not os.path.isfile(run_params['datafile']):
            ppr.stream_data(ini_file)
        dataset, fs = ppr.get_dataset(run_params['datafile'],
                                      data_type  = run_params['data_type'],
                                      subsystems = run_params['subsystems'])
    else:
        dataset, fs = ppr.stream_data(ini_file)

    if run_params['data_type'] == 'real':
        values = np.delete(dataset, 1, axis=1)  # get rid of bkgd

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # bandpass filter
    if run_params['preFilter']:
        scaled = ppr.phase_filter(scaled,
                                  fs      = run_params['fs'],
                                  order   = run_params['N_bp'],
                                  lowcut  = run_params['lowcut'],
                                  highcut = run_params['highcut'])

    # get lookback
    lookback = ppr.lstm_lookback(scaled, run_params['ts'], 1)
    darm = lookback.values[:, scaled.shape[1] * run_params['ts']]
    to_drop = [scaled.shape[1] * i for i in range(run_params['ts'] + 1)]
    lookback.drop(lookback.columns[to_drop], axis=1, inplace=True)

    # split into train and test sets
    values = lookback.values
    tfrac  = int(run_params['tfrac'] * dataset.shape[0])
    train  = values[:tfrac, :]
    test   = values[tfrac:, :]

    # split into input and outputs
    x_train, y_train = train, darm[:tfrac]
    x_test,  y_test  = test,  darm[tfrac:]

    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    ###############################################
    # YOU NEED TO SET EVERYTHING IN HERE MANUALLY #
    ###############################################
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = Sequential()
    model.add(LSTM({{choice([3, 6, 12])}}, return_sequences=True, input_shape=input_shape))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Activation({{choice(['relu', 'tanh', 'linear'])}}))
    model.add(LSTM(6, return_sequences=True))
    model.add(Activation({{choice(['relu', 'tanh', 'linear'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM(4))
    model.add(Activation({{choice(['relu', 'tanh', 'linear'])}}))
    model.add(Dense(1, activation='linear'))

    model.compile(loss={{choice(['mse', 'mae'])}}, metrics=['mse'],
                  optimizer={{choice(['rmsprop', 'adam'])}})

    model.fit(x_train, y_train,
              batch_size=4096,
              epochs=3,
              verbose=2,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("\nEvalutation of best performing model:")
    output = best_model.evaluate(X_test, Y_test)
    print(output)
    print("\nBest performing model chosen hyper-parameters:")
    print(best_run)
    print('\nResults written to hyper_opt_results.txt')

    if not os.path.isfile('hyper_opt_results.txt'):
        os.system('touch hyper_opt_results.txt')

    with open('hyper_opt_results.txt', 'a+') as f:
        today = time.strftime('%c')
        f.write('{0}\n{1}\n'.format(today, '-' * len(today)))
        f.write('  Test Accuracy: {}\n'.format(output[1]))
        f.write('  Loss: {}\n'.format(output[0]))
        for k, v in best_run.items():
            f.write('  {0}: {1}\n'.format(k, v))
        f.write('\n')
