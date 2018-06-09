#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import preprocessing as ppr
import analysis as ana
import models as mod
ana.set_plot_style()

from collections import OrderedDict
from ConfigParser import ConfigParser
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def run_network(
    # Data processing
    datafile   = 'Data/H1_data_array.mat',
    data_type  = 'real',
    loop       = 'Loop_0',
    hc_offset  = 5,
    lowcut     = 3.0,
    highcut    = 60.0,
    N_bp       = 8,
    preFilter  = True,
    postFilter = True,
    subsystems = 'all',
    tfrac      = 0.5,
    ts         = 1,

    # Plots
    fmin    = 4,
    fmax    = 256,
    plotDir = 'Plots',

    # Network
    beta_1     = 0.9,
    beta_2     = 0.999,
    clean_darm = [],
    decay      = None,
    epochs     = 100,
    epsilon    = 1e-8,
    loss       = 'mae',
    lr         = None,
    momentum   = 0.0,
    nesterov   = False,
    optimizer  = 'adam',
    rho        = None,
    verbose    = 1):

    # load dataset and scale
    dataset, fs = ppr.get_dataset(datafile,
                                  data_type  = data_type,
                                  subsystems = subsystems)
    nd = np.copy(dataset)
    if len(clean_darm) > 0:
        dataset[-len(clean_darm):, 0] = clean_darm

    if data_type == 'real':
        values = np.delete(dataset, 1, axis=1)  # get rid of bkgd

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # bandpass filter
    if preFilter:
        scaled = ppr.phase_filter(scaled,
                                  fs      = fs,
                                  order   = N_bp,
                                  lowcut  = lowcut,
                                  highcut = highcut)

    # get lookback
    lookback = ppr.lstm_lookback(scaled, ts, 1)
    darm = lookback.values[:, scaled.shape[1] * ts]
    to_drop = [scaled.shape[1] * i for i in range(ts + 1)]
    lookback.drop(lookback.columns[to_drop], axis=1, inplace=True)

    # split into train and test sets
    values = lookback.values
    tfrac  = int(tfrac * dataset.shape[0])
    train  = values[:tfrac, :]
    test   = values[tfrac:, :]

    # split into input and outputs
    train_X, train_y = train, darm[:tfrac]
    test_X,  test_y  = test,  darm[tfrac:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = mod.get_model(loop, input_shape)

    optimizer = mod.get_optimizer(optimizer,
                                  decay    = decay,
                                  lr       = lr,
                                  momentum = momentum,
                                  nesterov = nesterov,
                                  beta_1   = beta_1,
                                  beta_2   = beta_2,
                                  epsilon  = epsilon,
                                  rho      = rho)

    model.compile(loss=loss, optimizer=optimizer)

    # fit network
    batch_size = min(int(dataset.shape[0] / 100), 4096)
    history = model.fit(train_X, train_y,
                        epochs          = epochs,
                        batch_size      = batch_size,
                        validation_data = (test_X, test_y),
                        verbose         = verbose,
                        shuffle         = False)

    # make a prediction
    print('[+] Generating network prediction')
    yhat = model.predict(test_X)

    # rescale prediction, get target
    inv_yhat = np.concatenate((yhat, scaled[:len(yhat), :-1]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    inv_y    = nd[-len(inv_yhat):, 0]

    if postFilter:
        inv_yhat = ppr.phase_filter(inv_yhat,
                                    fs      = fs,
                                    order   = N_bp,
                                    lowcut  = lowcut,
                                    highcut = highcut - hc_offset)

    # calculate RMSE
    rmse = np.sqrt(np.mean(np.square(inv_y - inv_yhat)))
    print('[+] {0} RMSE: {1}'.format(loop.title(), rmse))

    if not os.path.isdir(plotDir):
        os.system('mkdir -p {}'.format(plotDir))

    # plot loss history
    plt.plot(history.history['loss'][1:], label='train')
    plt.plot(history.history['val_loss'][1:], label='test')
    plt.legend()
    plt.savefig('{0}/{1}_loss.png'.format(plotDir, loop))
    plt.close()

    # plot PSD
    ana.plot_psd(inv_y, inv_yhat,
                 fs      = fs,
                 plotDir = plotDir,
                 saveas  = '{}_frequency_validation'.format(loop),
                 title   = '{} Frequency Validation'.format(loop.title()),
                 fmin    = fmin,
                 fmax    = fmax)

    return inv_yhat, inv_y, fs


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)


    def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ini_file", "-i",
                            help    = "config file to read",
                            default = '../configs/configs.ini',
                            dest    = "ini_file",
                            type    = str)

        params = parser.parse_args()
        return params


    params    = parse_command_line()
    ini_file  = params.ini_file
    data_dict = ppr.get_run_params(ini_file, 'Data')
    to_run    = ppr.get_run_params(ini_file, 'To_Run')

    param_dict  = OrderedDict()
    predictions = OrderedDict()
    runs = []
    for ix in range(len(to_run)):
        loop = 'Loop_{}'.format(ix)

        if to_run[loop]:
            param_dict[loop]  = ppr.get_run_params(ini_file, loop)
            param_dict[loop]['loop'] = loop

            if len(runs) > 0:
                subtract_loop = 'Loop_{}'.format(len(runs) - 1)
                param_dict[loop]['clean_darm'] = predictions[subtract_loop]

            text = '[+] Running {}'.format(loop)
            print('\n{0}\n{1}'.format(text, '-' * len(text)))
            predictions[loop], darm, fs = run_network(**param_dict[loop])
            runs.append(loop)


    if len(runs) > 0:
        ana.plot_progress(darm, predictions.values(),
                          loops   = runs,
                          plotDir = 'Plots',
                          fs      = fs,
                          fmin    = 4,
                          fmax    = 256)

    print('[+] Done')
