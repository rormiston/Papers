import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(3301)
import sys
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import deepclean.preprocessing as ppr
import deepclean.analysis as ana
import deepclean.models as mod
ana.set_plot_style()

from ConfigParser import ConfigParser
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def run_network(
    # Data processing
    datafile   = 'Data/H1_data_array.mat',
    data_type  = 'real',
    loop       = 'Loop_0',
    hc_offset  = 5,
    ini_file   = '../configs/configs.ini',
    lowcut     = 3.0,
    highcut    = 60.0,
    N_bp       = 8,
    preFilter  = True,
    postFilter = True,
    save_mat   = False,
    subsystems = 'all',
    tfrac      = 0.5,

    # Plots
    fmin    = 4,
    fmax    = 256,
    plotDir = 'Plots',

    # Network
    activation = 'tanh',
    beta_1     = 0.9,
    beta_2     = 0.999,
    bias_initializer = 'glorot_uniform',
    clean_darm = [],
    decay      = None,
    dropout    = 0.1,
    epochs     = 100,
    epsilon    = 1e-8,
    kernel_initializer = 'glorot_uniform',
    lookback   = 15,
    loss       = 'mae',
    lr         = None,
    momentum   = 0.0,
    nesterov   = False,
    optimizer  = 'adam',
    recurrent_dropout = 0.0,
    rho        = None,
    verbose    = 1):


    # load dataset and scale
    if save_mat:
        if not os.path.isfile(datafile):
            ppr.stream_data(ini_file)
        dataset, fs = ppr.get_dataset(datafile,
                                      data_type  = data_type,
                                      subsystems = subsystems)
    else:
        dataset, fs = ppr.stream_data(ini_file)

    nd = np.copy(dataset)
    if len(clean_darm) > 0:
        dataset[-len(clean_darm):, 0] = clean_darm

    if data_type == 'real':
        values = np.delete(dataset, 1, axis=1)  # get rid of bkgd

    check_loops = ['Loop_1', 'Loop_2']
    if any(loop==check_loops[i] for i in range(len(check_loops))):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # bandpass filter
        if preFilter:
            scaled = ppr.phase_filter(scaled,
                                      fs      = fs,
                                      order   = N_bp,
                                      lowcut  = lowcut,
                                      highcut = highcut)

        lookback = ppr.lstm_lookback(scaled, 2, 1)
        darm = lookback.values[:, scaled.shape[1] * 2]
        to_drop = [scaled.shape[1] * i for i in range(2 + 1)]
        lookback.drop(lookback.columns[to_drop], axis=1, inplace=True)

        # split into train and test sets
        values = lookback.values
        tfrac  = int(tfrac * dataset.shape[0])
        train  = values[:tfrac, :]
        test   = values[tfrac:, :]

        # split into input and outputs
        x_train, y_train = train, darm[:tfrac]
        x_test,  y_test  = test,  darm[tfrac:]

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        # design network
        input_shape = (x_train.shape[1], x_train.shape[2])
        batch_size = min(int(dataset.shape[0] / 100), 4096)
        model = Sequential()
        model.add(LSTM(6, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(6, return_sequences=True))
        model.add(LSTM(6, return_sequences=True))
        model.add(LSTM(4, return_sequences=True))
        model.add(LSTM(4))
        model.add(Dense(1))

    else:
        # bandpass filter
        if preFilter:
            values = ppr.phase_filter(values,
                                      fs      = fs,
                                      order   = N_bp,
                                      lowcut  = lowcut,
                                      highcut = highcut)

        # normalize and standardize
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        minmax         = min_max_scaler.fit_transform(values)
        std_scaler     = StandardScaler()
        scaled         = std_scaler.fit_transform(minmax)

        # split into training and testing data
        tfrac  = int(tfrac * scaled.shape[0])
        x_train, y_train = scaled[:tfrac, 1:], scaled[:tfrac, 0]
        x_test,  y_test  = scaled[tfrac:, 1:], scaled[tfrac:, 0]

        # apply lookback
        x_train = ppr.do_lookback(x_train, steps=lookback)
        x_test  = ppr.do_lookback(x_test,  steps=lookback)

        # account for first samples (lookback - 1) with no lookback
        y_train = y_train[-x_train.shape[0]:]
        y_test  = y_test[-x_test.shape[0]:]

        # build network architecture and compile
        input_shape = (x_train.shape[1], x_train.shape[2])
        batch_size = min(int(dataset.shape[0] / 100), 400)

        model = mod.get_model(loop,
                              input_shape        = input_shape,
                              activation         = activation,
                              dropout            = dropout,
                              kernel_initializer = kernel_initializer,
                              bias_initializer   = bias_initializer,
                              recurrent_dropout  = recurrent_dropout)

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
    history = model.fit(x_train, y_train,
                        epochs          = epochs,
                        batch_size      = batch_size,
                        validation_data = (x_test, y_test),
                        verbose         = verbose)

    # make a prediction
    sys.stdout.write('\r\n[+] Generating network prediction... ')
    sys.stdout.flush()
    yhat = model.predict(x_test)

    if any(loop==check_loops[i] for i in range(len(check_loops))):
        # rescale min max
        yhat = min_max_scaler.inverse_transform(yhat)
    else:
        # rescale standard deviation, then min_max
        yhat = np.concatenate((yhat, scaled[:len(yhat), :-1]), axis=1)
        yhat = std_scaler.inverse_transform(yhat)
        yhat = min_max_scaler.inverse_transform(yhat)

    # get target value (original data instead of rescaling)
    inv_yhat = yhat[:, 0]
    inv_yhat = inv_yhat.reshape((len(inv_yhat)))
    inv_y    = nd[-len(inv_yhat):, 0]
    inv_y    = inv_y.reshape((len(inv_y)))

    # bandpass filter to remove artificats from prefilter
    if postFilter:
        inv_yhat = ppr.phase_filter(inv_yhat,
                                    fs      = fs,
                                    order   = N_bp,
                                    lowcut  = lowcut  + hc_offset,
                                    highcut = highcut - hc_offset)

    sys.stdout.write('Done\n')

    # calculate RMSE
    rmse = np.sqrt(np.mean(np.square(inv_y - inv_yhat)))
    print('[+] {0} RMSE: {1}'.format(loop.title(), rmse))

    if not os.path.isdir(plotDir):
        os.system('mkdir -p {}'.format(plotDir))

    # plot loss history
    plt.close()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss {}'.format(loop))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{0}/{1}_loss.png'.format(plotDir, loop))
    plt.close()

    # plot 2 seconds of the timeseries prediction vs target
    secs = 2
    plt.plot(inv_y[-int(fs * secs):], label='filtered target')
    plt.plot(inv_yhat[-int(fs * secs):], label='prediction')
    plt.legend()
    plt.title('LSTM Timeseries: Prediction vs Target {0}-{1} Hz'.format(lowcut, highcut))
    plt.xlabel('Timesteps')
    plt.ylabel('Amplitude')
    plt.savefig('{0}/{1}_timeseries.png'.format(plotDir, loop))
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
    dflt_ini = '{}/deepclean_ve/configs/configs.ini'.format(os.environ['HOME'])

    def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ini_file", "-i",
                            help    = "config file to read",
                            default = dflt_ini,
                            dest    = "ini_file",
                            type    = str)

        params = parser.parse_args()
        return params


    params    = parse_command_line()
    ini_file  = params.ini_file
    data_dict = ppr.get_run_params(ini_file, 'Data')
    to_run    = ppr.get_run_params(ini_file, 'To_Run')

    param_dict  = {}
    predictions = {}
    runs = []
    for ix in range(len(to_run)):
        loop = 'Loop_{}'.format(ix)

        if to_run[loop]:
            param_dict[loop]  = ppr.get_run_params(ini_file, loop)
            param_dict[loop]['loop'] = loop
            param_dict[loop]['ini_file'] = ini_file
            param_dict[loop].update(data_dict)

            if len(runs) > 0:
                subtract_loop = 'Loop_{}'.format(len(runs) - 1)
                param_dict[loop]['clean_darm'] = predictions[subtract_loop]

            text = '[+] Running {}'.format(loop)
            print('\n{0}\n{1}'.format(text, '-' * len(text)))
            predictions[loop], darm, fs = run_network(**param_dict[loop])
            runs.append(loop)


    if len(runs) > 1:
        ana.plot_progress(darm, predictions.values(),
                          loops   = runs,
                          plotDir = param_dict[param_dict.keys()[0]]['plotDir'],
                          fs      = fs,
                          fmin    = 4,
                          fmax    = 256)

    print('[+] Done')
