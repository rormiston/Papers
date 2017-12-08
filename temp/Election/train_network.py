#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re
import scipy.io as sio
import sys

import tools
# get plot formatting
tools.set_plot_style()

import os
# Hush AVX and processor warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'


def run_network(
    # Data
    datafile   = None,
    doFilter   = True,
    plotDir    = '.',
    train_frac = 0.75,

    # Neural network params
    activation = 'elu',
    dropout    = 0.0,
    loss       = 'mse',
    model_type = 'LSTM',
    optimizer  = 'adam',
    recurrent_dropout = 0.00,

    # Optimizer
    beta_1   = 0.9,
    beta_2   = 0.999,
    decay    = None,
    epsilon  = 1e-8,
    lr       = None,
    momentum = 0.0,
    nesterov = False,
    rho      = None,

    # Training
    batch_size = 100,
    epochs     = 10,
    Nlayers    = 8,
    shuffle    = False,
    verbose    = 1):

    ############################
    # Load and preprocess data #
    ############################
    # Make sure the data loads when running from any directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    # Get the model basename and version number
    regex_model = re.compile(r'[a-zA-Z]+')
    basename    = regex_model.findall(model_type)[0]
    regex_num   = re.compile(r'\d+')
    try:
        version = regex_num.findall(model_type)[0]
    except IndexError:
        version = '0'

    print("Network: {}".format(basename))

    # Load the dataset and remove the starting outliers
    dataset = tools.load_dataset(datafile)

    # Scale dataset to range (0, 1)
    scaled, mn, mx = tools.scale_features(dataset)

    # Split into train and test sets
    train_len = int(train_frac * scaled.shape[0])
    train     = scaled[:train_len, :]
    test      = scaled[train_len:, :]

    # Get training and testing feature matrices
    train_X = train[:, :-1]
    train_y = train[:, -1]
    test_X  = test[:, :-1]
    test_y  = test[:, -1]

    # Reshape input to [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X  = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    ######################
    # Get network design #
    ######################
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = tools.get_model(model_type  = model_type,
                            input_shape = input_shape,
                            dropout     = dropout,
                            Rdropout    = recurrent_dropout,
                            activation  = activation,
                            Nlayers     = Nlayers)

    optimizer = tools.get_optimizer(opt      = optimizer,
                                    decay    = decay,
                                    lr       = lr,
                                    momentum = momentum,
                                    nesterov = nesterov,
                                    beta_1   = beta_1,
                                    beta_2   = beta_2,
                                    epsilon  = epsilon,
                                    rho      = rho)

    model.compile(loss=loss, optimizer=optimizer)

    ###############
    # Fit network #
    ###############
    history = model.fit(train_X, train_y,
                        epochs          = epochs,
                        batch_size      = batch_size,
                        validation_data = (test_X, test_y),
                        verbose         = verbose,
                        shuffle         = shuffle)

    ##############################
    # Make Predictions and Plots #
    ##############################
    # make a prediction
    yhat = model.predict(test_X)

    # Rescale back to original units and reshape
    yhat = tools.rescale_features(yhat, mn, mx)
    yhat = yhat.reshape(len(yhat))

    # Get the rescaled target values
    target = tools.rescale_features(scaled[-len(yhat):, -1], mn, mx)
    target = target.reshape(len(target))

    # Make sure the output directories exist
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    # Plot the results. sample_rate is for lowpass filter
    sample_rate = 0.01
    tools.plot_results(target, yhat, sample_rate,
                       plotDir  = plotDir,
                       doFilter = doFilter)

    # Save the data
    output_data = {'history'   : history.history,
                   'fsample'   : sample_rate,
                   'subtracted': target - yhat}

    mat_str  = 'params/{0}/Regression_Results-{1}.mat'
    mat_name = mat_str.format(basename, version)

    if not os.path.isfile(mat_name):
        os.system('touch {0}'.format(mat_name))

    sio.savemat(mat_name, output_data, do_compression=True)


if __name__ == '__main__':


    def parse_command_line():
        """
        parse command line flags. use sensible defaults
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_type", "-m",
                            help    = "pick model type to use",
                            default = "MLP",
                            dest    = "model_type",
                            type    = str)

        parser.add_argument("--train_frac", "-tf",
                            help    = "ratio of dataset used for training",
                            default = 0.75,
                            dest    = "train_frac",
                            type    = float)

        parser.add_argument("--datafile", "-df",
                            help    = "data file to read from",
                            default = None,
                            dest    = "datafile",
                            type    = str)

        parser.add_argument("--dropout", "-D",
                            help    = "dropout regularization",
                            default = 0.0,
                            dest    = "dropout",
                            type    = float)

        parser.add_argument("--doFilter",
                            help    = "filter output data",
                            default = True,
                            dest    = "doFilter",
                            action  = 'store_false')

        parser.add_argument("--recurrent_dropout", "-RD",
                            help    = "recurrent dropout used in RNN memory blocks",
                            default = 0.0,
                            dest    = "recurrent_dropout",
                            type    = float)

        parser.add_argument("--loss",
                            help    = "loss function for neural network",
                            default = 'mse',
                            dest    = "loss",
                            type    = str)

        parser.add_argument("--activation", "-a",
                            help    = "activation function for neural network",
                            default = "tanh",
                            dest    = "activation",
                            type    = str)

        parser.add_argument("--optimizer", "-opt",
                            help    = "optimizing function for neural network",
                            default = 'sgd',
                            dest    = "optimizer",
                            type    = str)

        parser.add_argument("--epochs", "-e",
                            help    = "Number of iterations of NN training",
                            default = 250,
                            dest    = "epochs",
                            type    = int)

        parser.add_argument("--Nlayers", "-l",
                            help    = "Number of layers for the Dense network",
                            default = 8,
                            dest    = "Nlayers",
                            type    = int)

        parser.add_argument("--batch_size", "-b",
                            help    = "number of samples to be trained at once",
                            default = 250,
                            dest    = "batch_size",
                            type    = int)

        parser.add_argument("--shuffle", "-s",
                            help    = "shuffle training data",
                            default = False,
                            dest    = "shuffle",
                            action  = 'store_true')

        parser.add_argument("--verbose", "-v",
                            help    = "output verbosity",
                            default = 0,
                            dest    = "verbose",
                            type    = int)

        parser.add_argument("--plotDir",
                            help    = "directory to store plots",
                            default = 'Plots/',
                            dest    = "plotDir",
                            type    = str)

        parser.add_argument("--learning_rate", "-lr",
                            help    = "optimizer learning rate",
                            default = None,
                            dest    = "lr",
                            type    = float)

        parser.add_argument("--decay",
                            help    = "optimizer learning rate decay",
                            default = None,
                            dest    = "decay",
                            type    = float)

        parser.add_argument("--momentum",
                            help    = "optimizer momentum",
                            default = 0.0,
                            dest    = "momentum",
                            type    = float)

        parser.add_argument("--nesterov",
                            help    = "use nesterov momentum",
                            default = False,
                            dest    = "nesterov",
                            action  = 'store_true')

        parser.add_argument("--beta_1",
                            help    = "beta_1 params for optimizer",
                            default = 0.9,
                            dest    = "beta_1",
                            type    = float)

        parser.add_argument("--beta_2",
                            help    = "beta_2 params for optimizer",
                            default = 0.999,
                            dest    = "beta_2",
                            type    = float)

        parser.add_argument("--epsilon",
                            help    = "optimizer param",
                            default = 1e-8,
                            dest    = "epsilon",
                            type    = float)

        parser.add_argument("--rho",
                            help    = "adadelta & rmsprop optimizer params",
                            default = None,
                            dest    = "rho",
                            type    = float)

        params = parser.parse_args()

        # Convert params to a dict to feed into run_network as **kwargs
        model_params = {}
        for arg in vars(params):
            model_params[arg] = getattr(params, arg)

        return model_params


    # Get command line flags
    model_params = parse_command_line()

    # Set plotDir to use the current model
    model_type  = model_params['model_type']
    regex_model = re.compile(r'[a-zA-Z]+')
    basename    = regex_model.findall(model_type)[0]
    model_params['plotDir'] = 'params/{}/Figures/'.format(basename)

    # Run it!
    run_network(**model_params)
