import re
import sys

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

from keras.models import Sequential
from keras import optimizers
from keras.layers import (Dense, Dropout, LSTM, Flatten)


def get_model(loop,
              input_shape        = (None, None),
              activation         = 'tanh',
              dropout            = 0.1,
              kernel_initializer = 'glorot_uniform',
              bias_initializer   = 'glorot_uniform',
              recurrent_dropout  = 0):

    model = Sequential()
    model.add(LSTM(16,
                   input_shape        = input_shape,
                   return_sequences   = True,
                   activation         = activation,
                   dropout            = dropout,
                   kernel_initializer = kernel_initializer,
                   bias_initializer   = bias_initializer,
                   recurrent_dropout  = recurrent_dropout))

    for _ in range(6):
        model.add(LSTM(8,
                       return_sequences   = True,
                       activation         = activation,
                       kernel_initializer = kernel_initializer,
                       dropout            = dropout,
                       bias_initializer   = bias_initializer,
                       recurrent_dropout  = recurrent_dropout))

    model.add(LSTM(6,
                   return_sequences   = False,
                   activation         = activation,
                   kernel_initializer = kernel_initializer,
                   dropout            = dropout,
                   bias_initializer   = bias_initializer,
                   recurrent_dropout  = recurrent_dropout))

    for _ in range(8):
        model.add(Dense(8, activation=activation))

    model.add(Dense(1))

    # model.add(LSTM(6, return_sequences=True, input_shape=input_shape))
    # model.add(LSTM(6, return_sequences=True))
    # model.add(LSTM(6, return_sequences=True))
    # model.add(LSTM(4, return_sequences=True))
    # model.add(LSTM(4))
    # model.add(Dense(1))

    return model


def get_optimizer(opt,
                  decay    = None,
                  lr       = None,
                  momentum = 0.0,
                  nesterov = False,
                  beta_1   = 0.9,
                  beta_2   = 0.999,
                  epsilon  = 1e-8,
                  rho      = None):

    """
    get_optimizer is a wrapper for Keras optimizers.

    Parameters
    ----------
    beta_1 : `float`
        adam optimizer parameter in range [0, 1) for updating bias first
        moment estimate
    beta_2 : `float`
        adam optimizer parameter in range [0, 1) for updating bias second
        moment estimate
    decay : `None` or `float`
        learning rate decay
    epsilon : `float`
        parameter for numerical stability
    opt : `str`
        Keras optimizer. Options: "sgd", "adam", "nadam", "rmsprop",
        "adagrad", "adamax" and "adadelta"
    lr : `None` or `float`
        optimizer learning rate
    momentum : `float`
        accelerate the gradient descent in the direction that dampens
        oscillations
    nesterov : `bool`
        use Nesterov Momentum
    rho : `None` or `float`
        gradient history

    Returns
    -------
    optimizer : :class:`keras.optimizer`
        keras optimizer object
    """

    ###############################
    # Stochastic Gradient Descent #
    ###############################
    if opt == 'sgd':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.SGD(lr       = lr,
                                   momentum = momentum,
                                   decay    = decay,
                                   nesterov = nesterov)
    ########
    # Adam #
    ########
    elif opt == 'adam':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    ##########
    # Adamax #
    ##########
    elif opt == 'adamax':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    #########
    # Nadam #
    #########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'nadam':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.004

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)

    ###########
    # RMSprop #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values (except the learning
    # rate, which can be freely tuned).
    elif opt == 'rmsprop':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.9

        optimizer = optimizers.RMSprop(lr      = lr,
                                       rho     = rho,
                                       epsilon = epsilon,
                                       decay   = decay)
    ###########
    # Adagrad #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adagrad':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adagrad(lr      = lr,
                                       decay   = decay,
                                       epsilon = epsilon)

    ############
    # Adadelta #
    ############
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adadelta':

        if lr is None:
            lr = 1.0

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.95

        optimizer = optimizers.Adadelta(lr      = lr,
                                        rho     = rho,
                                        epsilon = epsilon,
                                        decay   = decay)

    else:
        print('ERROR: Unknown optimizer')
        sys.exit(1)

    return optimizer
