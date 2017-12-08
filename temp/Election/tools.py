import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
import scipy.signal as sig
from scipy import stats
import seaborn as sns

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

from cycler import cycler
from keras.models import Sequential
from keras import optimizers
from keras.constraints import maxnorm
from keras.layers import (Dense, Dropout, LSTM, GRU,
                          Flatten, Conv1D, MaxPooling1D)


def get_model(model_type  = None,
              input_shape = None,
              dropout     = 0.0,
              batch_size  = None,
              kinit       = 'glorot_uniform',
              neurons     = 8,
              Rdropout    = 0.0,
              activation  = None,
              Nlayers     = 8):


    # Strip the model number away
    regex_model = re.compile(r'[a-zA-Z]+')
    model_type  = regex_model.findall(model_type)[0]

    model = Sequential()

    if model_type == 'LSTM':
        model.add(LSTM(32,
                       input_shape        = input_shape,
                       dropout            = dropout,
                       recurrent_dropout  = Rdropout,
                       kernel_initializer = kinit,
                       return_sequences   = True))

        model.add(Dense(32))

        for _ in range(3):
            model.add(LSTM(32,
                           dropout              = dropout,
                           recurrent_activation = 'sigmoid',
                           recurrent_dropout    = Rdropout,
                           kernel_initializer   = kinit,
                           return_sequences     = True))

            model.add(Dense(32))

        model.add(LSTM(32,
                       dropout              = dropout,
                       recurrent_activation = 'sigmoid',
                       recurrent_dropout    = Rdropout,
                       kernel_initializer   = kinit,
                       return_sequences     = False))

        model.add(Dense(32))

        model.add(Dense(1, activation='linear'))

    elif model_type == 'MLP':
        model.add(Dense(256,
                        input_shape = input_shape,
                        kernel_initializer = kinit,
                        activation  = activation))

        for _ in range(5):
            model.add(Dense(256))
            model.add(Dropout(dropout))

        layer_sizes = range(1, Nlayers)
        layer_sizes.reverse()
        for k in layer_sizes:
            model.add(Dense(2**k,
                            activation = activation,
                            kernel_initializer = kinit))

        model.add(Flatten())

        model.add(Dense(1, activation='linear'))

    elif model_type == 'GRU':
        model.add(GRU(16,
                      input_shape        = input_shape,
                      dropout            = dropout,
                      recurrent_dropout  = Rdropout,
                      kernel_initializer = kinit,
                      return_sequences   = True))

        model.add(GRU(16,
                      dropout            = dropout,
                      recurrent_dropout  = Rdropout,
                      kernel_initializer = kinit,
                      return_sequences   = False))

        model.add(Dense(1, activation='linear'))


    elif model_type == 'CNN':
        model.add(Conv1D(filters     = 16,
                         kernel_size = 1,
                         input_shape = input_shape,
                         strides     = 1,
                         padding     = 'valid',
                         activation  = 'relu'))

        for _ in range(15):
            model.add(Conv1D(64, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

            model.add(Conv1D(32, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

            model.add(Conv1D(16, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(8))
        model.add(Dense(1, activation='linear'))

    elif model_type == 'random':
        model.add(Dropout(dropout, input_shape=input_shape))

        for _ in range(Nlayers):
            model.add(Dense(neurons,
                            kernel_constraint  = maxnorm(3),
                            kernel_initializer = kinit,
                            activation         = activation))

            model.add(Dropout(dropout))

        model.add(Dense(1, kernel_constraint=maxnorm(3), activation='linear'))
        model.add(Flatten())

    else:
        raise Exception('Model not found!')

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
        sys.exit('ERROR: Unknown optimizer')

    return optimizer


def filter_channels(dataset, N=8, fknee=3, fs=256):
    nyquist = fknee / ( fs / 2.0)
    b, a    = sig.butter(N, nyquist, btype='low', output='ba')
    dataset = sig.lfilter(b, a, dataset)
    dataset = dataset.reshape(len(dataset))
    return dataset


def plot_results(target, prediction, fs,
                 plotDir   = '.',
                 title_str = None,
                 doFilter  = True):

    if doFilter:
        # Filter the output to make is easier to read
        ratio      = filter_channels(prediction / target)
        target     = filter_channels(target)
        prediction = filter_channels(prediction)
        fil_ratio  = prediction / target
        rmse       = np.sqrt(np.mean(np.square(target - prediction)))
        print('Filtered RMSE: ${:.2f}'.format(rmse))

        # Remove artifacts from filter
        start      = 80
        end        = 80
        ratio      = ratio[start:-end]
        target     = target[start:-end]
        prediction = prediction[start:-end]
        fil_ratio = fil_ratio[start:-end]

    else:
        ratio = prediction / target
        rmse  = np.sqrt(np.mean(np.square(target - prediction)))
        print('Filtered RMSE: ${:.2f}'.format(rmse))

    # Fill the array so we can loop over the data
    ff1          = list(range(len(target)))
    strain       = np.zeros(shape=(target.shape[0], 3))
    strain[:, 0] = target
    strain[:, 1] = prediction

    # Trendline
    z = np.polyfit(ff1, prediction, 1)
    p = np.poly1d(z)
    trend = p(ff1)

    strain[:, 2] = trend

    if title_str is None:
        title_str = 'Raised Funds Validation'

    # make plots to evaluate success / failure of the regression
    fig, (ax1, ax2) = plt.subplots(nrows       = 2,
                                   sharex      = True,
                                   figsize     = (6, 8),
                                   gridspec_kw = {'height_ratios': [3, 1]})

    ax1.set_prop_cycle(cycler('color', ['xkcd:baby blue', 'xkcd:black', 'xkcd:maroon']))

    # Plot the target and the prediction
    if doFilter:
        labels = ['Target (Filtered)', 'Prediction (Filtered)', 'Trendline']
    else:
        labels = ['Target', 'Prediction', 'Trendline']

    opacity   = [0.7, 0.9, 0.9]
    linestyle = ['-', '--', '-']
    linewidth = [2.5, 1.0, 1.0]
    for i in range(strain.shape[1]):
        ax1.plot(ff1, strain[:,i],
                 linestyle  = linestyle[i],
                 linewidth  = linewidth[i],
                 alpha      = opacity[i],
                 rasterized = True,
                 label      = labels[i])

    ax1.legend(fontsize='small', loc=1)
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'Funds Raised (USD)')
    ax1.set_title(title_str)

    # Plot the ratio of the prediction to the targets.
    if doFilter:
        ax2.plot(ff1, fil_ratio,
                 label      = 'Prediction/Target (Filtered)',
                 c          = 'xkcd:dark seafoam',
                 linewidth  = 1.0,
                 rasterized = True)

        # Show where prediction is perfect
        ax2.plot(ff1, [1 for _ in range(len(ff1))],
                 label      = 'Perfect Fit',
                 c          = 'xkcd:black',
                 linestyle  = ':',
                 rasterized = True)

        ax2.set_ylim([0, 2])

    else:
        ax2.semilogy(ff1, np.abs(ratio),
                 label      = 'Prediction/Target (Raw)',
                 c          = 'xkcd:dark seafoam',
                 linewidth  = 1.0,
                 rasterized = True)

        # Show where prediction is perfect
        ax2.semilogy(ff1, [1 for _ in range(len(ff1))],
                 label      = 'Perfect Fit',
                 c          = 'xkcd:black',
                 linestyle  = ':',
                 rasterized = True)

        ax2.set_ylim([1e-2, 1e2])

    ax2.grid(True, which='minor')
    ax2.set_ylabel(r'Ratio')
    ax2.legend(fontsize='x-small', loc=1)

    plt.subplots_adjust(hspace=0.075)

    # save figure
    try:
        get_ipython
        plt.show()
    except NameError:
        figName = '{0}/Funds_Validation'.format(plotDir)
        plt.savefig('{}.png'.format(figName))


def set_plot_style():
    # Now alter my matplotlib parameters
    plt.style.use('bmh')
    mpl.rcParams.update({
        'axes.grid': True,
        'axes.titlesize': 'medium',
        'font.family': 'serif',
        'font.size': 12,
        'grid.color': 'w',
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.linewidth': 1,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 13,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.1,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{txfonts}'})

    mpl.rc("savefig", dpi=100)
    mpl.rc("figure", figsize=(7, 4))


def scale_features(array):
    mn = np.float64(np.min(array))
    mx = np.float64(np.max(array))
    array = (array - mn) * 1.0 / (mx - mn)
    return array, mn, mx


def rescale_features(array, mn, mx):
    rescaled = array * (mx - mn) + mn
    return rescaled


def load_dataset(datafile):
    if datafile is None:
        try:
            datafile = 'cleaned_input.csv'
            dataset  = np.genfromtxt(datafile, delimiter=',')
        except Exception as e:
            sys.exit(e)
    else:
        try:
            dataset = np.genfromtxt(datafile, delimiter=',')
        except:
            sys.exit('ERROR: Could not load {}'.format(datafile))

    print('Datafile: {}'.format(datafile))
    dataset = dataset.astype('float32')
    return dataset


def split_dict(d, elements):
    """
    split_dict is used in plot_density in order to take a dict `d` of
    length `n` and split it into a list of dicts of length `elements`

    Parameters
    ----------
    d : `dict`
        input dictionary

    elements : `int`
        number of elements in each sub-dict

    Returns
    -------
    output : `list`
        list containing dicts of length `elements`
    """
    output = []
    chunks = int(len(d) / elements) + int(bool(len(d) % elements))
    for i in range(chunks):
        temp = {}
        if i < chunks - 1:
            for j in range(elements):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]
        else:
            remainder = len(d) - elements * i
            for j in range(remainder):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]

        output.append(temp)

    return output


def plot_channel_correlations(dataset, plotNum=4, seconds=100):

    # Scale it for nicer plotting
    dataset, _, _ = scale_features(dataset)

    # Cut it down to a reasonable size (1 minute)
    duration = int(seconds)
    temp = np.zeros(shape=(duration, dataset.shape[1]))

    for i in range(dataset.shape[1]):
        temp[:, i] = dataset[:duration, i]

    darm = temp[:, -1]
    witness = temp[:, :-1]
    chans = ['Election_Jurisdiction','DistrictID','Incumbent','Open','VC01',
             'VC02','VC03','VC04','VC05','VC06','VC07','VC08','VC11','VC12',
             'VC13','VC14','VC15','VC16','VC17','VC18','VC19','VC20','VC21',
             'VC22','VC23','VC24','VC25','VC26','VC27','VC28','VC29','VC30',
             'VC31','VC32','VC33','VC34','VC35','VC36','VC37','VC38','VC39',
             'VC40','VC41','VC42','VC43','VC44','VC45','VC46','VC47','VC48',
             'VC49','VC50','VC51','VC52','VC53','VC54','VC55','VC56','VC57',
             'VC58','VC59','VC60','VC61','VC62','VC63','VC64','Total_$']

    data = dict(zip(chans, witness.T))
    data = split_dict(data, plotNum - 1)  # Not including DARM

    for index, chunk in enumerate(data):
        sys.stdout.write('\rMaking plot {0}/{1}'.format(index + 1, len(data)))
        sys.stdout.flush()
        if index == len(data) - 1:
            print('')

        # Add DARM and convert to DataFrame object
        chunk["Total_$"] = darm
        df = pd.DataFrame(chunk)

        # Make the plots
        G = sns.PairGrid(df)

        G.map_diag(sns.distplot,
                   fit   = stats.gamma,
                   kde   = False,
                   rug   = False,
                   color = 'k')

        cmap = sns.cubehelix_palette(as_cmap = True,
                                     dark    = 0,
                                     light   = 1,
                                     reverse = True)
        G.map_offdiag(sns.kdeplot,
                      cmap         = cmap,
                      shade        = True,
                      shade_lowest = False)

        plt.suptitle('Data Channel Correlations')
        plt.savefig('Corr/channel_corr_{0}.png'.format(index))
        plt.close()


def clean_data(datafile):
    states = ['AK','AL','AR','AS','AZ','CA','CO','CT','DC','DE','FL','GA','GU',
              'HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN',
              'MO','MP','MS','MT','NA','NC','ND','NE','NH','NJ','NM','NV','NY',
              'OH','OK','OR','PA','PR','RI','SC','SD','TN','TX','UT','VA','VI',
              'VT','WA','WI','WV','WY']

    state_dict = {}
    for i in range(len(states)):
        state_dict[states[i]] = i

    nd = []
    with open(datafile, 'r') as f:
        lines = f.readlines()
        passed = 0
        for line in lines:
            line = line.strip('\n').split(',')
            if '' in line:
                passed += 1
                pass
            elif float(line[-1]) == 0:
                passed += 1
                pass
            else:
                line = line[1:]
                line[0] = state_dict[line[0]]
                line = list(map(float, line))
                nd.append(line)

    nd = np.array(nd)
    np.random.shuffle(nd)

    temp = []
    for i in range(nd.shape[0]):
        if nd[i, -1] < 1e6:
            temp.append(nd[i, :])

    nd = np.array(temp)

    output = 'shuffle_complete.csv'
    np.savetxt(output, nd, delimiter=',')
    print('saved to {}'.format(output))
