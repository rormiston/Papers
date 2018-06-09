from __future__ import division
import pandas as pd
import scipy.signal as sig
import scipy.io as sio
import numpy as np

from collections import OrderedDict
from ConfigParser import ConfigParser


def lstm_lookback(data, n_in=1, n_out=1):
    """
    create lookback in the dataset

    Parameters
    ----------
    data : `numpy.ndarray`
        dataset for training and testing
    n_in : `int`
        number of timesteps to lookback
    n_out : `int`
        number of timesteps to forecast

    Returns
    -------
    combined : `numpy.ndarray`
        dataset with lookback
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # predict future timestep
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    combined = pd.concat(cols, axis=1)
    combined.columns = names
    combined.dropna(inplace=True)

    return combined


def phase_filter(dataset,
                 lowcut  = 4.0,
                 highcut = 20.0,
                 order   = 8,
                 btype   = 'bandpass',
                 fs      = 512):
    """
    phase preserving bandpass filter

    Parameters
    ----------
    btype : `str`
        filter type
    dataset : `numpy.ndarray`
        dataset for training and testing
    fs : `int`
        data sample rate
    highcut : `float`
        stop frequency for filter
    lowcut : `float`
        start frequency for filter
    order : `int`
        bandpass filter order

    Returns
    -------
    dataset : `numpy.ndarray`
        bandpassed dataset for training and testing
    """

    # Normalize the frequencies
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq

    # Make and apply filter
    z, p, k = sig.butter(order, [low, high], btype=btype, output='zpk')
    sos = sig.zpk2sos(z, p, k)

    if dataset.ndim == 2:
        for i in range(dataset.shape[1]):
            dataset[:, i] = sig.sosfiltfilt(sos, dataset[:, i])
    else:
        dataset = sig.sosfiltfilt(sos, dataset)

    return dataset


def get_run_params(ini_file, section):
    """
    function for reading parameters from the config file

    Parameters
    ----------
    ini_file : `str`
        path to config file
    section : `str`
        config file section to read from

    Returns
    -------
    run_params : `dict`
        dict of params from supplied config file and section
    """
    settings = ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)
    run_params = OrderedDict()

    if section == 'Data':
        run_params['datafile']  = settings.get(section, 'datafile')
        run_params['data_type'] = settings.get(section, 'data_type')

    elif section == 'To_Run':
        run_dict = settings.items('To_Run')
        for i, sect in enumerate(run_dict):
            loop = 'Loop_{}'.format(i)
            run_params[loop] = settings.getboolean(section, loop)

    else:
        run_params['beta_1']     = settings.getfloat(section, 'beta_1')
        run_params['beta_2']     = settings.getfloat(section, 'beta_2')

        run_params['decay'] = settings.get(section, 'decay')
        if run_params['decay'] == 'None':
            run_params['decay'] = None
        else:
            run_params['decay'] = settings.getfloat(section, 'decay')

        run_params['epochs']    = settings.getint(section,   'epochs')
        run_params['epsilon']   = settings.getfloat(section, 'epsilon')
        run_params['fmax']      = settings.getfloat(section, 'fmax')
        run_params['fmin']      = settings.getfloat(section, 'fmin')
        run_params['hc_offset'] = settings.getfloat(section, 'hc_offset')
        run_params['highcut']   = settings.getfloat(section, 'highcut')
        run_params['lowcut']    = settings.getfloat(section, 'lowcut')
        run_params['loss']      = settings.get(section, 'loss')

        run_params['lr'] = settings.get(section, 'lr')
        if run_params['lr'] == 'None':
            run_params['lr'] = None
        else:
            run_params['lr'] = settings.getfloat(section, 'lr')

        run_params['momentum']   = settings.getfloat(section, 'momentum')
        run_params['nesterov']   = settings.getboolean(section, 'nesterov')
        run_params['N_bp']       = settings.getint(section, 'N_bp')
        run_params['optimizer']  = settings.get(section, 'optimizer')
        run_params['plotDir']    = settings.get(section, 'plotDir')
        run_params['postFilter'] = settings.getboolean(section, 'postFilter')
        run_params['preFilter']  = settings.getboolean(section, 'preFilter')

        run_params['rho'] = settings.get(section, 'rho')
        if run_params['rho'] == 'None':
            run_params['rho'] = None
        else:
            run_params['rho'] = settings.getfloat(section, 'rho')

        subsystems = settings.get(section, 'subsystems')
        if ',' in subsystems:
            subsystems = subsystems.split(',')
        run_params['subsystems'] = [s.strip() for s in subsystems]

        run_params['tfrac']   = settings.getfloat(section, 'tfrac')
        run_params['ts']      = settings.getint(section, 'ts')
        run_params['verbose'] = settings.getint(section, 'verbose')

    return run_params


def get_dataset(datafile, subsystems='all', data_type='real', chanlist='all'):
    """
    get_dataset reads in a datafile and returns the dataset
    used during training. Optionally, particular subsystems
    may be given as witness channels

    Parameters
    ----------
    datafile : `string`
        full path to mat file

    subsystems : `list`
        subsystems to include in dataset
        e.g. subsystems = ['ASC', 'CAL', 'HPI', 'SUS']

    data_type : `str`
        use either "real", "mock" or "scatter"

    Returns
    -------
    dataset : `numpy.ndarray`
        test data. includes all channels except darm

    fs : `int`
        sample rate of data
    """
    mat_file = sio.loadmat(datafile)

    if data_type == 'mock' or data_type == 'scatter':
        bkgd = mat_file['background']
        darm = mat_file['darm'].T
        wits = mat_file['wit'].T
        fs   = mat_file['fs'][0][0]

    elif data_type == 'real':
        chans = [str(c.strip()) for c in mat_file['chans'][1:]]
        data  = mat_file['data']
        darm  = data[0, :].T
        bkgd  = np.zeros_like(darm)
        fs    = mat_file['fsample']

        if isinstance(subsystems, str):
            subsystems = [subsystems]

        if subsystems[0].lower() == 'all':
            if not chanlist == 'all':
                data_dict = dict(zip(chans, data))
                for k, v, in data_dict.items():
                    if k in chanlist:
                        wits.append(v)
                wits = np.array(wits).T
            else:
                wits = data.T[:, 1:]
        else:
            wits = []
            data_dict = dict(zip(chans, data))
            for subsystem in subsystems:
                for k, v, in data_dict.items():
                    if not chanlist == 'all':
                        if not k in chanlist: continue
                    if subsystem.upper() in k:
                        wits.append(v)
            wits = np.array(wits).T

    dataset = np.zeros(shape=(darm.shape[0], wits.shape[1] + 2))
    dataset[:, 0]  = darm
    dataset[:, 1]  = bkgd
    dataset[:, 2:] = wits

    return dataset, fs
