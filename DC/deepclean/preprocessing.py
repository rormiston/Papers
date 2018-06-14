from __future__ import division
import nds2
import os
os.environ['NDS2_CLIENT_ALLOW_DATA_ON_TAPE'] = '1'
import pandas as pd
import scipy.signal as sig
import scipy.io as sio
import sys
import numpy as np

from astropy.time import Time
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
        run_params['datafile'] = settings.get(section, 'datafile')
        run_params['data_type'] = settings.get(section, 'data_type')
        run_params['save_mat'] = settings.getboolean('Data', 'save_mat')

    elif section == 'To_Run':
        run_dict = settings.items('To_Run')
        for i, sect in enumerate(run_dict):
            loop = 'Loop_{}'.format(i)
            run_params[loop] = settings.getboolean(section, loop)

    elif section == 'Webpage':
        run_params['basedir'] = settings.get(section, 'basedir')

    else:
        run_params['activation'] = settings.get(section, 'activation')
        run_params['beta_1'] = settings.getfloat(section, 'beta_1')
        run_params['beta_2'] = settings.getfloat(section, 'beta_2')
        run_params['bias_initializer'] = settings.get(section, 'bias_initializer')

        run_params['decay'] = settings.get(section, 'decay')
        if run_params['decay'] == 'None':
            run_params['decay'] = None
        else:
            run_params['decay'] = settings.getfloat(section, 'decay')

        run_params['dropout'] = settings.getfloat(section, 'dropout')
        run_params['epochs'] = settings.getint(section,   'epochs')
        run_params['epsilon'] = settings.getfloat(section, 'epsilon')
        run_params['fmax'] = settings.getfloat(section, 'fmax')
        run_params['fmin'] = settings.getfloat(section, 'fmin')
        run_params['hc_offset'] = settings.getfloat(section, 'hc_offset')
        run_params['highcut'] = settings.getfloat(section, 'highcut')
        run_params['kernel_initializer'] = settings.get(section, 'kernel_initializer')
        run_params['lookback'] = settings.getint(section, 'lookback')
        run_params['lowcut'] = settings.getfloat(section, 'lowcut')
        run_params['loss'] = settings.get(section, 'loss')

        run_params['lr'] = settings.get(section, 'lr')
        if run_params['lr'] == 'None':
            run_params['lr'] = None
        else:
            run_params['lr'] = settings.getfloat(section, 'lr')

        run_params['momentum'] = settings.getfloat(section, 'momentum')
        run_params['nesterov'] = settings.getboolean(section, 'nesterov')
        run_params['N_bp'] = settings.getint(section, 'N_bp')
        run_params['optimizer'] = settings.get(section, 'optimizer')
        run_params['plotDir'] = settings.get(section, 'plotDir')
        run_params['postFilter'] = settings.getboolean(section, 'postFilter')
        run_params['preFilter'] = settings.getboolean(section, 'preFilter')
        run_params['dropout'] = settings.getfloat(section, 'dropout')
        run_params['recurrent_dropout'] = settings.get(section, 'recurrent_dropout')

        run_params['rho'] = settings.get(section, 'rho')
        if run_params['rho'] == 'None':
            run_params['rho'] = None
        else:
            run_params['rho'] = settings.getfloat(section, 'rho')

        subsystems = settings.get(section, 'subsystems')
        if ',' in subsystems:
            subsystems = [s.strip() for s in subsystems.split(',')]
        else:
            subsystems = [subsystems]
        run_params['subsystems'] = subsystems

        run_params['tfrac']= settings.getfloat(section, 'tfrac')
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


def read_chans_and_times(chanlist):
    chans_times = []
    with open(chanlist) as f:
        for line in f.readlines():
            if ',' in line:
                terms = line.split(',')
                chans = terms[0].strip()
                times = int(terms[-1].strip('\n').strip())
            else:
                chans = line.strip('\n')
                times = None

            chans_times.append((chans, times))

    chans = [chans_times[i][0] for i in range(len(chans_times))]
    times = [chans_times[i][1] for i in range(len(chans_times))]
    return chans, times


def stream_data(ini_file):

    # Read config file
    settings = ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)

    # Unpack configs
    dur        = settings.getint('Data', 'duration')
    fname      = settings.get('Data', 'chanlist')
    fsup       = settings.getint('Data', 'fs')
    ifo        = settings.get('Data', 'ifo')
    output     = settings.get('Data', 'output')
    portNumber = settings.getint('Data', 'portNumber')
    save       = settings.getboolean('Data', 'save_mat')
    times      = settings.get('Data', 'data_start')

    nds_osx = ('/opt/local/Library/Frameworks/Python.framework/' +
               'Versions/2.7/lib/python2.7/site-packages/')
    nds_sandbox = '/usr/lib/python2.7/dist-packages/'

    import sys
    if os.path.exists(nds_osx):
        sys.path.append(nds_osx)
    elif os.path.exists(nds_sandbox):
        sys.path.append(nds_sandbox)

    # Collect channels and times
    chan_head = ifo + ':'
    chanlines, custom_times = read_chans_and_times(fname)
    channels = [chan_head + line for line in chanlines]

    # Get data start time
    if ifo == 'L1':
        ndsServer = 'nds.ligo-la.caltech.edu'
    elif ifo == 'H1':
        ndsServer = 'nds.ligo-wa.caltech.edu'
    else:
        sys.exit("unknown IFO specified")

    # Setup connection to the NDS
    try:
        conn = nds2.connection(ndsServer, portNumber)
    except RuntimeError:
        print('ERROR: Need to run `kinit albert.einstein` before nds2 '
              'can establish a connection')
        sys.exit(1)

    if __debug__:
        print("Output sample rate: {} Hz".format(fsup))
        # print("Channel List:\n-------------")
        # print("\n".join(channels))

    # Setup start and stop times
    t = Time(times, format='iso', scale='utc')
    t_start = int(t.gps)

    print("Getting data from " + ndsServer + "...")
    data = []
    for i in range(len(custom_times)):
        if custom_times[i] == None:
            custom_times[i] = t_start

        try:
            temp = conn.fetch(custom_times[i], custom_times[i] + dur, [channels[i]])
            sys.stdout.write("\033[0;32m")
            sys.stdout.write(u'\r  [{}] '.format(u'\u2713'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(channels[i]))
            sys.stdout.write('\n')
            sys.stdout.flush()
        except:
            sys.stdout.write("\033[1;31m")
            sys.stdout.write(u'\r  [{}] '.format(u'\u2717'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(channels[i]))
            sys.stdout.write('\n')
            sys.stdout.flush()

        data.append(temp)

    # Get the data and stack it (the data are the columns)
    vdata = []
    for k in range(len(channels)):
        fsdown = data[k][0].channel.sample_rate
        down_factor = int(fsdown // fsup)

        fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                            window='blackmanharris')

        # Prevent ringing from DC offset
        DC = np.mean(data[k][0].data)

        # Using fir_aa[1:-1] cuts off a leading and trailing zero
        downdata = sig.decimate(data[k][0].data, down_factor,
                                ftype = sig.dlti(fir_aa[1:-1], 1.0),
                                zero_phase = True)
        vdata.append(downdata)

    if save:
        if not os.path.isdir('Data'):
            os.system('mkdir Data')

        # save to a hdf5 format
        if output == "None":
            funame = 'Data/' + ifo + '_data_array.mat'
        else:
            funame = 'Data/' + output

        sio.savemat(funame, mdict={'data': vdata, 'fsample': fsup, 'chans': channels},
                    do_compression=True)

        print("Data saved as " + funame)
    else:
        return np.array(vdata).T, fsup


# define lookback function
def do_lookback(data, steps=1, validation=False):
    temp = np.zeros((data.shape[0] - steps, steps + 1, data.shape[1]))
    temp[:, 0, :] = data[steps:, :]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[2]):
            temp[i, 1:, j] = data[i:i + steps, j][::-1]

    if validation:
        temp = temp.reshape((temp.shape[0], temp.shape[1]))

    for i in range(temp.shape[0]):
        for j in range(temp.shape[2]):
            temp[i, :, j] = temp[i, :, j][::-1]

    return temp
