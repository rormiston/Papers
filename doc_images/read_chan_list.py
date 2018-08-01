from __future__ import division
from ConfigParser import ConfigParser
import numpy as np
import sys
import nds2
from astropy.time import Time
import os
os.environ['NDS2_CLIENT_ALLOW_DATA_ON_TAPE'] = '1'
import pandas as pd
import scipy.signal as sig
import scipy.io as sio


def get_channels(ini_file, subsystems):
    configs = ConfigParser()
    configs.read(ini_file)
    configs.optionxform=str
    channels = configs.get(subsystems, 'channels').split('\n')
    channels = [c.strip('\n') for c in channels if len(c) > 1]
    return channels


def stream_data_from_master_list(ini_file, subsystems):

    # Read config file
    settings = ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)

    # Unpack configs
    dur        = settings.getint('Data', 'duration')
    fsup       = settings.getint('Data', 'fs')
    ifo        = settings.get('Data', 'ifo')
    output     = settings.get('Data', 'output')
    portNumber = settings.getint('Data', 'portNumber')
    save       = settings.getboolean('Data', 'save_mat')
    times      = settings.get('Data', 'data_start')
    fsup = 256

    nds_osx = ('/opt/local/Library/Frameworks/Python.framework/' +
               'Versions/2.7/lib/python2.7/site-packages/')
    nds_sandbox = '/usr/lib/python2.7/dist-packages/'

    import sys
    if os.path.exists(nds_osx):
        sys.path.append(nds_osx)
    elif os.path.exists(nds_sandbox):
        sys.path.append(nds_sandbox)

    # Collect channels and times
    channels = get_channels(ini_file, subsystems)
    channels.insert(0, 'H1:GDS-CALIB_STRAIN')

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

    for chan in channels:
        try:
            temp = conn.fetch(t_start, t_start + dur, [chan])
            sys.stdout.write("\033[0;32m")
            sys.stdout.write(u'\r  [{}] '.format(u'\u2713'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(chan))
            sys.stdout.write('\n')
            sys.stdout.flush()
        except:
            sys.stdout.write("\033[1;31m")
            sys.stdout.write(u'\r  [{}] '.format(u'\u2717'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(chan))
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
