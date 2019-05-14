# Custom rc file: $HOME/.config/matplotlib/stylelib/custom.mplstyle
from __future__ import division
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.signal as sig
import nds2
import os
os.environ['NDS2_CLIENT_ALLOW_DATA_ON_TAPE'] = '1'
import numpy as np
from scipy.linalg import toeplitz


def set_axes_style(fpath       = '/home/rich.ormiston/fonts/Custom/tnr.ttf',
                   title       = 'Title',
                   title_size  = 14,
                   xlabel      = 'xlabel',
                   xlabel_size = 12,
                   ylabel      = 'ylabel',
                   ylabel_size = 12,
                   xaxis_size  = 10,
                   yaxis_size  = 10):

    plt.style.use('custom')
    fig, ax = plt.subplots()

    prop = fm.FontProperties(fname=fpath)

    ax.set_title(title,   fontproperties=prop, size=title_size)
    ax.set_xlabel(xlabel, fontproperties=prop, size=xlabel_size)
    ax.set_ylabel(ylabel, fontproperties=prop, size=ylabel_size)

    for label in ax.get_xticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(xaxis_size)

    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(yaxis_size)

    return prop


def get_strain(signal, fs=512, nfft=4096):
    ff1, psd = sig.welch(signal, fs=fs, nperseg=nfft, axis=-1)
    strain = np.sqrt(psd).T
    return freqs, strain


def stream_data(start, channels,
                dur  = 600,
                fsup = 512,
                ifo  = 'H1'):

    if ifo == 'H1':
        server = 'nds.ligo-wa.caltech.edu'
    else:
        server = 'nds.ligo-la.caltech.edu'

    # Setup connection to the NDS
    conn = nds2.connection(server, 31200)
    data = []
    for i in range(len(channels)):
        temp = conn.fetch(start, start + dur, [channels[i]])
        data.append(temp)

    # Get the data and stack it (the data are the columns)
    vdata = []
    for k in range(len(channels)):
        fsdown = data[k][0].channel.sample_rate
        down_factor = int(fsdown // fsup)

        fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                            window='blackmanharris')

        # Using fir_aa[1:-1] cuts off a leading and trailing zero
        downdata = sig.decimate(data[k][0].data, down_factor,
                                ftype = sig.dlti(fir_aa[1:-1], 1.0),
                                zero_phase = True)
        vdata.append(downdata)

    return np.array(vdata).T


def bandpass_filter(dataset,
                    lowcut  = 4.0,
                    highcut = 20.0,
                    order   = 8,
                    btype   = 'bandpass',
                    fs      = 512):

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


def xcorr(x, y, maxlag=None):
    """Calculate cross correlation of `x` and `y`, which have equal lengths.

    This function accepts a `maxlag` parameter, which truncates the result to
    only return the cross correlation of samples that are within `maxlag`
    samples of each other.

    For long input vectors, it is more efficient to calculate this convolution
    via FFT. As a rough heuristic, this function switches to an FFT based
    method when the input length exceeds 500 samples.

    Parameters
    ----------
    x : array_like
        First input array.
    y : array_like
        Second input array.

    Returns
    -------
    c : ndarray
        Cross correlation result.
    """
    xl = x.size
    yl = y.size
    if xl != yl:
        raise ValueError('x and y must be equal length')

    if maxlag is None:
        maxlag = xl - 1
    else:
        maxlag = int(maxlag)
    if maxlag >= xl or maxlag < 1:
        raise ValueError('maglags must be None or strictly positive')

    if xl > 500:  # Rough estimate of speed crossover
        c = sig.fftconvolve(x, y[::-1])
        c = c[xl - 1 - maxlag:xl + maxlag]
    else:
        c = np.zeros(2*maxlag + 1)
        for i in xrange(maxlag+1):
            c[maxlag-i] = np.correlate(x[0:min(xl, yl-i)],
                                       y[i:i+min(xl, yl-i)])
            c[maxlag+i] = np.correlate(x[i:i+min(xl-i, yl)],
                                       y[0:min(xl-i, yl)])
    return c


def block_levinson(y, L):
    """
    Solve the matrix equation T x = y for symmetric, block Toeplitz, T.

    Block Levinson recursion for efficiently solving symmetric block Toeplitz
    matrix equations. The matrix T is never stored in full (because it is
    large and mostly redundant), so the input parameter L is actually the
    leftmost "block column" of T (the leftmost d columns where d is the block
    dimension).

    References
    ----------
        Akaike, Hirotugu (1973). "Block Toeplitz Matrix Inversion".  SIAM J.
        Appl. Math. 24 (2): 234-241
   """
    d = L.shape[1]               # Block dimension
    N = int(L.shape[0]/d)        # Number of blocks

    # This gets the bottom block row B from the left block column L
    B = np.reshape(L, [d, N, d], order='F')
    B = B.swapaxes(1, 2)
    B = B[..., ::-1]
    B = np.reshape(B, [d, N*d], order='F')

    f = np.linalg.inv(L[:d, :])  # "Forward" vector
    b = f                        # "Backward" vector
    x = np.dot(f, y[:d])         # Solution vector

    Ai = np.eye(2*d)
    G = np.zeros((d*N, 2*d))
    for n in range(2, N+1):
        ef = np.dot(B[:, (N-n)*d:N*d], np.vstack((f, np.zeros((d, d)))))
        eb = np.dot(L[:n*d, :].T, np.vstack((np.zeros((d, d)), b)))
        ex = np.dot(B[:, (N-n)*d:N*d], np.vstack((x, np.zeros((d, 1)))))
        Ai[:d, d:] = eb
        Ai[d:, :d] = ef
        A = np.linalg.inv(Ai)
        l = d*(n-1)
        G[:l, :d] = f
        G[d:l+d, d:] = b
        fn = np.dot(G[:l+d, :], A[:, :d])
        bn = np.dot(G[:l+d, :], A[:, d:])
        f = fn
        b = bn
        x = np.vstack((x, np.zeros((d, 1)))) + np.dot(b, y[(n-1)*d:n*d]-ex)

    W = x
    return W


def wiener_fir(tar, wit, N, method='levinson'):
    """
    Calculate the optimal FIR Wiener subtraction filter for multiple inputs.

    This function may use the Levinson-Durbin algorithm to greatly enhance the
    speed of calculation, at the expense of instability when given highly
    coherence input signals. Brute-force inversion is available as an
    alternative.

    Parameters
    ----------
    tar : array_like
        Time series of target signal.
    wit : list of 1D arrays, or MxN array
        List of the time series of M witness signals, each witness must have
        the same length as the target signal, N.
    N : integer
        FIR filter order to be used. The filter response time is given by the
        product N * fs, where fs is the sampling frequency of the input
        signals.
    method : { 'levinson', 'brute' }, optional
        Selects the matrix inversion algorithm to be used. Defaults to
        'levinson'.

    Returns
    -------
    W : ndarray, shape (N, M)
        Columns of FIR filter coefficents that optimally estimate the target
        signal from the witness signals.
    """

    method = method.lower()
    if method not in ['levinson', 'brute']:
        raise ValueError('Unknown method type')

    N = int(N)
    if isinstance(wit, np.ndarray):
        if len(wit.shape) == 1:
            wit = np.reshape(wit, (1, wit.size), order='A')
        M = wit.shape[0]
    elif isinstance(wit, list):
        M = len(wit)
        wit = np.vstack([w for w in wit])

    P = np.zeros(M*(N+1))
    # Cross correlation
    for m in range(M):
        top = m * (N+1)
        bottom = (m+1) * (N+1)
        p = xcorr(tar, wit[m, :], N)
        P[top:bottom] = p[N:2*N+1]

    if method.lower() == 'levinson':
        P = np.reshape(P, [N+1, M], order='F')
        P = np.reshape(P.T, [M*(N+1), 1], order='F')
        R = np.zeros((M*(N+1), M))
        for m in range(M):
            for ii in range(m+1):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = np.flipud(rmi[:N+1])
                top = m * (N+1)
                bottom = (m+1) * (N+1)
                R[top:bottom, ii] = Rmi
                if ii != m:
                    Rmi = rmi[N:]
                    top = ii * (N+1)
                    bottom = (ii+1) * (N+1)
                    R[top:bottom, m] = Rmi

        R = np.reshape(R, [N+1, M, M], order='F')
        R = R.swapaxes(0, 1)
        R = np.reshape(R, [M*(N+1), M], order='F')

        W = block_levinson(P, R)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N+1], order='F')

    elif method.lower() == 'brute':
        R = np.zeros((M*(N+1), M*(N+1)))
        for m in range(M):
            for ii in range(m, M):
                rmi = xcorr(wit[m, :], wit[ii, :], N)
                Rmi = toeplitz(np.flipud(rmi[:N+1]), rmi[N:2*N+1])
                top = m * (N+1)
                bottom = (m+1) * (N+1)
                left = ii * (N+1)
                right = (ii+1) * (N+1)
                R[top:bottom, left:right] = Rmi
                if ii != m:
                    R[left:right, top:bottom] = Rmi.T
        W = np.linalg.solve(R, P)

        #  Return each witness' filter as a row
        W = np.reshape(W, [M, N+1])

    return W
