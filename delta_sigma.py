from __future__ import division
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.plotter import SpectrogramPlot
import numpy as np
import os


FILE = 'badTimes2.txt'
times = np.genfromtxt(FILE, delimiter='\n', dtype=np.int)
dur = 192.0
N_freq_bands = 3
f1_low, f1_high = 0.0, 30.0
f2_low, f2_high = f1_high, 100.0
f3_low, f3_high = f2_high, 300.0
plotDir = './Plots/O3_T2'
sigma_cut = 2.0
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

for ix, start in enumerate(times):
    print('Making plot {}'.format(ix))
    h1strain = TimeSeries.get('H1:GDS-CALIB_STRAIN', start - dur / 2.0,
                              start + dur + dur / 2.0)
    specgram = h1strain.spectrogram(6, fftlength=4) ** (1/2.)
    for i in range(specgram.value.shape[1]):
        specgram.value[:, i] = specgram.value[:, i]\
                / np.mean(specgram.value[:, i])

    # plot original spectrograms
    plot = SpectrogramPlot()
    ax = plot.gca()
    ax.plot(specgram)
    ax.set_epoch(start)
    ax.set_xlim(start - dur / 2.0, start + dur + dur / 2.0)
    ax.set_ylim(7, 300)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title('LIGO-Hanford strain data: {0} - {1}'.format(start - dur / 2.0,
                 start + dur  + dur / 2.0))
    plot.add_colorbar(label=r'Strain noise [1/\rtHz]')
    plot.savefig('{0}/original_{1}.png'.format(plotDir, start))
    plot.close()
    plt.close()

    # integrate over frequencies and notch according to narrowband estimate
    df = specgram.df.value
    N_freqs1 = int((f1_high - f1_low) / df)
    narrowband = np.zeros(shape=(specgram.value.shape[0], N_freq_bands))

    for i in range(narrowband.shape[0]):
        narrowband[i, 0] = np.sum(df * specgram.value[i, :N_freqs1])
        narrowband[i, 1] = np.sum(df * specgram.value[i, N_freqs1:int(f2_high / df)])
        narrowband[i, 2] = np.sum(df * specgram.value[i, int(f2_high / df):int(f3_high / df)])

    for i in range(narrowband.shape[1]):
        narrowband[:, i] = (narrowband[:, i] - np.mean(narrowband[:, i]))\
                            / np.std(narrowband[:, i])

    to_notch1 = np.where(narrowband[:, 0] >= sigma_cut)
    to_notch2 = np.where(narrowband[:, 1] >= sigma_cut)
    to_notch3 = np.where(narrowband[:, 2] >= sigma_cut)
    total = list(narrowband[:, 0]) + list(narrowband[:, 1]) + list(narrowband[:, 2])

    for notch in to_notch1:
        specgram.value[notch, :N_freqs1] = 0
    for notch in to_notch2:
        specgram.value[notch, N_freqs1:int(f2_high / df)] = 0
    for notch in to_notch3:
        specgram.value[notch, int(f2_high / df):int(f3_high / df)] = 0

    for i in range(specgram.value.shape[1]):
        specgram.value[:, i] = specgram.value[:, i]\
                / np.mean(specgram.value[:, i])

    # plot histogram of narrowband estimates in each 6s segment
    plt.hist(narrowband[:, 0], bins=np.arange(np.min(narrowband) - 0.2,
                                              np.max(narrowband) + 2 * 0.3, 0.3),
                                              label='{0} - {1} Hz'.format(f1_low, f1_high),
                                              histtype='step')
    plt.hist(narrowband[:, 1], bins=np.arange(np.min(narrowband) - 0.2,
                                              np.max(narrowband) + 2 * 0.3, 0.3),
                                              label='{0} - {1} Hz'.format(f2_low, f2_high),
                                              histtype='step')
    plt.hist(narrowband[:, 2], bins=np.arange(np.min(narrowband) - 0.2,
                                              np.max(narrowband) + 2 * 0.3, 0.3),
                                              label='{0} - {1} Hz'.format(f3_low, f3_high),
                                              histtype='step')
    plt.title('Narrowband Estimates: {0} - {1}'.format(start - dur / 2.0,
                                                       start + 3.0 * dur / 2.0),
                                                       fontsize=16)
    plt.xlabel('$\sigma_{NB}$', fontsize=14)
    plt.legend()
    plt.ylabel('Counts', fontsize=14)
    plt.tight_layout()
    plt.savefig('{0}/hist_{1}.png'.format(plotDir, start))
    plt.close()

    # Narrowband stats from all 3 frequency bands with gaussian profile
    plt.hist(total, 70, density=True)
    plt.xlabel('$\sigma_{NB}$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    if np.min(total) > -3.5:
        ll = -3.5
    else:
        ll = np.min(total)

    if np.max(total) < 3:
        ul = 3.5
    else:
        ul = np.max(total)
    plt.axvline(3.0, color='red', linestyle='--')
    plt.axvline(-3.0, color='red', linestyle='--')
    plt.xlim([ll, ul])
    plt.title('Total Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig('{0}/total_{1}.png'.format(plotDir, start))
    plt.close()

    # plot spectrogram with removed times
    plot = SpectrogramPlot()
    ax = plot.gca()
    ax.plot(specgram)
    ax.set_epoch(start)
    ax.set_xlim(start - dur / 2.0, start + dur + dur / 2.0)
    ax.set_ylim(7, 300)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title('LIGO-Hanford strain data: {0} - {1}'.format(start - dur / 2.0,
                 start + dur  + dur / 2.0))
    plot.add_colorbar(label=r'Strain noise [1/\rtHz]')
    plot.savefig('{0}/{1}.png'.format(plotDir, start))
    plot.close()
    plt.close()


# for ix, start in enumerate(times[:50]):
#     print('Making plot {}'.format(ix))
#     h1strain = TimeSeries.get('H1:GDS-CALIB_STRAIN', start - dur / 2.0,
#                               start + dur + dur / 2.0)
#     specgram = h1strain.spectrogram(6, fftlength=4) ** (1/2.)
#     for i in range(specgram.value.shape[1]):
#         specgram.value[:, i] = specgram.value[:, i]\
#                 / np.mean(specgram.value[:, i])

#     # integrate over frequencies and notch according to narrowband estimate
#     df = specgram.df.value
#     narrowband = np.zeros(shape=specgram.value.shape[0])
#     for i in range(narrowband.shape[0]):
#         narrowband[i] = np.sum(df * specgram.value[i, :])

#     narrowband = (narrowband - np.mean(narrowband)) / np.std(narrowband)
#     to_notch = np.where(narrowband >= 3)
#     print('times to notch:', len(to_notch))
#     for notch_time in to_notch:
#         specgram.value[notch_time, :] = 0

#     for i in range(specgram.value.shape[1]):
#         specgram.value[:, i] = specgram.value[:, i]\
#                 / np.mean(specgram.value[:, i])

#     # plot histogram of narrowband estimates in each 6s segment
#     plt.hist(narrowband, bins=np.arange(np.min(narrowband) - 0.2,
#                                         np.max(narrowband) + 2 * 0.2, 0.2))
#     plt.title('Narrowband Estimates: {0} - {1}'.format(start - dur / 2.0,
#                                                        start + 3.0 * dur / 2.0))
#     plt.xlabel('Narrowband Estimate')
#     plt.legend()
#     plt.ylabel('Counts')
#     plt.savefig('Narrowband/hist_{}.png'.format(start - dur / 2.0))
#     plt.close()

#     # plot spectrogram with removed times
#     plot = SpectrogramPlot()
#     ax = plot.gca()
#     ax.plot(specgram)
#     ax.set_epoch(start)
#     ax.set_xlim(start - dur / 2.0, start + dur + dur / 2.0)
#     ax.set_ylim(7, 80)
#     ax.set_yscale('log')
#     ax.set_ylabel('Frequency [Hz]')
#     ax.set_title('LIGO-Hanford strain data: {0} - {1}'.format(start - dur / 2.0,
#                  start + dur  + dur / 2.0))
#     plot.add_colorbar(label=r'Strain noise [1/\rtHz]')
#     plot.savefig('Narrowband/{}.png'.format(start))
#     plot.close()
#     plt.close()
