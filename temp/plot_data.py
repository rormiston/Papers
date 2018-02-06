import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', '-d',
			help = 'CSV file to read',
			default = None,
			dest = "datafile")
    return parser.parse_args()
	
params = parse_command_line()
if not os.path.isfile(params.datafile):
    sys.exit('ERROR: File not found')

data = np.genfromtxt(params.datafile, delimiter=',')
data = data[500:900, :]
time = data[:, 0]

# Bottom Photodiode
bottom = data[:, 1]
top = data[:, 2]
plt.plot(time, bottom, label='bottom')
plt.title('Bottom Photodiode')
plt.ylabel('Volts (V)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('bottom_photodiode.png')
plt.close()

# Top Photodiode
plt.plot(time, top, label='top')
plt.title('Top Photodiode')
plt.ylabel('Volts (V)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('top_photodiode.png')
plt.close()

# z Magnetometer
z_mag = data[:, 3]
y_mag = data[:, 4]
x_mag = data[:, 5]
plt.plot(time, z_mag, label='z')
plt.title('$\hat{z}$-Magnetometer')
plt.ylabel('B (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('z_magnetometer.png')
plt.close()

# y Magnetometer
plt.plot(time, y_mag, label='y')
plt.title('$\hat{y}$-Magnetometer')
plt.ylabel('B (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('y_magnetometer.png')
plt.close()

# x Magnetometer
plt.plot(time, x_mag, label='x')
plt.title('$\hat{x}$-Magnetometer')
plt.ylabel('B (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('x_magnetometer.png')
plt.close()

z_S = data[:, 6]
y_S = data[:, 7]
x_S = data[:, 8]

# z Seismometer
plt.plot(time, z_S, label='z')
plt.title('$\hat{z}$-Seismometer')
plt.ylabel('Displacement (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('z_seismometer.png')
plt.close()

# y Seismometer
plt.plot(time, y_S, label='y')
plt.title('$\hat{y}$-Seismometer')
plt.ylabel('Displacement (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('y_seismometer.png')
plt.close()

# x Seismometer
plt.plot(time, x_S, label='x')
plt.title('$\hat{x}$-Seismometer')
plt.ylabel('Displacement (Volts)')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.savefig('x_seismometer.png')
plt.close()
