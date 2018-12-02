import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Dropout
import numpy as np
np.random.seed(3301)
import lstm_lib


# Make signals and noise
x     = np.linspace(0, 20 * np.pi, 2000)
tar   = np.sin(x)
wit1  = np.sin(6 * x)
wit2  = np.sin(15 * x)
wit3  = np.cos(21 * x)
wit4  = np.cos(9 * x)
noise = 0.5 * wit1 * wit2
y     = tar + noise

plt.plot(y[150:400], label='signal + noise', alpha=1.0)
plt.plot(wit3[150:400], label='witness channel', alpha=0.8)
plt.plot(wit4[150:400], label='witness channel', alpha=0.8)
plt.plot(tar[150:400], 'k', label='actual signal', alpha=0.8)
plt.title('Input Data Snippet')
leg = plt.legend(loc='upper right', frameon=True, framealpha=1.0)
leg.get_frame().set_edgecolor('k')
plt.savefig('NL/nonlinear_data.png')
plt.close()

# Make dataset
dataset = np.vstack((y, wit3, wit4)).T

# Split into training and testing data
tfrac   = 1000
X_train = dataset[:tfrac, 1:]
y_train = dataset[:tfrac, 0]
X_test  = dataset[tfrac:, 1:]
y_test  = dataset[tfrac:, 0]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
y_train = y_train.reshape(len(y_train))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_test  = y_test.reshape(len(y_test))

# Build the model and run it
input_shape = (X_train.shape[1], X_train.shape[2])
model = Sequential()
model.add(LSTM(8, input_shape=input_shape, return_sequences=False, kernel_initializer='glorot_normal', bias_initializer='ones'))

# model.add(LSTM(64, input_shape=input_shape, return_sequences=True, kernel_initializer='glorot_normal', bias_initializer='ones'))
# model.add(LSTM(32, return_sequences=True, kernel_initializer='glorot_normal', bias_initializer='ones'))
# model.add(LSTM(16, return_sequences=False, kernel_initializer='glorot_normal', bias_initializer='ones'))

model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, epochs=50, batch_size=10,
                    validation_data=(X_test, y_test), verbose=0)

# Make predictions
yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
residual = y[-len(yhat):] - yhat

rmse = np.sqrt(np.mean(np.square(residual - tar[-len(residual):])))
print('RMSE: {:.3f}'.format(rmse))

# Plot prediction, target and signal
plt.plot(yhat, label='prediction')
plt.plot(y[-len(yhat):], label='target')
plt.plot(tar[-len(yhat):], label='signal')
leg = plt.legend(loc='upper right', frameon=True, framealpha=1.0)
leg.get_frame().set_edgecolor('k')
plt.savefig('nonlinear_prediction.png')
plt.close()

# Plot residual with the desired signal
plt.plot(residual, label='Network Output')
plt.plot(tar[tfrac:], label='Signal')
plt.title('Nonlinear Subtraction Signal Recovery (RMSE: {:.3f})'.format(rmse))
leg = plt.legend(loc='upper right', frameon=True, framealpha=1.0)
leg.get_frame().set_edgecolor('k')
plt.ylim(top=1.8)
plt.savefig('nonlinear_recovery.png')
plt.close()

# Plot residual with the desired signal
plt.plot(residual, label='Network Output')
plt.plot(y[tfrac:], label='Signal + Noise')
plt.title('Prediction vs Target')
leg = plt.legend(loc='upper right', frameon=True, framealpha=1.0)
leg.get_frame().set_edgecolor('k')
plt.ylim(top=1.8)
plt.savefig('nonlinear_test.png')
plt.close()

plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Nonlinear Subtraction Loss')
plt.legend()
plt.savefig('nonlinear_loss.png')
plt.close()

# # Plot residual with the desired signal
# # yhat = lstm_lib.lowpass_filter(yhat, lowcut=5, fs=200)
# yhat = lstm_lib.phase_filter(yhat, lowcut=14.5, highcut=15.5, fs=200, btype='bandstop', order=12)
# residual = y[-len(yhat):] - yhat
# rmse = np.sqrt(np.mean(np.square(residual - tar[-len(residual):])))
# plt.plot(residual, label='6Hz Bandstopped Residual')
# plt.plot(tar[tfrac:], label='Signal')
# plt.plot(y[tfrac:], label='Noisy Data')
# plt.title('Linear Subtraction Signal Recovery (RMSE: {:.3f})'.format(rmse))
# leg = plt.legend(loc='upper right', frameon=True, framealpha=1.0)
# leg.get_frame().set_edgecolor('k')
# plt.ylim(top=1.8)
# plt.savefig('linear_recovery_6hz_bs.png')
# plt.close()
