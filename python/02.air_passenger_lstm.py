"""
Plain LSTM implementation for time series forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def train_test_split(data, fraction=0.7):
    training = int(len(data)*fraction)
    train = data[0:training, :]
    test = data[training:, :]
    return train, test


def prepare_data(data, time_step=1):
    dataX = []
    dataY = []
    for i in range(len(data)-time_step-1):
        temp = data[i:(i+time_step), 0]
        dataX.append(temp)
        dataY.append(data[i+time_step, 0])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)


# get the data
df = pd.read_csv('data/AirPassengers.csv', usecols=[1], header=0, engine='python')
data = df.values
data = data.astype('float32')
# print(data[:5])

# scalar transformation
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data)

# split the data into train and test samples
train, test = train_test_split(scaled_data, fraction=0.7)
# print(len(train))
# print(len(test))

# prepare the uni-variate data in the form of X, y
trainX, trainY = prepare_data(train, time_step=1)
testX, testY = prepare_data(test, time_step=1)


# create LSTM model
time_step = 1
features = 1

# reshape the input in the form of [#samples, time step, features]
trainX = np.reshape(trainX, (trainX.shape[0], time_step, features))
testX = np.reshape(testX, (testX.shape[0], time_step, features))

model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(time_step, features)))
model.add(Dense(1))
model.summary()

# training and model fitting
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, batch_size=1, epochs=100, verbose=2)

# prediction
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

# the results are in the form of scaled value, so inverse the transformation
train_pred_inverse = scalar.inverse_transform(train_pred)
test_pred_inverse = scalar.inverse_transform(test_pred)
trainY_inverse = scalar.inverse_transform(trainY)
testY_inverse = scalar.inverse_transform(testY)

# logging.info('Training data : {}\n'.format(trainY_inverse[:5]))
# logging.info('Training data prediction: {}\n'.format(train_pred_inverse[:5]))
# logging.info('Test data : {}\n'.format(testY_inverse[:5]))
# logging.info('Test data prediction: {}\n'.format(test_pred_inverse[:5]))


# RMSE calculation
train_rmse = np.sqrt(mean_squared_error(trainY_inverse, train_pred_inverse))
test_rmse = np.sqrt(mean_squared_error(testY_inverse, test_pred_inverse))

logger.info('Training RMSE: {}'.format(train_rmse))
logger.info('Test RMSE: {}'.format(test_rmse))


# plotting the results and comparision
# shift train predictions for plotting
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(trainY_inverse)+time_step, :] = trainY_inverse

# shift test predictions for plotting
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(trainY_inverse)+(time_step*2)+1:len(data)-1, :] = testY_inverse

# plot baseline and predictions
plt.plot(data, 'r', label='data')
plt.plot(train_plot, 'g--', label='training')
plt.plot(test_plot, 'b:', label='test')
plt.legend(loc=0)
plt.title('Basic LSTM for Passenger forecast')
plt.savefig('plots/ap_basic_lstm_result.jpg')
plt.show()
