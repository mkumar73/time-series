
# Weather Forecasting using Max Plank monthly data.
# data is captured every 10 min for more than 14 paramters
# like temperature, CO2 level, Air pressure etc.
# We will build a LSTM model to predict the temperature for next 24 hours.
# the data has details for Jan 2018-June 2018.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


df = pd.read_csv('data/mpi_weather_012018.csv', usecols=[2], header=0)
logger.info('Data loading completed..')

data = df.values
# print(data[:5, ])
logger.info('Actual data size: {}'.format(data.shape))

plt.plot(data)
plt.title('Raw temperature data for Jan 2018')
# plt.show()

scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data)
logger.info('Data scaling completed..')

# print(scaled_data[:5, ])
plt.plot(scaled_data)
plt.title('Scaled temperature data for Jan 2018')
# plt.show()

# separate the last 2 days hours data for testing
train_and_val = scaled_data[:-7*144, :]
test_set = scaled_data[-7*144:, :]
logger.info('Train + val data size: {}'.format(train_and_val.shape))
logger.info('Test data size: {}'.format(test_set.shape))
logger.info('Training test split..')


def train_val_split(data, fraction=0.70):
    train_size = int(len(data)*fraction)
    train_data = data[:train_size, :]
    val_data = data[train_size:, :]
    return train_data, val_data


training_data, val_data = train_val_split(train_and_val, fraction=0.70)
logger.info('Training data size: {}'.format(training_data.shape))
logger.info('Validation data size: {}'.format(val_data.shape))


# data generator for Keras fit_generator
def data_generator(data, look_back=720, batch_size=128):

    dataX = []
    dataY = []
    for i in range(len(data) - look_back - 1):
        temp = data[i:(i + look_back), 0]
        dataX.append(temp)
        dataY.append(data[i + look_back, 0])

    dataX = np.array(dataX)
    dataY = np.array(dataY).reshape(-1, 1)

    for i in range(len(data) // batch_size):
        start = i * batch_size
        end = start + batch_size
        yield dataX[start:end, :], dataY[start:end, :]


train_gen = data_generator(training_data, look_back=720, batch_size=128)
val_gen = data_generator(val_data, look_back=720, batch_size=128)
test_gen = data_generator(test_set, look_back=720, batch_size=128)
logger.info('Train, test and validation generator completed..')

# for i in range(2):
#     x, y = next(val_gen)
#     print(x.shape)
#     print(y[:5, :])

time_step = 6
look_back = 720
model = Sequential()
model.add(LSTM(10, activation='tanh', input_shape=(time_step, look_back)))
