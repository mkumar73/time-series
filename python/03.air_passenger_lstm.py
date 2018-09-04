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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# scale the data to (0, 1) for LSTM
def scalar_transformation(data, inverse=False):
    scalar = MinMaxScaler(feature_range=(0, 1))
    transformed_data = scalar.fit_transform(data)
    if inverse:
        data = scalar.inverse_transform(data)
        return data
    else:
        return transformed_data


def train_test_split(data, fraction=0.7):
    training = int(len(data)*fraction)
    train = data[0:training, :]
    test = data[training:, :]
    return train, test


def prepare_data(data, time_step=1):
    dataX = []
    dataY = []
    for i in range(len(data)-time_step-1):
        temp = data[i, :]
        dataX.append(temp)
        dataY.append(data[i+time_step, :])
    return np.array(dataX), np.array(dataY)


# get the data
df = pd.read_csv('data/AirPassengers.csv', usecols=[1], header=0, engine='python')
data = df.values
data = data.astype('float32')

# scalar transformation
data = scalar_transformation(data)
# print(data[:5])

# split the data into train and test samples
train, test = train_test_split(data, fraction=0.7)
# print(len(train))
# print(len(test))

# prepare the uni-variate data in the form of X, y
trainX, trainY = prepare_data(train, time_step=1)
testX, testY = prepare_data(test, time_step=1)

# reshape the input in the form of [samples, time step, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create LSTM model
time_step = 1

model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(1, time_step)))
model.add(Dense(1))
model.summary()

# training and model fitting
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, batch_size=1, epochs=100, verbose=2)

# prediction
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

# the results are in the form of scaled value, so inverse the transformation
train_pred_actual = scalar_transformation(train_pred, inverse=True)
test_pred_actual = scalar_transformation(test_pred, inverse=True)

logging.info('Actual data: {}'.format(data[:5]))
logging.info('Training data prediction: {}'.format(train_pred_actual[:5]))
logging.info('Test data prediction: {}'.format(test_pred_actual[:5]))
