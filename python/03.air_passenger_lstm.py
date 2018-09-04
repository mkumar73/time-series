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
from keras.layers import LSTM

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


dataframe = pd.read_csv('data/AirPassengers.csv', usecols=[1], header=0, engine='python')
data = dataframe.values
data = data.astype('float32')


def scalar_transformation(data):
    scalar = MinMaxScaler(feature_range=(0, 1))
    transformed_data = scalar.fit_transform(data)
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


#
# data = scalar_transformation(data)
# print(data[:5])
#
# train, test = train_test_split(data, fraction=0.7)
# print(len(train))
# print(len(test))
#
# X, y = prepare_data(train, time_step=1)
# print(len(X))
# print(len(y))

