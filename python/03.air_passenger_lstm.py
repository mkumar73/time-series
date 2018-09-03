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

print(data[:5])
plt.plot(data)
plt.show()
