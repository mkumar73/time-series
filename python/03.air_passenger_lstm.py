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


data = pd.read_csv('data/AirPassengers.csv', names=['Month', 'Passengers'], usecols=[1], header=0)

print(data.head())
plt.plot(data)
plt.show()
