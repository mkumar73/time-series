
# Weather Forecasting using Max Plank monthly data.
# data is captured every 10 min for more than 14 paramters
# like temperature, CO2 level, Air pressure etc.
# We will build a LSTM model to predict the temperature for next 24 hours.
# the data has details for Jan-2018.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

from keras.models import Sequential
from keras.layers import Dense, LSTM

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


df = pd.read_csv('data/mpi_weather_012018.csv', usecols=[2], header=0)
logger.info('Data loading completed..')

data = df.values
print(data[:5, ])
print(data.shape)
plt.plot(data)
plt.show()



