"""
LSTM implementation for time series forecasting
with window size three i,e. use of t-2, t-1, and t
for prediction of value at t+1
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


def prepare_data(data, look_back=3):
    dataX = []
    dataY = []
    for i in range(len(data)-look_back-1):
        temp = data[i:(i+look_back), 0]
        dataX.append(temp)
        dataY.append(data[i+look_back, 0])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)


# get the data
df = pd.read_csv('data/AirPassengers.csv', usecols=[1], header=0, engine='python')
data = df.values
data = data.astype('float32')
logger.info('Data loaded..')
# print(data[:5])


# split the data into train and test samples
train, test = train_test_split(data, fraction=0.7)
logger.info('Train test split done..')

# print(len(train))
# print(len(test))

# prepare the uni-variate data in the form of X, y
trainX, trainY = prepare_data(train, look_back=3)
testX, testY = prepare_data(test, look_back=3)
logger.info('Data prepared as per window sequence format..')

# print(trainX.shape)
# print(trainY.shape)

# scalar transformation as the prepared data
# fit and transform predictor and outcome variable separately.

scalar = MinMaxScaler(feature_range=(0, 1))

scalar_fitX = scalar.fit(trainX)
trainX = scalar_fitX.transform(trainX)
testX = scalar_fitX.transform(testX)

scalar_fitY = scalar.fit(trainY)
trainY = scalar_fitY.transform(trainY)
testY = scalar_fitY.transform(testY)
logger.info('Scale transformation completed..')


# create LSTM model
# trainX.shape[1] i.e number of columns in the data in simple words
features = 3
# time step to consider for LSTM model
time_step = 1

# reshape the input in the form of [samples, time step, features]
trainX = np.reshape(trainX, (trainX.shape[0], time_step, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
logger.info('Data reshaped for input to LSTM..')


logger.info('Keras model building started..')
model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(time_step, features)))
model.add(Dense(1))
model.summary()

# training and model fitting
logger.info('LSTM model training started..')

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, batch_size=1, epochs=100, verbose=2)

logger.info('LSTM model training completed..')

# prediction
train_pred = model.predict(trainX)
test_pred = model.predict(testX)
logger.info('LSTM prediction completed..')

# print(train_pred.shape)
# print(train_pred[:5])

# the results are in the form of scaled value, so inverse the transformation
logger.info('Inverse transformation of prediction result..')
train_pred_inverse = scalar_fitY.inverse_transform(train_pred)
test_pred_inverse = scalar_fitY.inverse_transform(test_pred)
trainY_inverse = scalar_fitY.inverse_transform(trainY)
testY_inverse = scalar_fitY.inverse_transform(testY)

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
logger.info('Plotting the result for comparision..')

look_back = 3

train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[look_back:len(trainY_inverse)+look_back, :] = trainY_inverse

# shift test predictions for plotting
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(trainY_inverse)+(look_back*2)+1:len(data)-1, :] = testY_inverse
#
# plot baseline and predictions
plt.plot(data, 'r', label='data')
plt.plot(train_plot, 'g--', label='training')
plt.plot(test_plot, 'b:', label='test')
plt.legend(loc=0)
plt.title('LSTM using time window for Passenger forecast')
plt.savefig('plots/ap_lstm_window_result.jpg')
plt.show()

logger.info('Process completed..')
