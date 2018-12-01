# this is an exercise to understand basic behind
# AR(P) models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as scs

# my utils library
from ts_utils import stationary_check, tsplot

np.random.seed(1)
n_samples = 1000
data = noise = np.random.normal(size=1000)

# ar(1) model : x[t] = alpha * x[t-1] + noise

# case : 1, alpha = 0.5
alpha = 0.7
for t in range(len(data)):
    data[t] = alpha * data[t-1] + noise[t]

tsplot(data, lags=20, show=True)
stationary_check(data)

# lets fit the model to detect the ar order

ar1 = smt.AR(data).fit(maxlag=10, ic='aic', trend='nc')
order = smt.AR(data).select_order(maxlag=10, ic='aic', trend='nc')
print('Estimated order of AR model: {}'.format(order))
print('Estimated alpha of AR model: {}'.format(ar1.params[0]))
