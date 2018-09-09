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
from ts_utils import tsplot

np.random.seed(1)
n_samples = 1000
data = noise = np.random.normal(size=1000)

# ar(1) model : x[t] = alpha * x[t-1] + noise

# case : 1, alpha = 0.5
alpha = 0.7
data_1 = np.zeros(len(data))
for t in range(len(data)):
    data[t] = alpha * data[t-1] + noise[t]


tsplot(data, lags=20, show=True)