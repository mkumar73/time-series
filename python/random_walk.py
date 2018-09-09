# Illustration of TSA for:
# 1. White noise
# 2. Random walk
# 3. Transform random walk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import scipy.stats as scs
# plt.style.use('ggplot')


def tsplot(data, lags, show=True):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    layout = (3, 2)

    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    prob_ax = plt.subplot2grid(layout, (2, 1))

    data.plot(ax=ts_ax)
    ts_ax.set_title('Line plot for the data')
    smt.graphics.plot_acf(data, ax=acf_ax, lags=lags)
    smt.graphics.plot_pacf(data, ax=pacf_ax, lags=lags)
    sm.qqplot(data, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(data, sparams=(data.mean(), data.std()), plot=prob_ax)

    plt.tight_layout()
    if show:
        plt.show()

    return


# 1. White noise
# normally distributed samples
n_samples = 1000
data = np.random.normal(size=1000)

tsplot(data, lags=20, show=False)


# 2. Random walk
# x[t] = x[t-1] + w[t]

x = w = np.random.normal(size=1000)

# transform the data for random walk
for t in range(len(x)):
    x[t] = x[t-1] + w[t]

tsplot(x, lags=20, show=False)

# 3. Difference the random walk data
# to check its stationary behaviour

x_new = np.diff(x)

tsplot(x_new, lags=20, show=True)
