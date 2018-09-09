
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as scs


def tsplot(data, lags, show=True):

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    with plt.style.context('bmh'):

        layout = (3, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        prob_ax = plt.subplot2grid(layout, (2, 1))

        data.plot(ax=ts_ax)
        ts_ax.set_title('Time plot for the data')
        smt.graphics.plot_acf(data, ax=acf_ax, lags=lags)
        smt.graphics.plot_pacf(data, ax=pacf_ax, lags=lags)
        sm.qqplot(data, ax=qq_ax, line='s')
        scs.probplot(data, sparams=(data.mean(), data.std()), plot=prob_ax)

        plt.tight_layout()
        if show:
            plt.show()
    return


# adfuller method works for 1d data so, the data should
# either be a series or a 1d numpy array

# write a funtion to check the stationarity of the data.
def stationary_check(data):
    fuller_results = smt.stattools.adfuller(data)

    print('Fuller Statistics : {}'.format(fuller_results[0]))
    print('Fuller test P-value : {}'.format(fuller_results[1]))
    print('#lags used : {}'.format(fuller_results[2]))
    print('#observation used: {}'.format(fuller_results[3]))

    for key, value in fuller_results[4].items():
        print('Significance Level and value: {0}, : {1}:'.format(key, value))

    return
