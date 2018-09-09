import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import scipy.stats as scs


def tsplot(data, lags):
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

    return


