import numpy as np
from scipy.stats import linregress
import sklearn.metrics
import statsmodels.api as sm
from continuous_calibration import prep


# Perform linear regression
def lin_regress(x, y, upper_lim, intercept=True):
    if intercept:
        fit_res = sm.OLS(y[:upper_lim], sm.add_constant(x[:upper_lim])).fit()
        pred = fit_res.predict(sm.add_constant(x))  # making predictions
    else:
        fit_res = sm.OLS(y[:upper_lim], x[:upper_lim]).fit()
        pred = fit_res.predict(x)  # making predictions

    # Evaluating the fit_res
    fit_res.mse = sklearn.metrics.mean_squared_error(y[:upper_lim], pred[:upper_lim])
    fit_res.rmse = sklearn.metrics.root_mean_squared_error(y[:upper_lim], pred[:upper_lim])
    fit_res.mae = np.mean(np.abs(fit_res.resid))

    return fit_res, pred.reshape(-1, 1)

