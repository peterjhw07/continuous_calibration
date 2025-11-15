"""CC Equation Fitting"""

import inspect
import math
import numpy as np
from scipy.optimize import curve_fit
import sklearn.metrics
import statsmodels.api as sm

from continuous_calibration.prep import get_prep


# Perform linear regression
def lin_regress(x, y, upper_lim, intercept=False):
    # Perform regression
    if intercept:
        fit_res = sm.OLS(y[:upper_lim], sm.add_constant(x[:upper_lim])).fit()
        pred = fit_res.predict(sm.add_constant(x[:upper_lim]))
    else:
        fit_res = sm.OLS(y[:upper_lim], x[:upper_lim]).fit()
        pred = fit_res.predict(x[:upper_lim])

    # Get fit metrics
    fit_res.mse = sklearn.metrics.mean_squared_error(y[:upper_lim], pred[:upper_lim])
    fit_res.rmse = math.sqrt(fit_res.mse)
    fit_res.mae = np.mean(np.abs(fit_res.resid))

    return fit_res, pred.reshape(-1, 1)


def fit_intensity_curve(conc, exp_intensity, fit_eq, intercept=False, limit_idx=None, p0=None, weight_factor=0):
    # Define model function
    model = get_prep.sort_fit_eq(fit_eq, intercept, fit_type='gen')

    if not p0:
        if intercept:
            p0 = [1.0] * (len(inspect.signature(model).parameters) - 2)
            p0.append(0.0)
        else:
            p0 = [1.0] * (len(inspect.signature(model).parameters) - 1)

    # Perform regression
    popt, perr, coeff, coeff_err = [], [], [], []
    fit_lines, fit_lines_resid = np.full(conc.shape, None, dtype=object), np.full(conc.shape, None, dtype=object)
    if limit_idx is None:
        limit_idx = [conc.shape[0]] * conc.shape[1]
    for i in range(conc.shape[1]):
        n = limit_idx[i] + 1

        if weight_factor > 0:
            sigma = (exp_intensity[:n, i] + 1) ** weight_factor
            absolute_sigma = True
            popt_it, pcov_it = curve_fit(model, conc[:n, i], exp_intensity[:n, i], p0=p0, sigma=sigma,
                                         absolute_sigma=absolute_sigma, maxfev=10000)
        else:
            popt_it, pcov_it = curve_fit(model, conc[:n, i], exp_intensity[:n, i], p0=p0, maxfev=10000)

        # Calculate the standard errors (square root of the diagonal elements of the covariance matrix)
        perr_it = np.sqrt(np.diag(pcov_it))
        popt.append(popt_it)
        perr.append(perr_it)
        coeff.append(dict(zip(list(inspect.signature(model).parameters.keys())[1:], popt_it)))
        coeff_err.append(dict(zip(list(inspect.signature(model).parameters.keys())[1:], perr_it)))

        # Get prediction
        fit_lines[:, i] = model(conc[:, i], *popt_it)
        fit_lines_resid[:, i] = exp_intensity[:, i] - fit_lines[:, i]

    return fit_lines, fit_lines_resid, coeff, coeff_err


# Calculate residuals
def residuals(data, res, num_param, limit_idx=None):
    if not isinstance(num_param, (list, tuple)):
        num_param = [num_param] * data.shape[1]
    if limit_idx is None:
        limit_idx = [data.shape[0]] * data.shape[1]
    rss, rmse, mae, r2, r2_adj, aic, bic = [], [], [], [], [], [], []
    for i in range(data.shape[1]):
        n = limit_idx[i]
        k = num_param[i]
        rss.append(np.sum(res[:n, i] ** 2))
        rmse.append(math.sqrt((rss[i] / n)))
        mae.append(np.sum(abs(res[:n, i])) / n)
        r2.append(1 - (rss[i] / np.sum((data[:n, i] - np.mean(data[:n, i])) ** 2)))
        r2_adj.append(1 - (((1 - r2[i]) * (n - 1)) / (n - k - 1)))
        aic.append(n * np.log(rss[i] / n) + 2 * k)
        bic.append(n * np.log(rss[i] / n) + np.log(n) * k)
    return rss, rmse, mae, r2, r2_adj, aic, bic
