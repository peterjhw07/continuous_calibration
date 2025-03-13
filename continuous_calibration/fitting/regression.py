import inspect
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sklearn.metrics
import statsmodels.api as sm
from continuous_calibration.fitting import gen_eqs


# Perform linear regression
def lin_regress(x, y, pre_upper_lim, intercept=False):
    upper_lim = pre_upper_lim + 1
    if intercept:
        fit_res = sm.OLS(y[:upper_lim], sm.add_constant(x[:upper_lim])).fit()
        pred = fit_res.predict(sm.add_constant(x[:upper_lim]))  # making predictions
    else:
        fit_res = sm.OLS(y[:upper_lim], x[:upper_lim]).fit()
        pred = fit_res.predict(x[:upper_lim])  # making predictions

    # Evaluating the fit_res
    fit_res.mse = sklearn.metrics.mean_squared_error(y[:upper_lim], pred[:upper_lim])
    fit_res.rmse = math.sqrt(fit_res.mse)
    fit_res.mae = np.mean(np.abs(fit_res.resid))

    return fit_res, pred.reshape(-1, 1)


def fit_intensity_curve(conc, exp_intensity, fit_eq, intercept=False, lol_idx=None, p0=None):
    """
    Fits the curve intensity = A * (1 - exp(-k * conc)) + conc0 to the given data
    and calculates the errors of the fitted parameters.

    Parameters:
        conc (array-like): Array of concentration values (x).
        exp_intensity (array-like): Array of intensity values (y).

    Returns:
        popt (tuple): Optimized parameters (A, k, conc0).
        perr (tuple): Standard errors of the fitted parameters (A_err, k_err, conc0_err).
        pcov (ndarray): Covariance matrix of the fitted parameters.
    """

    # Define the model function
    if "lin" in fit_eq.lower():
        if intercept:
            model = gen_eqs.fit_eq_map.get("Linear_intercept")
        else:
            model = gen_eqs.fit_eq_map.get("Linear")
    elif "exp" in fit_eq.lower():
        if intercept:
            model = gen_eqs.fit_eq_map.get("Exponential_intercept")
        else:
            model = gen_eqs.fit_eq_map.get("Exponential")
    elif "custom" in fit_eq.lower():
        model = gen_eqs.fit_eq_map.get("Custom")
    else:
        try:
            model = gen_eqs.fit_eq_map.get(fit_eq)
        except:
            print("Non-existent model name.")

    if not p0:
        if intercept:
            p0 = [1.0] * (len(inspect.signature(model).parameters) - 2)
            p0.append(0.0)
        else:
            p0 = [1.0] * (len(inspect.signature(model).parameters) - 1)


    # Perform the curve fitting
    popt, perr, coeff, coeff_err = [], [], [], []
    fit_lines, fit_lines_resid = np.full(conc.shape, None, dtype=object), np.full(conc.shape, None, dtype=object)
    if lol_idx is None:
        lol_idx = [conc.shape[0]] * conc.shape[1]
    for i in range(conc.shape[1]):
        n = lol_idx[i]
        popt_it, pcov_it = curve_fit(model, conc[:n, i],
                                     exp_intensity[:n, i], p0=p0, maxfev=5000)
        # Calculate the standard errors (square root of the diagonal elements of the covariance matrix)
        perr_it = np.sqrt(np.diag(pcov_it))
        popt.append(popt_it)
        perr.append(perr_it)
        coeff.append(dict(zip(list(inspect.signature(model).parameters.keys())[1:], popt_it)))
        coeff_err.append(dict(zip(list(inspect.signature(model).parameters.keys())[1:], perr_it)))
        fit_lines[:n, i] = model(conc[:n, i], *popt_it)
        fit_lines_resid[:n, i] = exp_intensity[:n, i] - fit_lines[:n, i]

    return fit_lines, fit_lines_resid, coeff, coeff_err


# Calculate residuals
def residuals(data, res, coeff, lol_idx=None):
    k = len(coeff)
    if lol_idx is None:
        lol_idx = [data.shape[0]] * data.shape[1]
    rss, rmse, mae, r2, r2_adj, aic, bic = [], [], [], [], [], [], []
    for i in range(data.shape[1]):
        n = lol_idx[i]
        rss.append(np.sum(res[:n, i] ** 2))
        rmse.append(math.sqrt((rss[i] / n)))
        mae.append(np.sum(abs(res[:n, i]) / n))
        r2.append(1 - (rss[i] / np.sum((data[:n, i] - np.mean(data[:n, i])) ** 2)))
        r2_adj.append(1 - (((1 - r2[i]) * (n - 1)) / (n - k - 1)))
        aic.append(n * np.log(rss[i] / n) + 2 * k)
        bic.append(n * np.log(rss[i] / n) + np.log(n) * k)
    return rss, rmse, mae, r2, r2_adj, aic, bic
