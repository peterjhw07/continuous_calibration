import inspect
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sklearn.metrics
import statsmodels.api as sm
from continuous_calibration.fitting import fit_eqs


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
    fit_res.rmse = sklearn.metrics.root_mean_squared_error(y[:upper_lim], pred[:upper_lim])
    fit_res.mae = np.mean(np.abs(fit_res.resid))

    return fit_res, pred.reshape(-1, 1)


def fit_intensity_curve(conc, exp_intensity, fit_eq, intercept=False, p0=None):
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
    if "exp" in fit_eq.lower():
        p0 = [1.0, 1.0]
        if intercept:
            model = fit_eqs.fit_eq_map.get("Exponential_intercept")
            p0.append(0.0)
        else:
            model = fit_eqs.fit_eq_map.get("Exponential")
    elif "custom" in fit_eq.lower() and not p0:
        model = fit_eqs.fit_eq_map.get("Custom")
        p0 = [1.0] * len(inspect.signature(model).parameters)
    else:
        try:
            model = fit_eqs.fit_eq_map.get(fit_eq)
        except:
            print("Non-existent model name.")

    # Perform the curve fitting
    popt, pcov, perr = [], [], []
    fit_lines, fit_lines_resid = np.empty(shape=conc.shape), np.empty(shape=conc.shape)
    for i in range(conc.shape[1]):
        popt_it, pcov_it = curve_fit(model, conc[:, i], exp_intensity[:, i], p0=p0)
        # Calculate the standard errors (square root of the diagonal elements of the covariance matrix)
        perr_it = np.sqrt(np.diag(pcov_it))
        popt.append(popt_it)
        pcov.append(pcov_it)
        perr.append(perr_it)
        fit_lines[:, i] = model(conc[:, i], *popt_it)
        fit_lines_resid[:, i] = exp_intensity[:, i] - fit_lines[:, i]

    FIX IT SO THAT ALL FIRST PARAMETERS ARE TOGETHER IN LIST, THEN SECOND, ETC. FOR EACH SPECIES
    coeff = dict(zip(list(inspect.signature(model).parameters.keys())[1:], popt))
    coeff_err = dict(zip(list(inspect.signature(model).parameters.keys())[1:], perr))

    return fit_lines, fit_lines_resid, coeff, coeff_err


