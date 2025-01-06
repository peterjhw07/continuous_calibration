import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sklearn.metrics
import statsmodels.api as sm
from continuous_calibration import prep


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


def fit_intensity_curve(conc, exp_intensity, initial_guess=[1.0, 1.0, 0.0]):
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
    def intensity_model(conc, A, k, conc0):
        return A * (1 - np.exp(-k * conc)) + conc0

    # Perform the curve fitting
    popt, pcov = curve_fit(intensity_model, conc, exp_intensity, p0=initial_guess)

    # Calculate the standard errors (square root of the diagonal elements of the covariance matrix)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr, pcov


