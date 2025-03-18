"""CC Fit Equations"""

import numpy as np


# Custom fit equation
def fit_eq_custom(conc):
    print("Fit to a custom rate equation")
    return


# Linear fit equation
def fit_eq_lin(conc, m):
    return m * conc


# Linear intercept fit equation
def fit_eq_lin_int(conc, m, c):
    return m * conc + c


# Exponential fit equation
def fit_eq_exp(conc, A, k):
    return A * (1 - np.exp(-k * conc))


# Exponential intercept fit equation
def fit_eq_exp_int(conc, A, k, c):
    return A * (1 - np.exp(-k * conc)) + c


# Fourier fit equation
def fit_eq_fourier(conc, a1, b1, a2, b2, w):
    return a1 * (np.cos(conc * w) - 1) + b1 * np.sin(conc * w) + a2 * (np.cos(2 * conc * w) - 1) + b2 * np.sin(2 * conc * w)


# Fourier fit intercept equation
def fit_eq_fourier_int(conc, a1, b1, a2, b2, w, c):
    return a1 * (np.cos(conc * w) - 1) + b1 * np.sin(conc * w) + a2 * (np.cos(2 * conc * w) - 1) + b2 * np.sin(2 * conc * w) + c


# Map to get required fit equation
fit_eq_map = {
    "Linear": fit_eq_lin,
    "Linear_intercept": fit_eq_lin_int,
    "Exponential": fit_eq_exp,
    "Exponential_intercept": fit_eq_exp_int,
    "Fourier": fit_eq_fourier,
    "Fourier_intercept": fit_eq_fourier_int,
    "custom": fit_eq_custom,
}