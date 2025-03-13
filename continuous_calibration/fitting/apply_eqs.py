"""CC Fit Equations"""

import numpy as np


# Custom fit equation
def fit_eq_custom(intensity):
    print("Fit to a custom rate equation")
    return


# Linear fit equation
def fit_eq_lin(intensity, m):
    return intensity / m


# Linear intercept fit equation
def fit_eq_lin_int(intensity, m, c):
    return (intensity - c) / m


# Exponential fit equation
def fit_eq_exp(intensity, A, k):
    return - np.log(1 - (intensity / A)) / k


# Exponential intercept fit equation
def fit_eq_exp_int(intensity, A, k, c):
    return - np.log(1 - ((intensity - c) / A)) / k


# Map to get required fit equation
fit_eq_map = {
    "Linear": fit_eq_lin,
    "Linear_intercept": fit_eq_lin_int,
    "Exponential": fit_eq_exp,
    "Exponential_intercept": fit_eq_exp_int,
    "custom": fit_eq_custom,
}