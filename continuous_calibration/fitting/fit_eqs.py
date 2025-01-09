"""CC Fit Equations"""

import inspect
import math
import numpy as np


# Custom fit equation
def fit_eq_custom(conc, A, k):
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


# Map to get required fit equation
fit_eq_map = {
    "Linear": fit_eq_lin,
    "Linear_intercept": fit_eq_lin_int,
    "Exponential": fit_eq_exp,
    "Exponential_intercept": fit_eq_exp_int,
    "custom": fit_eq_custom,
}