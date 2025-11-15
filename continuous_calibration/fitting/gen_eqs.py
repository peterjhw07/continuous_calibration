"""CC Fit Equations"""

import numpy as np


# Custom fit equation
def fit_eq_custom(conc):
    print('Fit to a custom rate equation')
    return


# Linear fit equation
def fit_eq_lin(conc, a):
    return a * conc


# Linear intercept fit equation
def fit_eq_lin_int(conc, a, c):
    return a * conc + c


# Logarithmic fit equation
def fit_eq_log(conc, a, b):
    return a * np.log((b * conc) + 1)


# Logarithmic intercept fit equation
def fit_eq_log_int(conc, a, b, c):
    return a * np.log((b * conc) + 1) + c


# Exponential fit equation
def fit_eq_exp(conc, a, b):
    return a * (1 - np.exp(-b * conc))


# Exponential intercept fit equation
def fit_eq_exp_int(conc, a, b, c):
    return a * (1 - np.exp(-b * conc)) + c


# Tangential fit equation
def fit_eq_tan(conc, a, b):
    return a * np.arctan(b * conc)


# Tangential intercept fit equation
def fit_eq_tan_int(conc, a, b, c):
    return a * np.arctan(b * conc) + c


# Michaelis-Menten fit equation
def fit_eq_mm(conc, a, b):
    return (a * conc) / (b + conc)


# Michaelis-Menten intercept fit equation
def fit_eq_mm_int(conc, a, b, c):
    return ((a * conc) / (b + conc)) + c


# Langmuir fit equation
def fit_eq_langmuir(conc, a, b):
    return (a * conc) / (1 + b * conc)


# Langmuir intercept fit equation
def fit_eq_langmuir_int(conc, a, b, c):
    return ((a * conc) / (1 + b * conc)) + c


# Map to get required fit equation
fit_eq_map = {
    'Linear': fit_eq_lin,
    'Linear_intercept': fit_eq_lin_int,
    'Logarithmic': fit_eq_log,
    'Logarithmic_intercept': fit_eq_log_int,
    'Exponential': fit_eq_exp,
    'Exponential_intercept': fit_eq_exp_int,
    'Tangent': fit_eq_tan,
    'Tangent_intercept': fit_eq_tan_int,
    'Michaelis-Menten': fit_eq_mm,
    'Michaelis-Menten_intercept': fit_eq_mm_int,
    'Langmuir': fit_eq_langmuir,
    'Langmuir_intercept': fit_eq_langmuir_int,
    'custom': fit_eq_custom,
}