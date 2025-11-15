"""CC Application Equations"""

import numpy as np


# Custom fit equation
def fit_eq_custom(intensity):
    print('Fit to a custom rate equation')
    return


# Linear fit equation
def fit_eq_lin(intensity, a):
    return intensity / a


# Linear intercept fit equation
def fit_eq_lin_int(intensity, a, c):
    return (intensity - c) / a


# Logarithmic fit equation
def fit_eq_log(intensity, a, b):
    return (np.exp(intensity / a) - 1) / b


# Logarithmic intercept fit equation
def fit_eq_log_int(intensity, a, b, c):
    return (np.exp((intensity - c) / a) - 1) / b


# Exponential fit equation
def fit_eq_exp(intensity, a, b):
    return - np.log(1 - (intensity / a)) / b


# Exponential intercept fit equation
def fit_eq_exp_int(intensity, a, b, c):
    return - np.log(1 - ((intensity - c) / a)) / b


# Tangential fit equation
def fit_eq_tan(intensity, a, b):
    return np.tan(intensity / a) / b


# Tangential intercept fit equation
def fit_eq_tan_int(intensity, a, b, c):
    return np.tan((intensity - c) / a) / b


# Michaelis-Menten fit equation
def fit_eq_mm(intensity, a, b):
    return (b * intensity) / (a - intensity)


# Michaelis-Menten intercept fit equation
def fit_eq_mm_int(intensity, a, b, c):
    return (b * (intensity - c)) / (a + c - intensity)


# Langmuir fit equation
def fit_eq_langmuir(intensity, a, b):
    return intensity / (a - b * intensity)


# Langmuir intercept fit equation
def fit_eq_langmuir_int(intensity, a, b, c):
    return (intensity - c) / (a - b * intensity + b * c)


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