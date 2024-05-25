import numpy as np
import scipy.optimize as optimize
from scipy.signal import savgol_filter
from continuous_calibration import prep
from continuous_calibration import regression


def savgol_smooth(x, y, window_size="len/5", poly_order=1):
    if isinstance(window_size, str):
        window_size = int(len(x) / 5)
    return x[:, 0], savgol_filter(y[:, 0], window_size, poly_order)


def convol_smooth(conc, intensity, lower_lim, upper_lim, fit_line):
    smooth_width = 1
    x1 = np.linspace(-3, 3, smooth_width)
    norm = np.sum(np.exp(-x1 ** 2)) * (x1[1] - x1[0])  # ad hoc normalization
    y1 = (4 * x1 ** 2 - 2) * np.exp(-x1 ** 2) / smooth_width * 8  # norm*(x1[1]-x1[0])

    y_conv = np.convolve(intensity[:, 0], y1, mode="same")

    print(intensity[lower_lim:upper_lim, 0] - fit_line[lower_lim:upper_lim, 0])

    res = np.zeros((len(conc[:, 0]), 2))
    for i in range(2, len(conc[:, 0])):
        gradient, intercept, r_value, p_value, std_err = regression.lin_regress(intensity[:i, 0], conc[:i, 0])
        fit_line = gradient * conc + intercept
        rss, aic, r2 = prep.fit_goodness_metrics(intensity[:i, 0], fit_line[:i, 0], [0])
        res = np.std(intensity[:i, 0] - fit_line[:i, 0])
        res[i, :] = [i, r2]
    res_sort = res[res[:, 1].argsort()]
    fit = res_sort[-2, 0]
    upper_lim = int(fit)
    return


def equation_smooth(x, y):
   popt, _ = optimize.curve_fit(equation, x, y, p0=[1, 1, 1], bounds=[(-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)], maxfev=100000)
   fit_line = equation(x, *popt)
   return popt, fit_line


def equation(x, m, n, z):
    return m * (x ** z) + n * x
