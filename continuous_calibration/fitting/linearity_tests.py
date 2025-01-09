"""CC Linearity Tests"""

import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.stats import shapiro
import statistics as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.regression.linear_model as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import kstest_normal


def harvey_collier_test(model):
    statistic, p_value = sms.linear_harvey_collier(model, skip=2)
    return p_value


def rainbow_test(model):
    statistic, p_value = sms.linear_rainbow(model)
    return p_value


def runs_test(model):
    statistic, p_value = runstest_1samp(model.resid)
    return p_value


def breusch_pagan_lagrange(x, y, upper_lim):
    model = smf.OLS(y[:upper_lim + 1], x[:upper_lim + 1]).fit()
    statistic, p_value, _, _ = het_breuschpagan(model.resid, sm.add_constant(model.model.exog))
    return p_value


def kolmogorov_smirnov_test(residuals):
    statistic, p_value = kstest_normal(residuals)
    return p_value


def shapiro_wilk_test(residuals):
    statistic, p_value = shapiro(residuals - np.mean(residuals))
    return p_value


def residuals_start_middle_end(conc, intensity, lower_lim, upper_lim, fit_line):
    print(stats.mean(intensity[lower_lim:lower_lim+10, 0] - fit_line[lower_lim:lower_lim+10, 0]),
          stats.mean(intensity[int((upper_lim-lower_lim)/2-5):int((upper_lim-lower_lim)/2+5), 0] -
                          fit_line[int((upper_lim-lower_lim)/2-5):int((upper_lim-lower_lim)/2+5), 0]),
          stats.mean(intensity[upper_lim-10:upper_lim, 0] - fit_line[upper_lim-10:upper_lim, 0]))


def p_value_pass(p_value, threshold=0.05):
    if p_value >= threshold:
        test_pass = True
    else:
        test_pass = False
    return test_pass
