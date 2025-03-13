"""CC Linearity Tests"""

from statsmodels.sandbox.stats.runs import runstest_1samp
import statistics as stats
import statsmodels.stats.api as sms


def rainbow_test(model):
    statistic, p_value = sms.linear_rainbow(model)
    return p_value


def runs_test(model):
    statistic, p_value = runstest_1samp(model.resid)
    return p_value


def harvey_collier_test(model):
    statistic, p_value = sms.linear_harvey_collier(model, skip=2)
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
