"""CC Limit of Fitting Tests"""

from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.stats.api as sms


# Runs test
def runs_test(model):
    statistic, p_value = runstest_1samp(model.resid)
    return p_value


# Rainbow test
def rainbow_test(model):
    statistic, p_value = sms.linear_rainbow(model)
    return p_value


# Harvey-Collier test
def harvey_collier_test(model):
    statistic, p_value = sms.linear_harvey_collier(model, skip=2)
    return p_value


# Shapiro-Wilk test
def shapiro_wilk_test(model):
    statistic, p_value = stats.shapiro(model.resid)
    return p_value


# Determine if test passes
def p_value_pass(p_value, threshold=0.05):
    if p_value >= threshold:
        test_pass = True
    else:
        test_pass = False
    return test_pass


# Map to get required test
lof_map = {
    'Runs': runs_test,
    'Rainbow': rainbow_test,
    'Harvey-Collier': harvey_collier_test,
    'Shapiro-Wilk': shapiro_wilk_test
}
