"""CC Limit of Fitting Calculation"""

import inspect
import numpy as np
from continuous_calibration.fitting import lof_tests, regression
from continuous_calibration.prep import get_prep


# Calculate limit of fitting
def get_lof(conc, intensity, num_spec, fit_eq='lin', intercept=False, limit_idx=None,
            limit_test='Runs', limit_method='auto', threshold=0.05, path_length=None):
    if limit_idx is None:
        limit_idx = [conc.shape[0]] * conc.shape[1]
    if not limit_test or not limit_method:
        lof_idx, lof = limit_idx, None
        fit_lines, fit_lines_resid, coeff, coeff_err = regression.fit_intensity_curve(conc, intensity, fit_eq,
                                                                                      intercept=intercept, limit_idx=limit_idx)
        indices, test_res = None, None
    else:
        if 'run' in limit_test.lower():
            lof_test = lof_tests.lof_map.get('Runs')
            if 'auto' in limit_method.lower(): limit_method = 'last'
        elif 'rain' in limit_test.lower() and 'lin' in fit_eq.lower():
            lof_test = lof_tests.lof_map.get('Rainbow')
            if 'auto' in limit_method.lower(): limit_method = 'last'
        elif ('harvey' in limit_test.lower() or 'hc' in limit_test.lower()) and 'lin' in fit_eq.lower():
            lof_test = lof_tests.lof_map.get('Harvey-Collier')
            if 'auto' in limit_method.lower(): limit_method = 'last'
        elif 'shap' in limit_test.lower() or 'sw' in limit_test.lower():
            lof_test = lof_tests.lof_map.get('Shapiro-Wilk')
            if 'auto' in limit_method.lower(): limit_method = 'last'
        elif 'rmse' in limit_test.lower() or 'mae' in limit_test.lower():
            if 'auto' in limit_method.lower(): limit_method = 'min'
            if 'rmse' in limit_test.lower():
                metric_index = 0
            else:
                metric_index = 1
        elif 'r2' in limit_test.lower():
            if 'auto' in limit_method.lower(): limit_method = 'max'
            metric_index = 2

        lof_idx, lof = [], []
        lower_lim = len(inspect.signature(get_prep.sort_fit_eq(fit_eq, intercept, fit_type='gen')).parameters) + 3
        for i in range(num_spec):
            indices = list(range(lower_lim, limit_idx[i]))
            test_res = np.full(len(indices), np.nan)
            if 'lin' in fit_eq.lower() and not any(metric in limit_test.lower() for metric in ['rmse', 'mae', 'r2']):
                for idx in indices:
                    try:
                        fit_res, _ = regression.lin_regress(conc[:, i], intensity[:, i], idx, intercept=intercept)
                        test_res[idx - lower_lim] = lof_test(fit_res)
                    except:
                        pass
            else:
                if any(metric in limit_test.lower() for metric in ['rmse', 'mae', 'r2']):
                    for idx in indices:
                        try:
                            _, fit_lines_resid, _, _ = regression.fit_intensity_curve(conc[:, i].reshape(-1, 1),
                                                                                      intensity[:, i].reshape(-1, 1), fit_eq,
                                                                                      intercept=intercept, limit_idx=[idx])
                            rss, rmse, mae, r2, r2_adj, aic, bic = regression.residuals(intensity[:, i].reshape(-1, 1),
                                                                        fit_lines_resid, lower_lim, limit_idx=[idx])
                            test_res[idx - lower_lim] = [rmse, mae, r2][metric_index][0]
                        except:
                            pass
                else:
                    for idx in indices:
                        try:
                            _, fit_lines_resid, _, _ = regression.fit_intensity_curve(conc[:, i].reshape(-1, 1),
                                                                                      intensity[:, i].reshape(-1, 1), fit_eq,
                                                                                      intercept=intercept, limit_idx=[idx])
                            test_res[idx - lower_lim] = lof_test(Model(fit_lines_resid[:idx, :]))
                        except:
                            pass

            lof_idx.append(try_except_test(indices, test_res, limit_method, threshold))
            lof.append(conc[lof_idx[i], i])

        fit_lines, fit_lines_resid, coeff, coeff_err = regression.fit_intensity_curve(conc, intensity, fit_eq,
                                                                                      intercept=intercept, limit_idx=lof_idx)
    if 'lin' in fit_eq.lower() and path_length:
        mac = [coe['a'] / path_length for coe in coeff]
    else:
        mac = None

    return fit_lines, fit_lines_resid, coeff, coeff_err, mac, lof_idx, lof, indices, test_res


# Determine limit of fitting from appropriate test
def try_except_test(indices, test, criterion, threshold=0.05):
    test = np.nan_to_num(test, nan=1.0)
    try:
        if isinstance(criterion, (int, float)):
            return int(criterion)
        elif 'min' in criterion.lower():
            return max(np.array(indices)[test == min(test)])
        elif 'max' in criterion.lower():
            return max(np.array(indices)[test == max(test)])
        elif 'first' in criterion.lower():
            try:
                return max(0, min(np.array(indices)[test < threshold]) - 1)
            except:
                return max(indices)
        elif 'last' in criterion.lower():
            return max(np.array(indices)[test >= threshold])
        else:
            return max(np.array(indices)[test >= threshold])
    except:
        return 1


class Model:
    def __init__(self, resid):
        self.resid = resid
