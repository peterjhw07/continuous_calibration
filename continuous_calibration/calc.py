import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import statistics
from continuous_calibration import linearity_tests, plotter, prep, regression, smooth, volume


def run(df, t_col, col, vol0, mol0, add_sol_conc, add_cont_rate, t_cont,
        add_one_shot, t_one_shot, sub_cont_rate, upper_fit_lim='max', intercept=False, win=1, inc=1):

    if isinstance(col, (int, float)): col = (col,)
    num_spec = len(col)

    data_org = df.to_numpy()
    t = prep.data_smooth(data_org, t_col, win, inc)

    org_intensity = np.empty((len(t), num_spec))
    for i in range(num_spec):
        if col[i] is not None: org_intensity[:, i] = prep.data_smooth(data_org, col[i], win, inc)

    # Convert time and conc_events into conc
    conc, mol, vol = volume.get_conc_events(t, num_spec, vol0, mol0, add_sol_conc, add_cont_rate, t_cont,
                           add_one_shot, t_one_shot, sub_cont_rate)

    # Average intensity values of equal concentrations
    conc, intensity = prep.avg_repeats(conc, org_intensity)
    indices = np.arange(conc.shape[0]).reshape(-1, 1)

    # Smooth data
    try:
        smooth_intensity = savgol_filter(intensity[:, 0], int(len(conc) / 5), 1)
        new_y_value = 5
        interp_func = interp1d(smooth_intensity, conc[:, 0], kind='linear')
        x_estimate = interp_func(new_y_value)
        print(x_estimate)
    except:
        smooth_intensity = None
    # intensity = smooth_intensity.reshape(-1, 1)

    # Regression
    limit = len(conc)

    # Perform optimization to find the number of data points that maximizes R^2
    fit_df = pd.DataFrame(index=range(limit), columns=["Limit", "RMSE", "MAE", "Runs Test",
                                                       "Rainbow Test", "Harvey-Collier Test"])
    fit_df["Limit"] = range(limit)
    for limit in range(5, limit):
        fit_res, fit_line = regression.lin_regress(conc[:, 0], intensity[:, 0], limit, intercept=intercept)
        fit_df.at[limit, "RMSE"] = fit_res.rmse
        fit_df.at[limit, "MAE"] = fit_res.mae
        fit_df.at[limit, "Runs Test"] = linearity_tests.runs_test(fit_res.resid)
        fit_df.at[limit, "Rainbow Test"] = linearity_tests.rainbow_test(conc[:, 0], intensity[:, 0], limit)
        fit_df.at[limit, "Harvey-Collier Test"] = linearity_tests.harvey_collier_test(conc[:, 0], intensity[:, 0], limit)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(fit_df)

    threshold = 0.1

    res_res, res_line = regression.lin_regress(conc[:, 0], fit_res.resid, int(limit), intercept=intercept)

    runs_test_lol = max(indices[fit_df["Runs Test"] >= threshold])[0]
    rainbow_test_lol = max(indices[fit_df["Rainbow Test"] >= threshold])[0]
    hc_test_lol = max(indices[fit_df["Harvey-Collier Test"] >= threshold])[0]
    min_lol = min(runs_test_lol, rainbow_test_lol, hc_test_lol)
    print(runs_test_lol, rainbow_test_lol, hc_test_lol)

    fit_res, fit_line = regression.lin_regress(conc[:, 0], intensity[:, 0], min_lol, intercept=intercept)

    conc_df = pd.DataFrame(conc)
    intensity_df = pd.DataFrame(intensity)

    plotter.test_plot(fit_df["Limit"], fit_df[["RMSE", "MAE", "Runs Test", "Rainbow Test", "Harvey-Collier Test"]])
    # plotter.org_plot(t / 60, org_intensity)
    plotter.cc_plot(conc, intensity, smooth_intensity, min_lol, fit_line, res_line)
    # plotter.cc_plot(conc, intensity, smooth_conc, smooth_intensity, limit, fit_line, res_line)
    plotter.plot_conc_vs_time(conc_df, intensity_df)



