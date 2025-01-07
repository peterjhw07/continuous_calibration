import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from continuous_calibration.fitting import lol, breakpoint, regression
from continuous_calibration.prep import export, get_prep, store_obj, volume


def run(df, spec_name=None, t_col=0, col=1, mol0=None, vol0=None, add_sol_conc=[], add_cont_rate=[], t_cont=[],
        add_one_shot=[], t_one_shot=[], sub_cont_rate=0, diffusion_delay=0,
        fit_eq="linear", intercept=False, get_lol=None, p_thresh=0.05, path_length=None, win=1, inc=1, sg_win=11,
        time_unit="time_unit", conc_unit="moles_unit volume_unit$^{-1}$", intensity_unit="AU",
        path_length_unit="path_length_unit"):

    spec_name, num_spec, col, mol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot = \
        get_prep.process_input(spec_name, col, mol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot)

    if len(spec_name) > 1:
        species_alt = [spec + ' ' for spec in spec_name]
    else:
        species_alt = ['']

    # Extract
    data_arr = df.to_numpy()

    # Remove diffusion delayed data points
    data_arr = get_prep.remove_diffusion_delay(data_arr, t_col, t_one_shot, diffusion_delay)

    # Define t and corresponding intensities
    t = get_prep.data_smooth(data_arr, t_col, win, inc)
    intensity = np.empty((len(t), num_spec))
    for i in range(num_spec):
        if col[i] is not None: intensity[:, i] = get_prep.data_smooth(data_arr, col[i], win, inc)

    # Create object to store data
    data = store_obj.Data(spec_name, num_spec, mol0, t_one_shot, diffusion_delay, t, intensity, intercept, get_lol,
                          p_thresh, time_unit, conc_unit, intensity_unit, path_length_unit)

    # Convert time and conc_events into conc
    data.conc, data.mol, data.vol = volume.get_conc_events(t, num_spec, vol0, mol0, add_sol_conc, add_cont_rate, t_cont,
                                                           add_one_shot, t_one_shot, sub_cont_rate)
    # breakpoint.get_breakpoints(t, intensity[:, 0])

    # Store time data
    data.raw_df = pd.DataFrame(np.concatenate([t.reshape(-1, 1), intensity, data.conc], axis=1),
                               columns=['Time / ' + time_unit] +
                                       [i + 'Intensity / ' + intensity_unit for i in species_alt] +
                                       [i + 'Conc. / ' + conc_unit for i in species_alt])

    # Average avg_intensity values of equal concentrations
    data.avg_conc, data.avg_intensity, data.error = get_prep.avg_repeats(data.conc, intensity, zero=True)
    if not t_one_shot:
        data.error = None

    # Smooth data
    try:
        data.sg_smooth_intensity = savgol_filter(data.avg_intensity[:, i], sg_win, 1)
        lowess_frac = 0.1
        lowess_smooth_intensity = lowess(data.avg_intensity[:, i], data.avg_conc[:, i], frac=lowess_frac)[:, 1]
    except:
        data.sg_smooth_intensity = None

    try:
        new_y_value = 5
        interp_func = interp1d(data.sg_smooth_intensity, data.avg_conc[:, 0], kind='linear')
        x_estimate = interp_func(new_y_value)
        # print(x_estimate)
    except:
        pass

    if "lin" in fit_eq and get_lol:
        data.lol_tests_df, data.lol_df, data.fit_lines, data.fit_resid, data.gradient, data.intercept, data.mec = \
            lol.get_lol(data.avg_conc, data.avg_intensity, spec_name, intercept=intercept, path_length=path_length,
                        threshold=p_thresh, intensity_unit=intensity_unit, conc_unit=conc_unit,
                        path_length_unit=path_length_unit)
        data.lol = data.lol_df["LOL"].to_list()
    elif "exp" in fit_eq:
        a = regression.fit_intensity_curve(data.avg_conc, data.avg_intensity)
    else:
        data.fit_lines, data.fit_resid, data.lol, data.gradient, data.intercept, data.mec = None, None, None, None, \
                                                                                            None, None

    data.proc_df = pd.DataFrame(np.concatenate([data.avg_conc, data.avg_intensity], axis=1),
                                columns=[spec + "Conc. / " + conc_unit for spec in species_alt] +
                                        [spec + "Intensity / " + intensity_unit for spec in species_alt])
    if data.sg_smooth_intensity is not None:
        data.proc_df = pd.concat([data.proc_df, pd.DataFrame(data.sg_smooth_intensity,
                            columns=[spec + "Smoothed Intensity / " + intensity_unit for spec in species_alt])], axis=1)

    if get_lol:
        data.proc_df = pd.concat([data.proc_df, pd.DataFrame(np.concatenate([data.fit_lines, data.fit_resid], axis=1),
                            columns=[spec + "Fit Conc. / " + conc_unit for spec in species_alt] +
                                    [spec + "Fit Resid. / " + conc_unit for spec in species_alt])], axis=1)

    return data
