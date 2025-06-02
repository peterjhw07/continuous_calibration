"""Continuous Calibration Run Script"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from continuous_calibration.fitting import apply_eqs as fit_eqs
from continuous_calibration.prep import export, get_prep, store_obj


# Apply calibration
def apply(df, spec_name=None, col=0, t_col=None, calib_df=None, conc_col=None, intensity_col=None,
          fit_eq=None, params=None, calib_win=1, sg_win=1, win=1, inc=1,
          conc_unit='moles_unit volume_unit$^{-1}$', intensity_unit='AU', time_unit='time_unit'):
    """
        Function to fit Continuous Calibration data.

        Parameters
        ----------
        All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

        DATA
        df : numpy.array or pandas.DataFrame, required
            Experimental data requiring calibration, including signal intensity from monitored species.

        SPECIES INFORMATION
        spec_name : str, list of str, or None, optional
            Name of each species.
            Default is None (species are given default names).

        DATA LOCATION
        col : int, str, or list of int and str, optional
            Index(es) or name(s) of species column(s) (in spec_name order if used).
            Default is 0 (first column).
        t_col : int, or str, optional
            Index or name of time column.
            Default is None (no time values).

        FITTING PARAMETERS
        calib_df : numpy.array or pandas.DataFrame, optional
            The calibration data, including time and signal intensity from monitored species.
            col : int, str, or list of int and str, optional
            Default is None (uses fit_eq and params).
        conc_col : int, str, or list of int and str, optional
            Index(es) or name(s) of species column(s) (in spec_name order if used).
            Default is 0 (first column).
        intensity_col : int, or str, optional
            Index or name of intensity column.
            Default is None (no time values).
        fit_eq : str, optional
            Equation type used for fitting. Available options are 'linear', 'exponential', or 'custom'.
            A 'custom' rate equation can be formatted in the gen_eqs.py file.
            Default is None.
        params : list, optional
            Parameters obtained from fitting to fit_eq.
            Default is None

        DATA MANIPULATION
        sg_win : int, optional
            Number of data points to use for Savitzky-Golay smoothing.
            Default is 1, i.e. no Savitzky-Golay smoothing.
        win : int, optional
            Smoothing window.
            Default is 1 (no smoothing).
        inc : int, or float, optional
            Increments between adjacent points for improved simulation (>1)
            or fraction of points removed, e.g. if unnecessarily large amount of data (<1).
            Default is 1 (no addition or subtraction of points).

        UNITS
        conc_unit : str, optional
            Concentration units. Optional but must be consistent between all parameters - only used in output and plotting.
            Default is 'moles_unit volume_unit$^{-1}$'.
        intensity_unit : str, optional
            Intensity units. Optional but must be consistent between all parameters - only used in output and plotting.
            Default is 'AU'.
        time_unit : str, optional
            Time units. Optional but must be consistent between all parameters - only used in output and plotting.
            Default is 'time_unit'.

        Returns
        -------
        res : obj, with the following fields defined:
            DATA
            raw_df : pandas.DataFrame
                Processed input data.
            fit_df : pandas.DataFrame
                Fitted concentration data in moles_unit volume_unit^-1.
            all_df : pandas.DataFrame
                The above combined.

            FUNCTIONS
            plot_intensity_vs_time(self, f_format='svg', save_to='cc_intensity_vs_time.svg',
                                   return_fig=False, return_image=False, transparent=False) : function
                 Function to plot raw experimental data.
                 See cc plot_intensity_vs_time documentation for more details.
            plot_conc_vs_time(self, f_format='svg', save_to='applied_intensity_vs_time.svg',
                          return_fig=False, return_image=False, transparent=False) : function
                 Function to plot calibrated experimental data.
                 See cc plot_intensity_vs_time documentation for more details.
        """

    data_arr, spec_name, species_alt, num_spec, t_col, col, params = get_prep.process_apply_input(df, spec_name,
                                                                                                  t_col, col, params)

    # Define intensities and corresponding t (if applicable)
    intensity = np.empty((data_arr.shape[0], num_spec))
    for i in range(num_spec):
        intensity[:, i] = get_prep.data_smooth(data_arr, col[i], win, inc)
        # Smooth data
        try:
            intensity[:, i] = savgol_filter(intensity[:, i], sg_win, 1)
        except:
            pass

    # Create object to store data
    data = store_obj.ApplyData(spec_name, num_spec, intensity, conc_unit, intensity_unit)

    if t_col is not None:
        t = get_prep.data_smooth(data_arr, t_col, win, inc)
        data.t = t
        data.time_unit = time_unit

    # Store time data
    if t_col:
        data.raw_df = pd.DataFrame(np.concatenate([t.reshape(-1, 1), intensity], axis=1),
                                   columns=['Time / ' + time_unit] +
                                           [i + 'Intensity / ' + intensity_unit for i in species_alt])
    else:
        data.raw_df = pd.DataFrame(intensity, columns=[i + 'Intensity / ' + intensity_unit for i in species_alt])

    if calib_df is not None:
        calib_arr, _, conc_col = get_prep.process_data_input(calib_df, num_spec, None, conc_col)
        calib_arr, _, intensity_col = get_prep.process_data_input(calib_df, num_spec, None, intensity_col)
        calib_conc = calib_arr[:, conc_col]
        calib_intensity = calib_arr[:, intensity_col]

        data_fit = np.zeros(intensity.shape)
        for spec in range(num_spec):
            for i in range(intensity.shape[0]):
                conc_all = calib_conc[:, spec][np.argsort(np.abs(calib_intensity[:, spec] - intensity[i, spec]))[:calib_win]]
                conc_best = conc_all[0]
                conc_above_below = min(calib_win, len(calib_conc[:, spec][calib_conc[:, spec] < conc_best]),
                                       len(calib_conc[:, spec][calib_conc[:, spec] > conc_best]))
                if conc_above_below != 0:
                    conc_allowed = np.concatenate(([conc_best], conc_all[conc_all < conc_best][:conc_above_below],
                                                   conc_all[conc_all > conc_best][:conc_above_below]))
                    data_fit[i, spec] = np.mean(conc_allowed)
                    conc_std = np.std(conc_allowed)
                else:
                    data_fit[i, spec] = conc_best
                    conc_std = 0

    if fit_eq:
        eq_fit = np.zeros(intensity.shape)
        if 'lin' in fit_eq.lower():
            if len(params[0]) == 2:
                model = fit_eqs.fit_eq_map.get('Linear_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Linear')
        elif 'log' in fit_eq.lower():
            if len(params[0]) == 3:
                model = fit_eqs.fit_eq_map.get('Logarithmic_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Logarithmic')
        elif 'exp' in fit_eq.lower():
            if len(params[0]) == 3:
                model = fit_eqs.fit_eq_map.get('Exponential_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Exponential')
        elif 'tan' in fit_eq.lower():
            if len(params[0]) == 3:
                model = fit_eqs.fit_eq_map.get('Tangent_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Tangent')
        elif 'mich' in fit_eq.lower() or 'mm' in fit_eq.lower():
            if len(params[0]) == 3:
                model = fit_eqs.fit_eq_map.get('Michaelis-Menten_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Michaelis-Menten')
        elif 'lang' in fit_eq.lower():
            if len(params[0]) == 3:
                model = fit_eqs.fit_eq_map.get('Langmuir_intercept')
            else:
                model = fit_eqs.fit_eq_map.get('Langmuir')
        elif 'custom' in fit_eq.lower():
            model = fit_eqs.fit_eq_map.get('Custom')
        else:
            try:
                model = fit_eqs.fit_eq_map.get(fit_eq)
            except:
                print('Non-existent model name.')
        if model is not None:
            for spec in range(num_spec):
                eq_fit[:, spec] = model(intensity[:, spec], *params[spec])

    if calib_df is not None:
        data.data_fit = data_fit
        data.data_fit_df = pd.DataFrame(data_fit, columns=[spec + 'Data Fit Concentration / ' + conc_unit for spec in species_alt])
        if fit_eq is None:
            data.fit = data_fit
            data.fit_df = pd.DataFrame(data_fit, columns=[spec + 'Fit Concentration / ' + conc_unit for spec in species_alt])
            data.all_df = pd.concat([data.raw_df, data.fit_df], axis=1)
    if fit_eq:
        data.eq_fit = eq_fit
        data.eq_fit_df = pd.DataFrame(eq_fit, columns=[spec + 'Equation Fit Concentration / ' + conc_unit for spec in species_alt])
        if calib_df is None:
            data.fit = eq_fit
            data.fit_df = pd.DataFrame(eq_fit, columns=[spec + 'Fit Concentration / ' + conc_unit for spec in species_alt])
            data.all_df = pd.concat([data.raw_df, data.fit_df], axis=1)
    if calib_df is not None and fit_eq:
        data.fit = np.concatenate([data_fit, eq_fit], axis=1)
        data.fit_df = pd.concat([data.data_fit_df, data.eq_fit_df], axis=1)
        data.all_df = pd.concat([data.raw_df, data.fit_df], axis=1)

    return data


if __name__ == "__main__":
    data = apply([3, 5, 5.5], fit_eq='Exponential', params=[5, 0.5, 1])
    print(data.all_df)
