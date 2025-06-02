"""Continuous Calibration Run Script"""

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.signal import savgol_filter
from scipy.optimize import fsolve
from continuous_calibration.fitting import lof, breakpoint, regression, smooth
from continuous_calibration.prep import export, get_prep, store_obj, volume


# Generate calibration
def gen(df, spec_name=None, t_col=0, col=1, mol0=0, vol0=None, add_sol_conc=[], add_cont_rate=[], t_cont=[],
        add_one_shot=[], t_one_shot=[], sub_cont_rate=0, path_length=None, fit_eq='linear', intercept=False,
        fit_lim=None, lof_test='Runs', lof_method=None, p_thresh=0.05, smooth_eq='concave', sg_win=1,
        breakpoint_lim=0, diffusion_delay=0, zero=False, win=1, inc=1,
        time_unit='time_unit', conc_unit='moles_unit volume_unit$^{-1}$', intensity_unit='AU',
        path_length_unit='length_unit'):
    """
    Function to fit Continuous Calibration data.

    Parameters
    ----------
    All data must use consistent units throughout, i.e., all parameters must be inputted with identical units.

    DATA
    df : numpy.array or pandas.DataFrame, required
        The calibration data, including time and signal intensity from monitored species.

    SPECIES INFORMATION
    spec_name : str, list of str, optional
        Name of each species.
        Default is None (species are given default names).

    DATA LOCATION
    t_col : int, or str, optional
        Index or name of time column.
        Default is 0 (first column).
    col : int, str, or list of int and str, optional
        Index(es) or name(s) of species column(s) (in spec_name order if used).
        Default is 1 (second column).

    REACTION CONDITIONS
    mol0 : float, or list of float, optional
        Initial moles of species (in spec_name/col order) in monitored solution in moles_unit.
        Default is None (mol0 will be fitted for all species).
    vol0 : float, required
        Initial monitored solution volume in volume_unit.

    SOLUTION ADDITIONS AND SUBTRACTIONS
    add_sol_conc : list of float, optional
        Concentration of solution being added for each species (in spec_name/col order) in moles_unit volume_unit^-1.
        Default is None (no addition solution for any species).
    add_cont_rate : list of tuple of float, or tuple of float, or float, optional
        Continuous addition rates of species (in spec_name/col order) in volume_unit time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with t_cont.
        Default is None (no continuous addition for any species).
    t_cont : list of tuple of float, or tuple of float, or float, optional
        Times at which continuous addition began for each species (in spec_name/col order) in time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with add_cont_rate.
        Default is None (no continuous addition for any species).
    add_one_shot : list of tuple of float, or tuple of float, or float, optional
        One shot additions for each species (in spec_name/col order) in volume_unit.
        Multiple conditions for each species are input as tuples.
        Is paired with t_one_shot.
        Default is None (no one-shot additions for any species).
    t_one_shot : list of tuple of float, or tuple of float, or float, optional
        Times at which one shot additions occurred for each species (in spec_name/col order) in time_unit^-1.
        Multiple conditions for each species are input as tuples.
        Is paired with add_one_shot.
        Default is None (no additions for any species).
    sub_cont_rate : float, optional
        Continuous subraction rate in volume_unit time_unit^-1.
        Default is 0.0.
    path_length : float, optional
        Path length for calculating molar extinction coefficient in length_unit.
        Default is None (not calculated).

    FITTING PARAMETERS
    fit_eq : str, optional
        Equation type used for fitting. Available options are 'linear', 'exponential', or 'custom'.
        A 'custom' rate equation can be formatted in the gen_eqs.py file.
        Default is 'linear'.
    intercept : bool, optional
        Apply intercept to fitting.
        Default is False.
    fit_lim : num, optional
        Upper concentration lof for fitting equation in concentration_unit.
        Default is None.
    lof_test : str, optional
        Algorithm to determine statistically valid fitting region.
        Available algorithms are Runs, Rainbow, Harvey-Collier or Shapiro-Wilk.
        Default is None.
    lof_method : str, optional
        Method to determine statistically valid fitting region.
        Available methods are 'max' (value which maximizes test score), 'first' (value before test fails first time),
        and 'last' (last value for which test passes)
        Default is None (linearity test not applied).
    p_thresh : float, optional
        Threshold of p value above which fitting is determined to be statistically justified.
        Default is 0.05.
    smooth_eq : str, optional
        Method for smoothing. Available methods are 'monotonic', 'concave', and 'Savtisky-Golay'
    sg_win : int, optional
        Number of data points to use for Savitzky-Golay smoothing.
        Default is 1, i.e. no Savitzky-Golay smoothing.

    DATA MANIPULATION
    breakpoint_lim : list of tuple of float, or tuple of float, or float, optional
        Additional time in which t_cont may have occurred in time_units. t_cont are fit to a maximum of breakpoint_lim
        Default is 0.0.
    diffusion_delay : float, optional
        Delay for diffusion for t_one_shot in time_units.
        Default is 0.0.
    zero : bool, optional
        Set intercept to zero (no noise).
        Default is False.
    win : int, optional
        Smoothing window.
        Default is 1 (no smoothing).
    inc : int, or float, optional
        Increments between adjacent points for improved simulation (>1)
        or fraction of points removed, e.g. if unnecessarily large amount of data (<1).
        Default is 1 (no addition or subtraction of points).

    UNITS
    time_unit : str, optional
        Time units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'time_unit'.
    conc_unit : str, optional
        Concentration units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'moles_unit volume_unit$^{-1}$'.
    intensity_unit : str, optional
        Intensity units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'AU'.
    path_length_unit : str, optional
        Path length units. Optional but must be consistent between all parameters - only used in output and plotting.
        Default is 'length_unit'.

    Returns
    -------
    res : obj, with the following fields defined:
        DATA
        conc_df : pandas.DataFrame
            Processed concentration data in moles_unit volume_unit^-1.
        intensity_df : pandas.DataFrame
            Processed intensity data in intensity_unit.
        sg_smooth_df : pandas.DataFrame
            Savistky-Golay smoothed data in intensity_unit.
        fit_df : pandas.DataFrame
            Fitted concentration data in moles_unit volume_unit^-1.
        fit_resid_df : pandas.DataFrame
            Fitted residuals of concentration data in moles_unit volume_unit^-1.
        all_df : pandas.DataFrame
            The above combined.

        FITTED PARAMETERS
        params : list of list of float
            Fitted parameter(s) (in spec_name/col order) in appropriate units
        param_err : list of list of float
            Fitted parameter error(s) (in spec_name/col order) in appropriate units
        lof_idx : list of float
            Fitted lof of linearity (in spec_name/col order) in moles_unit volume_unit^-1.
        mec : list of float
            Fitted molar extinction coefficient(s) (in spec_name/col order) in moles_unit^-1 volume_unit length_unit^-1.

        GOODNESS OF FIT
        rss : float
            Residual sum of squares.
        r2 : float
            R squared.
        rmse : float
            Root mean square average.
        mae : float
            Mean average error.
        aic : float
            Akaike information criterion (AIC).
        bic : float
            Bayesian information criterion (BIC).

        FUNCTIONS
        apply(self, df, spec_name=None, col=0, t_col=None, calib=None, sg_win=1, win=1, inc=1) : function
            Function to directly apply calibration to experimental data.
            See cc apply documentation for more details.
        plot_intensity_vs_time(self, f_format='svg', save_to='cc_intensity_vs_time.svg',
                               return_fig=False, return_image=False, transparent=False) : function
             Function to plot raw continuous calibration data.
             See cc plot_intensity_vs_time documentation for more details.
        plot_intensity_vs_conc(self, conc_unit='', f_format='svg', save_to='cc_intensity_vs_conc.svg',
                               plot_resid=False, return_fig=False, return_image=False, transparent=False) : function
             Function to plot calibration results.
             See cc plot_intensity_vs_conc documentation for more details.
        plot_lof_test(self, conc_unit='', f_format='svg', save_to='cc_lof_test.svg',
                       return_fig=False, return_image=False, transparent=False) : function
             Function to plot lof of linteraity test results.
             See cc plot_lof_test documentation for more details.
    """

    (data_arr, spec_name, species_alt, num_spec, t_col, col, mol0, mol0_temp,
     add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, fit_lim) = (
        get_prep.process_gen_input(df, spec_name, t_col, col, mol0,
                                   add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, fit_lim))

    # Remove diffusion delayed data points
    data_arr = get_prep.remove_diffusion_delay(data_arr, t_col, t_one_shot, diffusion_delay)

    # Define t and corresponding intensities
    t = get_prep.data_smooth(data_arr, t_col, win, inc)
    intensity = np.empty((len(t), num_spec))
    for i in range(num_spec):
        intensity[:, i] = get_prep.data_smooth(data_arr, col[i], win, inc)

    # Create object to store data
    data = store_obj.GenData(spec_name, num_spec, mol0, t_one_shot, t, intensity, fit_eq, intercept, lof_method,
                             p_thresh, breakpoint_lim, diffusion_delay, time_unit, conc_unit, intensity_unit,
                             path_length_unit, sg_win, win, inc)

    # Estimate breakpoints
    if breakpoint_lim:
        t_cont = breakpoint.get_breakpoints(t, intensity, t_cont, guesses=t_cont, bounds=breakpoint_lim)
        data.est_t_cont = t_cont

    def raw_data_process(data, mol0):
        # Convert time and conc_events into conc
        data.conc, data.mol, data.vol = volume.get_conc_events(t, num_spec, vol0, mol0, add_sol_conc, add_cont_rate,
                                                               t_cont, add_one_shot, t_one_shot, sub_cont_rate)

        # Store time data
        data.raw_df = pd.DataFrame(np.concatenate([t.reshape(-1, 1), intensity, data.conc], axis=1),
                                   columns=['Time / ' + time_unit] +
                                           [i + 'Intensity / ' + intensity_unit for i in species_alt] +
                                           [i + 'Conc. / ' + conc_unit for i in species_alt])

        # Average avg_intensity values of equal concentrations
        data.avg_conc, data.avg_intensity, data.error = get_prep.avg_repeats(data.conc, intensity, zero=zero)

        return data

    data = raw_data_process(data, mol0_temp)

    if not t_one_shot or (isinstance(t_one_shot, (list, tuple)) and len(t_one_shot) == 1 and not t_one_shot[0]):
        data.error = None

    # Apply fitting
    if fit_eq:
        limit_idx = []
        for spec in range(num_spec):
            if fit_lim is not None and fit_lim[spec] is not None:
                limit_idx.append(np.where(data.avg_conc[:, spec] <= fit_lim[spec])[0][-1])
            else:
                limit_idx.append(data.avg_conc[:, spec].shape[0])
        data.limit_idx = limit_idx

        data.fit, data.resid, data.params, data.param_err, data.mec, data.lof_idx, data.lof, data.indices, data.test = \
            lof.get_lof(data.avg_conc, data.avg_intensity, num_spec, fit_eq=fit_eq, intercept=intercept,
                        limit_idx=data.limit_idx, limit_test=lof_test, limit_method=lof_method,
                        threshold=p_thresh, path_length=path_length)
        data.rss, data.rmse, data.mae, data.r2, data.r2_adj, data.aic, data.bic = (
            regression.residuals(data.avg_intensity, data.resid, len(data.params), intercept=intercept,
                                 limit_idx=data.lof_idx))

    # Apply smoothing
    data = smooth.smooth(data, smooth_eq=smooth_eq, intercept=intercept, sg_win=sg_win)

    # Fit standard addition experiments
    if None in mol0:
        model = get_prep.sort_fit_eq(fit_eq, intercept, fit_type='apply')
        for i in range(num_spec):
            if mol0[i] is None:
                mol0_temp[i] = -model(0, *list(data.params[i].values())) * vol0
            data.mol0_fit = mol0_temp
        data = raw_data_process(data, mol0_temp)
        if fit_eq and lof_test and lof_method and data.lof is not None:
            data.lof = [data.lof[spec] + (data.mol0_fit[spec] / vol0) for spec in range(num_spec)]

    # Make DataFrames
    data.conc_df = pd.DataFrame(data.avg_conc, columns=[spec + 'Conc. / ' + conc_unit for spec in species_alt])
    data.intensity_df = pd.DataFrame(data.avg_intensity,
                                     columns=[spec + 'Intensity / ' + intensity_unit for spec in species_alt])
    data.all_df = pd.concat([data.conc_df, data.intensity_df], axis=1)
    if data.error is not None:
        data.error_df = pd.DataFrame(data.error, columns=[spec + 'Intensity Error / ' + intensity_unit
                                                          for spec in species_alt])
        data.all_df = pd.concat([data.all_df, data.error_df], axis=1)
    if data.smooth_intensity is not None:
        data.smooth_intensity_df = pd.DataFrame(data.smooth_intensity,
                                                columns=[spec + 'Smoothed Intensity / ' + intensity_unit
                                                            for spec in species_alt])
        data.all_df = pd.concat([data.all_df, data.smooth_intensity_df], axis=1)
    if data.fit is not None:
        data.fit_df = pd.DataFrame(data.fit,
                                   columns=[spec + 'Fit Intensity / ' + intensity_unit for spec in species_alt])
        data.all_df = pd.concat([data.all_df, data.fit_df], axis=1)
    if data.resid is not None:
        data.resid_df = pd.DataFrame(data.resid,
                                     columns=[spec + 'Fit Resid. / ' + intensity_unit for spec in species_alt])
        data.all_df = pd.concat([data.all_df, data.resid_df], axis=1)


    return data


if __name__ == "__main__":
    df = pd.DataFrame(columns=['Time', 'Intensity'])

    time = np.linspace(0, 100, 101)
    df['Time'] = time

    t_cont = 10
    A = 100
    k = 1
    vol0 = 100
    add_sol_conc = 1
    add_cont_rate = 1

    conc = ((time - t_cont) * add_cont_rate * add_sol_conc) / (vol0 + (time - t_cont) * add_cont_rate)
    conc = np.where(conc < 0, 0, conc)
    noise_frac = 0.01
    df['Intensity'] = np.array(A * (1 - np.exp(-k * conc)) + A * np.random.uniform(-noise_frac, noise_frac, size=time.shape))

    fit_eq = 'lin'
    intercept = False
    lof_test = 'R2'
    lof_method = 'max'
    smooth_eq = 'concave'
    sg_win = 1

    data = gen(df, vol0=vol0, add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont, fit_eq=fit_eq,
               intercept=intercept, lof_test=lof_test, lof_method=lof_method, smooth_eq=smooth_eq, sg_win=sg_win)

    data.plot_intensity_vs_conc(plot_resid=True, f_format='png', save_to='gen_test_intensity_vs_conc.png')
    if 'lin' in fit_eq.lower() and lof_test:
        print(data.lof)
        print(data.mec)
        data.plot_lof_test(f_format='png', save_to='gen_test_lof_test.png')

    print(data.params)
    # apply_data = data.apply([10, 20, 30])
    apply_data = data.apply(np.array([[10, 10], [20, 20], [30, 30], [35, 35], [36, 35.1], [37, 35.05], [38, 35.07], [39, 35.11], [40, 35.09]]), col=1, t_col=0)
    print(apply_data.fit_df)
    if hasattr(apply_data, 't'):
        apply_data.plot_conc_vs_time(f_format='png', save_to='apply_test_conc_vs_time.png')
