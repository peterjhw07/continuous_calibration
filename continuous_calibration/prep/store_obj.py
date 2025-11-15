"""CC Object Creation"""

import pandas as pd
from continuous_calibration.apply import apply
from continuous_calibration.plot import plot_conc_vs_time, plot_intensity_vs_time, plot_intensity_vs_conc, plot_lof_test


# Generated data object
class GenData:
    def __init__(self, spec_name, num_spec, mol0, t_disc_add, t, intensity, fit_eq, intercept, lof_method,
                 p_thresh, lod_stds, breakpoint_lim, diffusion_delay, time_unit, conc_unit, intensity_unit, path_length_unit,
                 sg_win, win, inc):
        self.spec_name = spec_name
        self.num_spec = num_spec
        self.mol0 = mol0
        self.t_disc_add = t_disc_add
        self.t = t
        self.intensity = intensity
        self.fit_eq = fit_eq
        self.intercept = intercept
        self.lof_method = lof_method
        self.p_thresh = p_thresh
        self.lod_stds = lod_stds
        self.breakpoint_lim = breakpoint_lim
        self.diffusion_delay = diffusion_delay
        self.time_unit = time_unit
        self.conc_unit = conc_unit
        self.intensity_unit = intensity_unit
        self.path_length_unit = path_length_unit

        self.sg_win = sg_win
        self.win = win
        self.inc = inc
        self.est_t_cont = None

        self.avg_conc, self.avg_intensity, self.std = None, None, None

        self.smooth_intensity, self.smooth_model, self.smooth_intercept = None, None, None
        self.fit, self.resid, self.params, self.param_err = None, None, None, None
        self.mac, self.mol0_fit, self.lof_idx, self.lof, self.indices, self.test = None, None, None, None, None, None
        self.rss, self.rmse, self.mae, self.r2, self.r2_adj, self.aic, self.bic = None, None, None, None, None, None, None
        self.lod_idx, self.lod = None, None

        self.conc_df, self.intensity_df, self.std_df = None, None, None
        self.smooth_intensity_df, self.fit_df, self.resid_df, self.all_df = None, None, None, None

    # Object link for simple plotting
    def apply(self, df, col=0, t_col=None, calib='', calib_win=1, sg_win=1, win=1, inc=1):
        if self.smooth_intensity is not None and 'data' not in calib.lower():
            intensity = self.smooth_intensity_df
        else:
            intensity = self.intensity_df
        calib_df = pd.concat([self.conc_df, intensity], axis=1)
        conc_col = list(range(self.num_spec))
        intensity_col = list(range(self.num_spec, 2 * self.num_spec))
        if 'data' in calib.lower() or 'smooth' in calib.lower():
            fit_eq = None
        elif 'fit' in calib.lower():
            calib_df = None
            fit_eq = self.fit_eq
        else:
            fit_eq = self.fit_eq

        apply_data = apply(df, spec_name=self.spec_name, col=col, t_col=t_col, calib_df=calib_df, conc_col=conc_col,
                           intensity_col=intensity_col, fit_eq=fit_eq, params=self.params, calib_win=calib_win,
                           sg_win=sg_win, win=win, inc=inc, conc_unit=self.conc_unit,
                           intensity_unit=self.intensity_unit, time_unit=self.time_unit)
        return apply_data

    # Object links for simple plotting
    def plot_intensity_vs_time(self, f_format='svg', save_to='cc_intensity_vs_time.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_intensity_vs_time(self.t, self.intensity, diffusion_delay=self.diffusion_delay,
                                               time_unit=self.time_unit, intensity_unit=self.intensity_unit,
                                               f_format=f_format, save_to=save_to, return_fig=return_fig,
                                               return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_intensity_vs_conc(self, conc_unit='', f_format='svg', save_to='cc_intensity_vs_conc.svg',
                               plot_resid=False, return_fig=False, return_img=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        if plot_resid:
            resid = self.resid
        else:
            resid = None
        img, mimetype = plot_intensity_vs_conc(self.avg_conc, self.avg_intensity,
                                               smooth_intensity=self.smooth_intensity, intensity_error=self.std,
                                               upper_lim=self.lof_idx, fit_line=self.fit, resid=resid,
                                               conc_unit=conc_unit, intensity_unit=self.intensity_unit,
                                               f_format=f_format, save_to=save_to, return_fig=return_fig,
                                               return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_lof_test(self, conc_unit='', f_format='svg', save_to='cc_lof_test.svg',
                      return_fig=False, return_img=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        img, mimetype = plot_lof_test(self.avg_conc[self.indices, :], self.test, self.p_thresh, conc_unit=conc_unit,
                                      f_format=f_format, save_to=save_to,
                                      return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype


# Applied data object
class ApplyData:
    def __init__(self, spec_name, num_spec, intensity, conc_unit, intensity_unit):
        self.spec_name = spec_name
        self.num_spec = num_spec
        self.intensity = intensity
        self.conc_unit = conc_unit
        self.intensity_unit = intensity_unit

    # Add fitted data
    def add_fit(self, intensity_df, fit_df, all_df):
        self.intensity_df = intensity_df
        self.fit_df = fit_df
        self.all_df = all_df

    # Object links for simple plotting
    def plot_intensity_vs_time(self, f_format='svg', save_to='applied_intensity_vs_time.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_intensity_vs_time(self.t, self.intensity, diffusion_delay=self.diffusion_delay,
                                               time_unit=self.time_unit, intensity_unit=self.intensity_unit,
                                               f_format=f_format, save_to=save_to, return_fig=return_fig,
                                               return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_conc_vs_time(self, f_format='svg', save_to='applied_intensity_vs_time.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_conc_vs_time(self.t, self.fit, time_unit=self.time_unit, conc_unit=self.conc_unit,
                                          f_format=f_format, save_to=save_to,
                                          return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype
