import pandas as pd
from continuous_calibration.apply import apply
from continuous_calibration.plot import plot_conc_vs_time, plot_intensity_vs_time, plot_intensity_vs_conc, plot_lol_tests


# Generated data object
class GenData:
    def __init__(self, spec_name, num_spec, mol0, t_one_shot, t, intensity, fit_eq, intercept, lol_method,
                 p_thresh, breakpoint_lim, diffusion_delay, time_unit, conc_unit, intensity_unit, path_length_unit):
        self.spec_name = spec_name
        self.num_spec = num_spec
        self.mol0 = mol0
        self.t_one_shot = t_one_shot
        self.t = t
        self.intensity = intensity
        self.fit_eq = fit_eq
        self.intercept = intercept
        self.lol_method = lol_method
        self.p_thresh = p_thresh
        self.breakpoint_lim = breakpoint_lim
        self.diffusion_delay = diffusion_delay
        self.time_unit = time_unit
        self.conc_unit = conc_unit
        self.intensity_unit = intensity_unit
        self.path_length_unit = path_length_unit

    # Add fitted data
    def add_fit(self, t_df, fit_conc_df, fit_rate_df, temp_df, all_df):
        self.t_df = t_df
        self.fit_conc_df = fit_conc_df
        self.fit_rate_df = fit_rate_df
        self.temp_df = temp_df
        self.all_df = all_df

    # Object link for simple plotting
    def apply(self, df, col=0, t_col=None, calib=None, sg_win=1, win=1, inc=1):
        if calib and "data" in calib.lower():
            calib_df = self.intensity_df
        elif calib and ("sg" in calib.lower() or "sav" in calib.lower()):
            calib_df = self.sg_smooth_intensity_df
        elif self.fit_eq is None:
            if hasattr(self, "sg_smooth_intensity_df"):
                calib_df = self.sg_smooth_intensity_df
            else:
                calib_df = self.intensity_df
        else:
            calib_df = None
        if calib_df is not None:
            conc_col = list(range(len(calib_df.shape[0])))
            intensity_col = list(range(len(conc_col), 2 * len(conc_col)))
            calib_df = pd.concat([self.conc_df, calib_df], axis=1)
        else:
            conc_col, intensity_col = None, None

        apply_data = apply(df, spec_name=self.spec_name, col=col, t_col=t_col, calib_df=calib_df, conc_col=conc_col,
                           intensity_col=intensity_col, fit_eq=self.fit_eq, params=self.params, sg_win=sg_win, win=win,
                           inc=inc, conc_unit="moles_unit volume_unit$^{-1}$", intensity_unit="AU",
                           time_unit="time_unit")
        return apply_data

    # Object link for simple plotting
    def plot_intensity_vs_time(self, f_format='svg', save_to='cc_intensity_vs_time.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_intensity_vs_time(self.t, self.intensity, t_one_shot=self.t_one_shot,
                                               diffusion_delay=self.diffusion_delay, time_unit=self.time_unit,
                                               intensity_unit=self.intensity_unit, f_format=f_format, save_to=save_to,
                                               return_fig=return_fig, return_img=return_img,
                                               transparent=transparent)
        return img, mimetype

    def plot_intensity_vs_conc(self, conc_unit="", f_format='svg', save_to='cc_intensity_vs_conc.svg',
                               plot_resid=False, return_fig=False, return_img=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        if plot_resid:
            resid = self.resid
        else:
            resid = None
        img, mimetype = plot_intensity_vs_conc(self.avg_conc, self.avg_intensity, smooth_intensity=self.sg_smooth,
                                               intensity_error=self.error, limit=self.lol_idx, fit_line=self.fit,
                                               resid=resid, conc_unit=conc_unit, intensity_unit=self.intensity_unit,
                                               f_format=f_format, save_to=save_to, return_fig=return_fig,
                                               return_img=return_img, transparent=transparent)
        return img, mimetype

    def plot_lol_tests(self, conc_unit="", f_format='svg', save_to='cc_lol_tests.svg',
                       return_fig=False, return_img=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        img, mimetype = plot_lol_tests(self.avg_conc, self.lol_tests_df, self.p_thresh,
                                       conc_unit=conc_unit,
                                       intensity_unit=self.intensity_unit, f_format=f_format, save_to=save_to,
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

    # Object link for simple plotting
    def plot_intensity_vs_time(self, f_format='svg', save_to='applied_intensity_vs_time.svg',
                               return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_intensity_vs_time(self.t, self.intensity, t_one_shot=self.t_one_shot,
                                               diffusion_delay=self.diffusion_delay, time_unit=self.time_unit,
                                               intensity_unit=self.intensity_unit, f_format=f_format, save_to=save_to,
                                               return_fig=return_fig, return_img=return_img,
                                               transparent=transparent)
        return img, mimetype

    def plot_conc_vs_time(self, f_format='svg', save_to='applied_intensity_vs_time.svg',
                          return_fig=False, return_img=False, transparent=False):
        img, mimetype = plot_conc_vs_time(self.t, self.fit, time_unit=self.time_unit, conc_unit=self.conc_unit,
                                          f_format=f_format, save_to=save_to,
                                          return_fig=return_fig, return_img=return_img, transparent=transparent)
        return img, mimetype
