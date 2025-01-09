from continuous_calibration.plot import plot_intensity_vs_time, plot_intensity_vs_conc, plot_lol_tests


# Sim data object
class Data:
    def __init__(self, spec_name, num_spec, mol0, t_one_shot, diffusion_delay, t, intensity, intercept, get_lol,
                 p_thresh, time_unit, conc_unit, intensity_unit, path_length_unit):
        self.spec_name = spec_name
        self.num_spec = num_spec
        self.mol0 = mol0
        self.t_one_shot = t_one_shot
        self.diffusion_delay = diffusion_delay
        self.t = t
        self.intensity = intensity
        self.intercept = intercept
        self.get_lol = get_lol
        self.p_thresh = p_thresh
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
    def plot_intensity_vs_time(self, f_format='svg', save_to='cc_intensity_vs_time.svg',
                               return_fig=False, return_image=False, transparent=False):
        plot_intensity_vs_time(self.t, self.intensity, t_one_shot=self.t_one_shot,
                               diffusion_delay=self.diffusion_delay, time_unit=self.time_unit,
                               intensity_unit=self.intensity_unit, f_format=f_format, save_to=save_to,
                               return_fig=return_fig, return_image=return_image, transparent=transparent)

    def plot_intensity_vs_conc(self, conc_unit="", f_format='svg', save_to='cc_intensity_vs_conc.svg',
                               plot_resid=False, return_fig=False, return_image=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        if plot_resid:
            fit_resid = self.fit_resid
        else:
            fit_resid = None
        plot_intensity_vs_conc(self.avg_conc, self.avg_intensity, smooth_intensity=self.sg_smooth_intensity,
                               intensity_error=self.error, limit=self.lol, fit_line=self.fit_lines,
                               fit_resid=fit_resid, conc_unit=conc_unit, intensity_unit=self.intensity_unit,
                               f_format=f_format, save_to=save_to,
                               return_fig=return_fig, return_image=return_image, transparent=transparent)

    def plot_lol_tests(self, conc_unit="", f_format='svg', save_to='cc_lol_tests.svg',
                       return_fig=False, return_image=False, transparent=False):
        if not conc_unit:
            conc_unit = self.conc_unit
        plot_lol_tests(self.avg_conc, self.lol_tests_df, self.p_thresh,
                       conc_unit=conc_unit,
                       intensity_unit=self.intensity_unit, f_format=f_format, save_to=save_to,
                       return_fig=return_fig, return_image=return_image, transparent=transparent)