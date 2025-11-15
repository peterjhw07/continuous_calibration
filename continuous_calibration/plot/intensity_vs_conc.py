"""CC Intensity vs. Concentration Plotting"""

import matplotlib.pyplot as plt
import numpy as np
from continuous_calibration.plot.plot_func import units_adjust, plot_process, calc_mono_lim, calc_multi_lim


# Plot intensity vs. concentration
def plot_intensity_vs_conc(conc, intensity, smooth_intensity=None, intensity_error=None, upper_lim=None, fit_line=None,
                           resid=None, xlim=None, conc_unit='moles_unit volume_unit^-1',
                           intensity_unit='AU', f_format='svg', save_to='', return_fig=False, return_img=False,
                           transparent=False, font_size=12):

    num_spec = conc.shape[1]
    conc_unit, intensity_unit = units_adjust([conc_unit, intensity_unit])
    if resid is not None:
        fig, axes = plt.subplots(nrows=2, ncols=num_spec, figsize=(num_spec * 6, 5),
                                 gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_spec, figsize=(num_spec * 6, 5))

    for col in range(num_spec):
        y_data = []
        if num_spec > 1 and resid is not None:
            ax = axes[0, col]
        elif num_spec > 1 and not resid is not None:
            ax = axes[col]
        elif resid is not None:
            ax = axes[0]
        else:
            ax = axes
        ax.scatter(conc, intensity[:, col], 8, 'k', label='Data')
        y_data.append(intensity[:, col])
        if (intensity_error is not None and intensity_error[:, col] is not None and
                np.count_nonzero(intensity_error[:, col]) > 0.2 * intensity_error[:, col].size):
            ax.errorbar(conc, intensity[:, col], yerr=intensity_error[:, col],
                              fmt='none', ecolor='k', capsize=5, capthick=1, elinewidth=1)
            y_data.append(intensity[:, col] + intensity_error[:, col])
        if smooth_intensity is not None and smooth_intensity[:, col] is not None:
            ax.plot(conc, smooth_intensity[:, col], 'g', label='Smoothed Data')
            y_data.append(smooth_intensity[:, col])
        if fit_line is not None and upper_lim is not None:
            ax.plot(conc[:upper_lim[col] + 1, col], fit_line[:upper_lim[col] + 1, col], 'r', label='Fit')
            y_data.append(fit_line[:upper_lim[col] + 1, col])
            try:
                ax.axvline(x=conc[upper_lim, col], color='b', linestyle='--', label='Limit of Fitting')
            except:
                pass
        elif fit_line is not None:
            ax.plot(conc[:, col], fit_line[:, col], 'r', label='Fit')
            y_data.append(fit_line[:, col])
        if not xlim:
            ax.set_xlim(calc_mono_lim(conc[:, col], edge_adj=0))
            ax.set_ylim(calc_multi_lim(y_data))
        else:
            ax.set_xlim(xlim, edge_adj=0)
        ax.set_xlabel('Concentration' + conc_unit, fontsize=font_size)
        ax.set_ylabel('Intensity' + intensity_unit, fontsize=font_size)
        if resid is not None:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)
        ax.legend(loc='lower right', fontsize=font_size, frameon=False)

        if resid is not None:
            y_data = []
            if num_spec > 1 and resid is not None:
                ax = axes[1, col]
            else:
                ax = axes[1]
            if upper_lim:
                ax.scatter(conc[:upper_lim[col] + 1, col], resid[:upper_lim[col] + 1, col], 8, 'k', label='Residuals')
                y_data.append(resid[:upper_lim[col] + 1, col])
                try:
                    ax.axvline(x=conc[upper_lim[col], col], color='b', linestyle='--', label='Limit of Fitting')
                except:
                    pass
            else:
                ax.scatter(conc[:, col], resid[:, col], 8, 'k', label='Residuals')
                y_data.append(resid[:, col])
            #if intensity_error is not None:
            #    ax.errorbar(conc, intensity.flatten().tolist(), yerr=intensity_error.flatten().tolist(),
            #                fmt='none', ecolor='k', capsize=5, capthick=1, elinewidth=1)
            if not xlim:
                ax.set_xlim(calc_mono_lim(conc[:, col], edge_adj=0))
                ax.set_ylim(calc_multi_lim(y_data))
            else:
                ax.set_xlim(xlim)
            ax.set_xlabel('Concentration' + conc_unit, fontsize=font_size)
            ax.set_ylabel('Intensity' + intensity_unit, fontsize=font_size)
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            # ax_lower.legend(loc='lower right', fontsize=font_size, frameon=False)

    plt.subplots_adjust(hspace=0)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
