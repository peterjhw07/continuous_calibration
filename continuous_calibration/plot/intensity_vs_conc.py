"""CC Intensity vs. Concentration Plotting"""

import matplotlib.pyplot as plt
import base64
from continuous_calibration.plot.plot_func import plot_process


def plot_intensity_vs_conc(conc, intensity, smooth_intensity=None, intensity_error=None, limit=None, fit_line=None,
                           fit_resid=None, xlim=None, conc_unit="moles_unit volume_unit^-1",
                           intensity_unit="AU", f_format='svg', save_to='', return_fig=False,
                           return_image=False, transparent=False, font_size=12):

    num_spec = conc.shape[1]
    if fit_resid is not None:
        fig, axes = plt.subplots(nrows=2, ncols=num_spec, figsize=(num_spec * 6, 5),
                                 gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_spec, figsize=(num_spec * 6, 5))

    for col in range(num_spec):
        if num_spec > 1 and fit_resid is not None:
            ax = axes[0, col]
        elif num_spec > 1 and not fit_resid is not None:
            ax = axes[col]
        elif fit_resid is not None:
            ax = axes[0]
        else:
            ax = axes
        ax.scatter(conc, intensity, 8, 'k', label='Data')
        if intensity_error is not None:
            ax.errorbar(conc, intensity.flatten().tolist(), yerr=intensity_error.flatten().tolist(),
                              fmt='none', ecolor='k', capsize=5, capthick=1, elinewidth=1)
        if smooth_intensity is not None:
            ax.plot(conc, smooth_intensity, 'g', label='Smoothed Data')
        if fit_line is not None and limit is not None:
            ax.plot(conc[:limit[col] + 1, col], fit_line[:limit[col] + 1, col], 'r', label='Linear Fit')
            try:
                ax.axvline(x=conc[limit, col], color='b', linestyle='--', label='Limit of Linearity')
            except:
                pass
        elif fit_line is not None:
            ax.plot(conc[:, col], fit_line[:, col], 'r', label='Fit')
        if not xlim:
            ax.set_xlim([min(conc[:, col]), max(conc[:, col])])
        else:
            ax.set_xlim(xlim)
        # ax_upper.set_ylim([-10, 50])
        ax.set_xlabel('Conc. / ' + conc_unit, fontsize=font_size)
        ax.set_ylabel('Intensity / ' + intensity_unit, fontsize=font_size)
        if fit_resid is not None:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)
        ax.legend(loc='lower right', fontsize=font_size, frameon=False)

        if fit_resid is not None:
            if num_spec > 1 and fit_resid is not None:
                ax = axes[1, col]
            else:
                ax = axes[1]
            if limit:
                ax.scatter(conc[:limit[col] + 1, col], fit_resid[:limit[col] + 1, col], 8, 'k', label='Residuals')
                try:
                    ax.axvline(x=conc[limit[col], col], color='b', linestyle='--', label='Limit of Linearity')
                except:
                    pass
            else:
                ax.scatter(conc[:, col], fit_resid[:, col], 8, 'k', label='Residuals')
            #if intensity_error is not None:
            #    ax.errorbar(conc, intensity.flatten().tolist(), yerr=intensity_error.flatten().tolist(),
            #                fmt='none', ecolor='k', capsize=5, capthick=1, elinewidth=1)
            if not xlim:
                ax.set_xlim([min(conc[:, col]), max(conc[:, col])])
            else:
                ax.set_xlim(xlim)
            ax.set_xlabel('Conc. / ' + conc_unit, fontsize=font_size)
            ax.set_ylabel('Intensity / ' + intensity_unit, fontsize=font_size)
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            # ax_lower.legend(loc='lower right', fontsize=font_size, frameon=False)

    plt.subplots_adjust(hspace=0)

    # Save figure and return as required
    img, mimetype = plot_process(f_format, save_to, transparent)
    if return_fig:
        return fig, fig.get_axes()
    elif return_image:
        plt.close()
        img.seek(0)
        return img, mimetype
    else:
        plt.close()
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
