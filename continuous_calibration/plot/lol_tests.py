"""CC Limit of Linearity Test Plotting"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from continuous_calibration.fitting import smooth
from continuous_calibration.plot.plot_func import plot_process


def plot_lol_tests(conc, lol_test_df, threshold, conc_unit="moles_unit volume_unit^-1",
                   intensity_unit="AU", f_format='svg',
                   save_to='', return_fig=False, return_image=False, transparent=False, font_size=12):
    fig, ax = plt.subplots(1, 1)

    sg_filter_val = int(len(conc) / 20)
    for header in ['Runs Test', 'Rainbow Test', 'Harvey-Collier Test']:
        # sg = smooth.savgol_filter(lol_test_df[header], sg_filter_val, 2)
        # ax.plot(conc[sg_filter_val:], sg[sg_filter_val:] / sg[sg_filter_val:].max(), label=header)
        ax.plot(conc, np.array(lol_test_df[header]), label=header)
        # ax.plot(conc, np.array(lol_test_df[header] / lol_test_df[header].max()), label=header)
    ax.axhline(y=threshold, color='b', linestyle='--', label='Limit of Linearity')
    ax.set_xlim([min(conc), max(conc)])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Concentration / ' + conc_unit, fontsize=font_size)
    ax.set_ylabel('p value', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(fontsize=font_size, frameon=False)

    # plt.subplots_adjust(hspace=0)

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
