"""CC Limit of Linearity Test Plotting"""

import matplotlib.pyplot as plt
import numpy as np
from continuous_calibration.plot.plot_func import plot_process, calc_mono_lim, calc_multi_lim


def plot_lof_test(conc, test, threshold, conc_unit='moles_unit volume_unit^-1',
                   intensity_unit='AU', f_format='svg',
                   save_to='', return_fig=False, return_img=False, transparent=False, font_size=12):
    fig, ax = plt.subplots(1, 1)

    ax.plot(conc, np.array(test))
    # ax.axhline(y=threshold, color='r', linestyle='--', label='Limit of Fitting')
    ax.set_xlim(calc_mono_lim(conc, edge_adj=0))
    ax.set_ylim(calc_mono_lim(np.array(test)))
    # ax.set_ylim([0, 1.05])
    ax.set_xlabel('Concentration / ' + conc_unit, fontsize=font_size)
    ax.set_ylabel('Metric', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.legend(fontsize=font_size, frameon=False)

    # plt.subplots_adjust(hspace=0)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
