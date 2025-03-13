"""CC Limit of Linearity Test Plotting"""

import matplotlib.pyplot as plt
import numpy as np
from continuous_calibration.plot.plot_func import plot_process


def plot_lol_tests(conc, lol_test_df, threshold, conc_unit="moles_unit volume_unit^-1",
                   intensity_unit="AU", f_format='svg',
                   save_to='', return_fig=False, return_img=False, transparent=False, font_size=12):
    fig, ax = plt.subplots(1, 1)

    for header in ["Rainbow Test", "Runs Test", "Harvey-Collier Test"]:
        ax.plot(conc, np.array(lol_test_df[header]), label=header)
    ax.axhline(y=threshold, color='b', linestyle='--', label='Limit of Linearity')
    ax.set_xlim([min(conc), max(conc)])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Concentration / ' + conc_unit, fontsize=font_size)
    ax.set_ylabel('p value', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(fontsize=font_size, frameon=False)

    # plt.subplots_adjust(hspace=0)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
