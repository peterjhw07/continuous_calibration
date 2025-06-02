"""CC Concentration vs. Time Plotting"""

import matplotlib.pyplot as plt
from continuous_calibration.plot.plot_func import plot_process, calc_mono_lim, calc_multi_lim


def plot_conc_vs_time(t, conc, time_unit='time_unit', conc_unit='AU', f_format='svg',
                      save_to='', return_fig=False, return_img=False, transparent=False, font_size=12):
    num_spec = conc.shape[1]
    fig = plt.figure(figsize=(num_spec * 6, 1 * 5))

    for col in range(num_spec):
        ax = plt.subplot(1, num_spec, col + 1)
        ax.scatter(t, conc[:, col], 8, 'k', label='GenData')
        ax.set_xlim(calc_mono_lim(t, edge_adj=0))
        ax.set_ylim(calc_mono_lim(conc[:, col]))
        ax.set_xlabel('Time / ' + time_unit, fontsize=font_size)
        ax.set_ylabel('Concentration / ' + conc_unit, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

    # plt.subplots_adjust(hspace=0)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
