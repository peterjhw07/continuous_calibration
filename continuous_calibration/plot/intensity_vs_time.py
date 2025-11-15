"""CC Intensity vs. Time Plotting"""

import matplotlib.pyplot as plt
from continuous_calibration.plot.plot_func import units_adjust, plot_process, calc_mono_lim, calc_multi_lim


# Plot intensity vs. time
def plot_intensity_vs_time(t, intensity, t_disc_add=[], diffusion_delay=0,
                           time_unit='time_unit', intensity_unit='AU', f_format='svg',
                           save_to='', return_fig=False, return_img=False, transparent=False, font_size=12):
    num_spec = intensity.shape[1]
    time_unit, intensity_unit = units_adjust([time_unit, intensity_unit])
    fig = plt.figure(figsize=(1 * 6, num_spec * 5))

    for col in range(num_spec):
        ax = plt.subplot(1, num_spec, col + 1)
        ax.scatter(t, intensity[:, col], 8, 'k', label='GenData')
        if t_disc_add and diffusion_delay:
            for add in t_disc_add[col]:
                ax.axvline(x=add, color='b', linestyle='--')
                ax.axvline(x=add + diffusion_delay, color='b', linestyle='--')
        ax.set_xlim(calc_mono_lim(t, edge_adj=0))
        ax.set_ylim(calc_mono_lim(intensity[:, col]))
        ax.set_xlabel('Time' + time_unit, fontsize=font_size)
        ax.set_ylabel('Intensity' + intensity_unit, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

    return plot_process(f_format, save_to, transparent, fig, return_fig, return_img)
