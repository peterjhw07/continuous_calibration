"""CC Intensity vs. Time Plotting"""

import matplotlib.pyplot as plt
import base64
from continuous_calibration.plot.plot_func import plot_process


def plot_intensity_vs_time(t, intensity, t_one_shot=[], diffusion_delay=0,
                           time_unit="time_unit", intensity_unit="AU", f_format='svg',
                           save_to='', return_fig=False, return_image=False, transparent=False, font_size=12):
    num_spec = intensity.shape[1]
    fig = plt.figure(figsize=(1 * 6, num_spec * 5))

    for col in range(num_spec):
        ax = plt.subplot(1, num_spec, col + 1)
        ax.scatter(t, intensity[:, col], 8, 'k', label='Data')
        if t_one_shot and diffusion_delay:
            for shot in t_one_shot[col]:
                ax.axvline(x=shot, color='b', linestyle='--')
                ax.axvline(x=shot + diffusion_delay, color='b', linestyle='--')
        ax.set_xlim([min(t), max(t)])
        ax.set_xlabel('Time / ' + time_unit, fontsize=font_size)
        ax.set_ylabel('Intensity / ' + intensity_unit, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

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