"""CC Plotting Functions"""

# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import io
import base64
import logging
from continuous_calibration import smooth


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# calculate x limits from x data
def calc_x_lim(t, edge_adj):
    return [float(min(t) - (edge_adj * max(t))), float(max(t) * (1 + edge_adj))]


# calculate y limits from y data
def calc_y_lim(exp, fit, edge_adj):
    lower_lim = min(np.nanmin(exp), np.nanmin(fit))
    upper_lim = max(np.nanmax(exp), np.nanmax(fit))
    if lower_lim != upper_lim:
        return [float(lower_lim - edge_adj * upper_lim), float(upper_lim * (1 + edge_adj))]
    else:
        return [float(lower_lim * (1 - edge_adj)), float(lower_lim * (1 + edge_adj))]


# processes plotted data
def plot_process(return_fig, fig, f_format, save_disk, save_to, transparent):
    if return_fig:
        return fig, fig.get_axes()

    # correct mimetype based on filetype (for displaying in browser)
    if f_format == 'svg':
        mimetype = 'image/svg+xml'
    elif f_format == 'png':
        mimetype = 'image/png'
    elif f_format == 'jpg':
        mimetype = 'image/jpg'
    elif f_format == 'pdf':
        mimetype = 'application/pdf'
    elif f_format == 'eps':
        mimetype = 'application/postscript'
    else:
        raise ValueError('Image format {} not supported.'.format(format))

    # save to disk if desired
    if save_disk:
        plt.savefig(save_to, transparent=transparent)

    # save the figure to the temporary file-like object
    # plt.show()
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=transparent)
    plt.close()
    img.seek(0)
    return img, mimetype


# plot time vs conc
def plot_conc_vs_time(t_df, exp_conc_df=None, fit_conc_df=None, col=None, show_asp=None,
                      method="lone", f_format='svg', return_image=False, save_disk=False,
                      save_to='take_fit.svg', return_fig=False, transparent=False):

    t = pd.DataFrame.to_numpy(t_df)

    if exp_conc_df is not None:
        exp_conc_headers = ['a'] # [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(exp_conc_df.columns)]
        exp_conc = pd.DataFrame.to_numpy(exp_conc_df)
    else:
        exp_conc_headers = []
        exp_conc = pd.DataFrame.to_numpy(fit_conc_df)
    if fit_conc_df is not None:
        fit_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(fit_conc_df.columns)]
        fit_conc = pd.DataFrame.to_numpy(fit_conc_df)
    else:
        fit_conc_headers = []
        fit_conc = exp_conc
        fit_col = []

    if isinstance(show_asp, str): show_asp = [show_asp]
    if col is not None and show_asp is None:
        fit_col = [i for i in range(len(col)) if col[i] is not None]
        non_fit_col = [i for i in range(len(col)) if col[i] is None]
    if ("lone_all" in method or "sep_all" in method) and fit_conc_df is not None:
        show_asp = ["y"] * len(fit_conc_headers)
    if show_asp is not None:
        fit_col = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]
        non_fit_col = [i for i in range(len(show_asp)) if 'n' in show_asp[i]]
    if "comp" in method and (len(non_fit_col) == 0 or fit_conc_df is None): method = "lone"
    if show_asp is not None and 'y' not in show_asp and 'y' not in show_asp[0]:
        print("If used, show_asp must contain at least one 'y'. Plot time_vs_conc has been skipped.")
        return

    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 10

    t_adj = t * x_ax_scale
    exp_conc_adj = exp_conc * y_ax_scale
    fit_conc_adj = fit_conc * y_ax_scale

    x_label_text = list(t_df.columns)[0]
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"

    cur_exp = 0
    cur_clr = 0
    if "lone" in method:  # lone plots a single figure containing all exps and fits as specified
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        #plt.rcParams.update({'font.size': 15})
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        for i in range(len(exp_conc_headers)):
            if len(t_adj) <= 50:
                ax1.scatter(t_adj, exp_conc_adj[:, i], label=exp_conc_headers[i])
            else:
                ax1.plot(t_adj, exp_conc_adj[:, i], label=exp_conc_headers[i])
        for i in fit_col:
            ax1.plot(t_adj, fit_conc_adj[:, i], label=fit_conc_headers[i])
        if len(fit_col) == 0: fit_col = range(len(exp_conc_headers))
        ax1.set_xlim(calc_x_lim(t_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(exp_conc_adj, fit_conc_adj[:, fit_col], edge_adj))
        ax1.legend(prop={'size': 10}, frameon=False)

    elif "comp" in method:  # plots two figures, with the first containing show_asp (or col if show_asp not specified) and the second containing all fits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #plt.rcParams.update({'font.size': 15})
        for i in range(len(exp_conc_headers)):
            if len(t_adj) <= 50:
                ax1.scatter(t_adj, exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=exp_conc_headers[i])
                ax2.scatter(t_adj, exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=exp_conc_headers[i])
                cur_clr += 1
            else:
                ax1.plot(t_adj, exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=exp_conc_headers[i])
                ax2.plot(t_adj, exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=exp_conc_headers[i])
                cur_clr += 1
        for i in fit_col:
            ax1.plot(t_adj, fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=fit_conc_headers[i])
            ax2.plot(t_adj, fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=fit_conc_headers[i])
            cur_clr += 1
        for i in non_fit_col:
            ax2.plot(t_adj, fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=fit_conc_headers[i])
            cur_clr += 1

        ax1.set_xlim(calc_x_lim(t_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(exp_conc_adj, fit_conc_adj[:, fit_col], edge_adj))
        ax2.set_xlim(calc_x_lim(t_adj, edge_adj))
        ax2.set_ylim(calc_y_lim(exp_conc_adj, fit_conc_adj, edge_adj))

        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    elif "sep" in method:
        num_spec = max([len(exp_conc_headers), len(fit_conc_headers)])
        grid_shape = (int(round(np.sqrt(len(fit_col)))), int(math.ceil(np.sqrt(len(fit_col)))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for j, i in enumerate(fit_col):
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            if col is not None and col[i] is not None and exp_conc_df is not None:
                if len(t_adj) <= 50:
                    ax.scatter(t_adj, exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=exp_conc_headers[cur_exp])
                else:
                    ax.plot(t_adj, exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=exp_conc_headers[cur_exp])
                ax.set_ylim(calc_y_lim(exp_conc_adj[:, cur_exp], fit_conc_adj[:, i], edge_adj))
                cur_exp += 1
                cur_clr += 1
            else:
                set_y_lim = calc_y_lim(fit_conc_adj[:, i], fit_conc_adj[:, i], edge_adj)
                if set_y_lim[0] != set_y_lim[1]: ax.set_ylim(set_y_lim)
            if fit_conc_df is not None:
                ax.plot(t_adj, fit_conc_adj[:, i], color=std_colours[cur_clr], label=fit_conc_headers[i])
            cur_clr += 1

            ax.set_xlim(calc_x_lim(t_adj, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            plt.legend(prop={'size': 10}, frameon=False)
    else:
        print("Invalid method inputted. Please enter appropriate method or remove method argument.")
        return

    # plt.show()
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


def org_plot(t, intensity, font_size=12):
    fig, ax = plt.subplots(1, 1)

    ax.scatter(t, intensity, 8, 'k', label='Data')
    ax.set_xlim([min(t), max(t)])
    ax.set_xlabel('Time / min', fontsize=font_size)
    ax.set_ylabel('Intensity', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # plt.subplots_adjust(hspace=0)

    plt.show()


def cc_plot(conc, intensity, limit, fit_line, res_fit, smooth_intensity=None, font_size=12):
    fig, (ax_upper, ax_lower) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax_upper.scatter(conc, intensity, 8, 'k', label='Data')
    if smooth_intensity:
        ax_upper.plot(conc, smooth_intensity, 'g', label='Smoothed Fit')
    ax_upper.plot(conc[:, 0], fit_line[:, 0], 'r', label='Linear Fit')
    ax_upper.axvline(x=conc[limit, 0], color='b', linestyle='--', label='Limit of Linearity')
    ax_upper.set_xlim([min(conc[:, 0]), max(conc[:, 0])])
    # ax_upper.set_xlabel('Concentration / mM', fontsize=font_size)
    ax_upper.set_ylabel('Intensity', fontsize=font_size)
    ax_upper.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_upper.tick_params(axis='y', which='major', labelsize=font_size)
    ax_upper.legend(loc='lower right', fontsize=font_size, frameon=False)

    ax_lower.scatter(conc[:limit, 0], intensity[:limit, 0] - fit_line[:limit, 0], 8, 'k', label='Residuals')
    ax_lower.plot(conc[:limit, 0], res_fit[:limit, 0], 'r', label='Residual Fit')
    ax_lower.axvline(x=conc[limit, 0], color='b', linestyle='--', label='Limit of Linearity')
    ax_lower.set_xlim([min(conc[:, 0]), max(conc[:, 0])])
    ax_lower.set_xlabel('Concentration / mM', fontsize=font_size)
    ax_lower.set_ylabel('Intensity deviation', fontsize=font_size)
    ax_lower.tick_params(axis='both', which='major', labelsize=font_size)
    # ax_lower.legend(loc='lower right', fontsize=font_size, frameon=False)

    plt.subplots_adjust(hspace=0)

    plt.show()


def test_plot(t, df, font_size=12):
    fig, ax = plt.subplots(1, 1)

    for header in df.columns:
        sg = smooth.savgol_filter(df[header], 50, 2)
        ax.plot(t[50:], sg[50:] / sg[50:].max(), label=header)
        # ax.plot(t, np.array(df[header] / df[header].max()), label=header)
    ax.set_xlim([min(t), max(t)])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Time / min', fontsize=font_size)
    ax.set_ylabel('Intensity', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(fontsize=font_size, frameon=False)

    # plt.subplots_adjust(hspace=0)

    plt.show()
