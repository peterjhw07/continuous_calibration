import math
import numpy as np


#
def make_int_float_list(item):
    if isinstance(item, (int, float)):
        item = [item]
    return item


# Converts int and float into lists inside tuples
def make_list_into_lists(item):
    if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (int, float)):
        return [item]


def process_input(spec_name, col, mol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot):
    col, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot = \
        map(make_int_float_list, [col, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot])

    num_spec = len(col)

    add_cont_rate, t_cont, add_one_shot, t_one_shot = \
        map(make_list_into_lists, [add_cont_rate, t_cont, add_one_shot, t_one_shot])

    if spec_name is None:
        if num_spec == 1:
            spec_name = [""]
        else:
            spec_name = ["Species " + i for i in range(1, num_spec + 1)]

    if not mol0:
        mol0 = [0] * num_spec
    else:
        mol0 = make_int_float_list(mol0)

    return spec_name, num_spec, col, mol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot


# smooth data (if required)
def data_smooth(arr, d_col, win=1, inc=1):
    if win <= 1:
        d_ra = arr[:, d_col]
    else:
        ret = np.cumsum(arr[:, d_col], dtype=float)
        ret[win:] = ret[win:] - ret[:-win]
        d_ra = ret[win - 1:] / win
    if inc < 1:
        exp_rows = np.linspace(0, len(d_ra) - 1, num=int(((len(d_ra) - 1) * inc))).astype(int)
        d_ra = d_ra[exp_rows]
    return d_ra


def avg_repeats(conc, intensity, zero=False):
    # Find unique values in each column of conc
    unique_conc, inverse_indices = np.unique(conc, axis=0, return_inverse=True)

    # Initialize arrays to store the sums and counts for averaging
    sums = np.zeros_like(unique_conc, dtype=float)
    counts = np.zeros_like(unique_conc, dtype=int)
    error = np.zeros_like(unique_conc, dtype=float)

    # Accumulate sums and counts for each unique row in conc
    np.add.at(sums, inverse_indices, intensity)
    np.add.at(counts, inverse_indices, 1)

    # Avoid division by zero and calculate the average
    unique_intensity = sums / counts.astype(float)
    unique_intensity[np.isnan(unique_intensity)] = 0

    if zero:
        zero_shift = unique_intensity[0]
        unique_intensity -= zero_shift
    else:
        zero_shift = 0

    # Calculate standard deviation of unique concentrations
    np.add.at(error, inverse_indices, ((intensity - zero_shift) - unique_intensity[inverse_indices]) ** 2)
    error = np.sqrt(error / counts.astype(float))

    return unique_conc, unique_intensity, error


def remove_diffusion_delay(data_org, t_col, t_one_shot, diffusion_delay):
    if t_one_shot and diffusion_delay:
        indices = []
        for t in t_one_shot[0]:
            indices += np.where((data_org[:, t_col] >= t) & (data_org[:, t_col] < t + diffusion_delay))[0].tolist()
        data_org = np.delete(data_org, indices, axis=0)
    return data_org
