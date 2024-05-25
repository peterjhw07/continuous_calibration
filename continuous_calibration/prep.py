import math
import numpy as np


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


def avg_repeats(conc, intensity):
    # Find unique values in each column of conc
    unique_conc, inverse_indices = np.unique(conc, axis=0, return_inverse=True)

    # Initialize arrays to store the sums and counts for averaging
    sums = np.zeros_like(unique_conc, dtype=float)
    counts = np.zeros_like(unique_conc, dtype=int)

    # Accumulate sums and counts for each unique row in conc
    np.add.at(sums, inverse_indices, intensity)
    np.add.at(counts, inverse_indices, 1)

    # Avoid division by zero and calculate the average
    unique_intensity = sums / counts.astype(float)
    unique_intensity[np.isnan(unique_intensity)] = 0

    return unique_conc, unique_intensity

