"""CC General Data Preparation"""

import pandas as pd
import math
import numpy as np
from continuous_calibration.fitting import gen_eqs, apply_eqs


# Converts int and float to list
def make_int_float_list(item):
    if isinstance(item, (int, float)):
        return [item]
    else:
        return item


# Converts int and float into lists inside tuples
def make_list_into_lists(item):
    if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (int, float)):
        return [item]
    else:
        return item


# Process species input
def process_spec_input(spec_name, col):

    col = make_int_float_list(col)
    num_spec = len(col)

    if spec_name is None:
        if num_spec == 1:
            spec_name = ['']
        else:
            spec_name = ['Species ' + i for i in range(1, num_spec + 1)]

    if len(spec_name) > 1:
        species_alt = [spec + ' ' for spec in spec_name]
    else:
        species_alt = ['']

    return spec_name, species_alt, num_spec, col


# Process data input
def process_data_input(df, num_spec, t_col, col):
    if isinstance(df, pd.DataFrame):
        data_arr = df.to_numpy()
    elif isinstance(df, (int, float, list, tuple)):
        if isinstance(df, (int, float)):
            df = [df]
        data_arr = np.zeros((len(df), num_spec))
        data_arr[:, 0] = df
    else:
        data_arr = df
    if t_col and isinstance(t_col, str):
        t_col = df.columns.get_loc(t_col)
    if col and isinstance(col, str):
        col = df.columns.get_loc(col)
    elif col and isinstance(col, (list, tuple)):
        if isinstance(col[0], str):
            col = [df.columns.get_loc(i) for i in col]
    return data_arr, t_col, col


# Process generation input
def process_gen_input(df, spec_name, t_col, col, mol0, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol,
                      t_disc_add, fit_lim):

    spec_name, species_alt, num_spec, col = process_spec_input(spec_name, col)
    data_arr, t_col, col = process_data_input(df, num_spec, t_col, col)

    add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, fit_lim = \
        map(make_int_float_list, [add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, fit_lim])

    cont_add_rate, t_cont_add, disc_add_vol, t_disc_add = \
        map(make_list_into_lists, [cont_add_rate, t_cont_add, disc_add_vol, t_disc_add])

    if mol0 is None:
        mol0 = [None] * num_spec
        mol0_temp = [0] * num_spec
    elif isinstance(mol0, (list, tuple)):
        mol0_temp = [i if i is not None else 0 for i in mol0]
    elif isinstance(mol0, (int, float)) and mol0 == 0:
        mol0 = [0] * num_spec
        mol0_temp = mol0.copy()
    else:
        mol0 = make_int_float_list(mol0)
        mol0_temp = mol0.copy()

    return (data_arr, spec_name, species_alt, num_spec, t_col, col, mol0, mol0_temp,
            add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add, fit_lim)


# Process application input
def process_apply_input(df, spec_name, t_col, col, param):

    spec_name, species_alt, num_spec, col = process_spec_input(spec_name, col)
    data_arr, t_col, col = process_data_input(df, num_spec, t_col, col)

    if isinstance(param, (list, tuple)) and len(param) > 0 and isinstance(param[0], (int, float)):
        param = [param]
    elif isinstance(param, (list, tuple)) and len(param) > 0 and isinstance(param[0], dict):
        param = [list(p.values()) for p in param]

    return data_arr, spec_name, species_alt, num_spec, t_col, col, param


# Adjusts DataFrame and axis titles
def units_adjust(units):
    if not isinstance(units, list):
        units = [units]
    for i in range(len(units)):
        if units[i]:
            units[i] = ' / ' + units[i]
        else:
            units[i] = ''
    return units


# Smooth data (if required)
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


# Average identical concentrations
def avg_repeats(conc, intensity, zero=False):
    # Find unique values in each column of conc
    unique_conc, inverse_indices = np.unique(conc, axis=0, return_inverse=True)

    # Initialize arrays to store the sums and counts for averaging
    sums = np.zeros_like(unique_conc, dtype=float)
    counts = np.zeros_like(unique_conc, dtype=int)
    std = np.zeros_like(unique_conc, dtype=float)

    # Accumulate sums and counts for each unique row in conc
    np.add.at(sums, inverse_indices, intensity)
    np.add.at(counts, inverse_indices, 1)

    # Avoid division by zero and calculate the average
    unique_average_intensity = sums / counts.astype(float)
    unique_average_intensity[np.isnan(unique_average_intensity)] = 0

    if zero:
        zero_shift = unique_average_intensity[0, :].flatten()
        unique_average_intensity -= zero_shift
    else:
        zero_shift = 0

    # Calculate standard deviation of unique concentrations
    np.add.at(std, inverse_indices, ((intensity - zero_shift) - unique_average_intensity[inverse_indices]) ** 2)
    std = np.sqrt(std / counts.astype(float))

    return unique_conc, unique_average_intensity, std


# Allow for delay due to diffusion
def remove_diffusion_delay(data_org, t_col, t_disc_add, diffusion_delay):
    if t_disc_add and diffusion_delay:
        indices = []
        for t in t_disc_add[0]:
            indices += np.where((data_org[:, t_col] >= t) & (data_org[:, t_col] < t + diffusion_delay))[0].tolist()
        data_org = np.delete(data_org, indices, axis=0)
    return data_org


# Sort fitting equations
def sort_fit_eq(fit_eq, intercept, fit_type='gen'):
    if 'gen' in fit_type.lower():
        eqs = gen_eqs
    else:
        eqs = apply_eqs
    if 'lin' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Linear_intercept')
        else:
            model = eqs.fit_eq_map.get('Linear')
    elif 'log' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Logarithmic_intercept')
        else:
            model = eqs.fit_eq_map.get('Logarithmic')
    elif 'exp' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Exponential_intercept')
        else:
            model = eqs.fit_eq_map.get('Exponential')
    elif 'tan' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Tangent_intercept')
        else:
            model = eqs.fit_eq_map.get('Tangent')
    elif 'mich' in fit_eq.lower() or 'mm' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Michaelis-Menten_intercept')
        else:
            model = eqs.fit_eq_map.get('Michaelis-Menten')
    elif 'lang' in fit_eq.lower():
        if intercept:
            model = eqs.fit_eq_map.get('Langmuir_intercept')
        else:
            model = eqs.fit_eq_map.get('Langmuir')
    elif 'custom' in fit_eq.lower():
        model = eqs.fit_eq_map.get('Custom')
    else:
        try:
            model = eqs.fit_eq_map.get(fit_eq)
        except:
            print('Non-existent model name.')
    return model
