"""
Continuous Calibration Example 1
Continuous calibration experiment with concentrated analyte solution being added to the monitored solution.
The calibration data is first simulated as an exponential function with noise added.
These simulated data are then fit to maximize the R2 of a linear function.
The obtained calibration curve is then used to fit three measured samples of unknown concentration.
"""

import numpy as np
import pandas as pd
from continuous_calibration import gen

# System parameters
vol0 = 0.01  # initially 0.01 L monitored solution volume
add_sol_conc = 1  # analyte addition solution of concentration 1 M
cont_add_rate = 0.0002  # analyte addition solution added at rate 0.0002 L / min
t_cont_add = 1  # addition of calibrant solution started at 1 min

time_unit = 'min'
conc_unit = 'M'

# Simulated calibration curve parameters
A = 100  # calibration gradient
k = 3  # exponential deviation
noise_frac = 0.01  # 1% noise of maximum intensity

# Calibration data simulation
df = pd.DataFrame(columns=['Time', 'Intensity'])
time = np.linspace(0, 10, 10 * 60 + 1)  # one reading every 1 s between 0-10 min
df['Time'] = time
conc = ((time - t_cont_add) * cont_add_rate * add_sol_conc) / (vol0 + (time - t_cont_add) * cont_add_rate)
conc = np.where(conc < 0, 0, conc)
df['Intensity'] = np.array(
    A * (1 - np.exp(-k * conc)) + A * np.random.uniform(-noise_frac, noise_frac, size=time.shape))

# Fitting parameters
fit_eq = 'linear'  # fitting using a linear function
intercept = False  # fixing the intercept at 0
lof_test = 'R2'  # maximizing R2
lof_method = 'auto'
smooth_eq = 'concave'  # also apply a GAM fit

# Run fitting
data = gen(df, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate, t_cont_add=t_cont_add, fit_eq=fit_eq,
           intercept=intercept, lof_test=lof_test, lof_method=lof_method, smooth_eq=smooth_eq, time_unit=time_unit,
           conc_unit=conc_unit)
data.plot_intensity_vs_time(f_format='png', save_to='figures/1.1)_intensity_vs_time.png')
data.plot_intensity_vs_conc(plot_resid=True, f_format='png', save_to='figures/1.2)_intensity_vs_conc.png')
data.plot_lof_test(f_format='png', save_to='figures/1.3)_lof_test.png')
print('Limit of fitting: ' + str(*data.lof) + ' ' + conc_unit)
print('Gradient: ' + str(data.params[0]['a']) + ' ' + conc_unit + '⁻¹')

# Apply fitted calibration curve to convert intensities into concentrations
apply_data = data.apply([10, 20, 30])
print(apply_data.fit_df)
