import numpy as np
import continuous_calibration as cc


# t = np.linspace(0, 5, 25+0)
num_spec = 1
vol0 = 5
mol0 = [0]
add_sol_conc = [(8,)]
add_cont_rate = [(0.05 / 60, 0)]
t_cont = [(60, 570)]
add_one_shot = []
t_one_shot = []
sub_cont_rate = 0
t_col = 1
col = 2
filename = r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_23_11_24.xlsx'

df = cc.raw_import(filename, t_col=1, col=[2])

cc.run(df, t_col, col, vol0, mol0, add_sol_conc, add_cont_rate, t_cont,
                     add_one_shot, t_one_shot, sub_cont_rate)
