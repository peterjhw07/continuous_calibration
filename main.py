import numpy as np
import continuous_calibration as cc


# t = np.linspace(0, 5, 25+0)
t_col = 1
t_col = 0
col = 2
col = 2
mol0 = [0]
vol0 = 5
vol0 = 0.017951
add_sol_conc = [(12.5,)]
add_cont_rate = [(0.05 / 60, 0)]
add_cont_rate = [(0.00002095, 0)]
t_cont = [(60, 1260)]
t_cont = [(2.5, 42.5)]
add_one_shot = []  # [(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)]
t_one_shot = []  # [(60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080)]
sub_cont_rate = 0.00002
get_lol = "min"
# filename = r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_04_11.xlsx'
filename = r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_06_24.xlsx'
sheet_name = 'Sheet1'

df = cc.raw_import(filename, sheet_name=sheet_name, t_col=t_col, col=col)

data = cc.run(df, t_col=t_col, col=col, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate,
              t_cont=t_cont, add_one_shot=add_one_shot, t_one_shot=t_one_shot, sub_cont_rate=sub_cont_rate,
              diffusion_delay=0, get_lol=get_lol)

data.plot_intensity_vs_time(f_format='png', save_to='intensity_vs_time.png')
data.plot_intensity_vs_conc(f_format='png', save_to='intensity_vs_conc.png')
# if get_lol:
#    data.plot_lol_tests(f_format='png', save_to='lol_tests.png')

cc.export_xlsx(data.proc_df, r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_06_24_test_output.xlsx')
