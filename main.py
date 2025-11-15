"""CC Run Script"""

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
cont_add_rate = [(0.05 / 60, 0)]
cont_add_rate = [(0.00002095, 0)]
t_cont_add = [(60, 1260)]
t_cont_add = [(2.5, 42.5)]
disc_add_vol = []
t_disc_add = []
cont_sub_rate = 0.00002
get_lof = "min"
# filename = r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_04_11.xlsx'
filename = r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_06_24.xlsx'
sheet_name = 'Sheet1'

df = cc.raw_import(filename, sheet_name=sheet_name, t_col=t_col, col=col)

data = cc.gen(df, t_col=t_col, col=col, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc, cont_add_rate=cont_add_rate,
              t_cont_add=t_cont_add, disc_add_vol=disc_add_vol, t_disc_add=t_disc_add, lof_method=get_lof, diffusion_delay=0)

data.plot_intensity_vs_time(f_format='png', save_to='intensity_vs_time.png')
data.plot_intensity_vs_conc(f_format='png', save_to='intensity_vs_conc.png')
if get_lof:
    data.plot_lof_test(f_format='png', save_to='lof_test.png')

cc.export_xlsx(data.all_df, r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\NT_24_06_24_test_output.xlsx')
