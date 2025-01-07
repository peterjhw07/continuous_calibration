"""Run CAKE"""

# Imports
import continuous_calibration as cc
import ast
import copy
import cProfile
import numpy as np
import pandas as pd
import timeit
from datetime import date
import re
import pickle


# Function for sorting Excel data into pythonic format.
def input_sort(s):
    if isinstance(s, str) and s != 'min' and s != 'max':
        return ast.literal_eval(s)
    else:
        return s


if __name__ == "__main__":
    export_fit_csv = "n"
    export_fit_excel = "n"
    export_param = "y"
    gen_folder = 'C:/Users/Peter/Documents/Postdoctorate_McIndoe/Work/CC/Program/'
    plot_org = True
    plot_lol_tests = True

    # create dataframe
    input_df = pd.read_excel(gen_folder + 'CC_Input.xlsx', sheet_name='Run', dtype=str)
    input_df.replace('""', None, inplace=True)
    input_df = input_df[["Number", "Description", "spec_name", "t_col", "col", "mol0", "vol0", "add_sol_conc",
                         "add_cont_rate", "t_cont", "add_one_shot", "t_one_shot", "sub_cont_rate", "diffusion_delay",
                         "fit_eq", "intercept", "get_lol", "path_length", "win", "inc", "sg_win", "time_unit",
                         "conc_unit", "intensity_unit", "path_length_unit", "filename", "sheet_name", "pic_save"]]

    total = np.empty([len(input_df), 7], object)
    for i in range(0, len(input_df)):
        [number, description, spec_name, t_col, col, mol0, vol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
         t_one_shot, sub_cont_rate, diffusion_delay, fit_eq, intercept, get_lol, path_length, win, inc, sg_win,
         time_unit, conc_unit, intensity_unit, path_length_unit, filename, sheet_name, pic_save] = input_df.iloc[i, :]
        print(number, description)

        spec_name, t_col, col, mol0, vol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, \
        sub_cont_rate, diffusion_delay, intercept, get_lol, path_length, win, inc, sg_win \
            = map(input_sort, [spec_name, t_col, col, mol0, vol0, add_sol_conc,
            add_cont_rate, t_cont, add_one_shot, t_one_shot, sub_cont_rate, diffusion_delay, intercept, get_lol,
            path_length, win, inc, sg_win])

        df = cc.raw_import(filename, sheet_name=sheet_name, t_col=t_col, col=col)
        starttime = timeit.default_timer()

        # cProfile.run('print(
        data = cc.run(df, spec_name=spec_name, t_col=t_col, col=col, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc,
                      add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot, t_one_shot=t_one_shot,
                      sub_cont_rate=sub_cont_rate, diffusion_delay=diffusion_delay, fit_eq=fit_eq, intercept=intercept,
                      get_lol=get_lol, path_length=path_length, win=win, inc=inc, sg_win=sg_win, time_unit=time_unit,
                      conc_unit=conc_unit, intensity_unit=intensity_unit, path_length_unit=path_length_unit)

        with open("fit_output.pkl", 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        time_taken = timeit.default_timer() - starttime

        if plot_org:
            save_to_replace = pic_save.replace('.png', '_intensity_vs_time.png')
            data.plot_intensity_vs_time(f_format='png', save_to=save_to_replace)

        save_to_replace = pic_save.replace('.png', '_intensity_vs_conc.png')
        data.plot_intensity_vs_conc(f_format='png', save_to=save_to_replace)

        if "lin" in fit_eq and get_lol and plot_lol_tests:
            save_to_replace = pic_save.replace('.png', '_lol_tests.png')
            data.plot_lol_tests(f_format='png', save_to=save_to_replace)

        export_df = data.proc_df
        if 'y' in export_fit_csv:
            export_df.to_csv(gen_folder + 'CC_Fit_Results_' + str(number) + '.txt', index=False)
        if 'y' in export_fit_excel:
            cc.export_xlsx(export_df, gen_folder + 'CC_Fit_Results.xlsx', 'a', 'replace', str(number))

        total[i] = [number, description, data.lol, data.gradient, data.intercept, data.mec, time_taken]

    if 'y' in export_param:
        export_df = pd.DataFrame(total, columns=["Number", "Description", "LOL", "Gradient", "Intercept", "MEC",
                                                 "script_runtime"])
        # "RSS", "R2", "R2_adj", "RMSE", "MAE", "AIC", "BIC",
        cc.export_xlsx(export_df, gen_folder + 'CC_Results.xlsx', 'a', 'new', date.today().strftime("%y%m%d"))
