"""Run CC"""

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
    if isinstance(s, str) and s != 'min' and s != 'max' and s != 'first' and s != 'last':
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
                         "add_cont_rate", "t_cont", "add_one_shot", "t_one_shot", "sub_cont_rate", "fit_eq",
                         "intercept", "lol_method", "path_length", "sg_win", "breakpoint_lim", "diffusion_delay",
                         "zero", "win", "inc", "time_unit", "conc_unit", "intensity_unit", "path_length_unit",
                         "filename", "sheet_name", "pic_save"]]

    total = np.empty([len(input_df), 15], object)
    for i in range(len(input_df)):
        [number, description, spec_name, t_col, col, mol0, vol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
         t_one_shot, sub_cont_rate, fit_eq, intercept, lol_method, path_length, sg_win, breakpoint_lim, diffusion_delay,
         zero, win, inc, time_unit, conc_unit, intensity_unit, path_length_unit, filename, sheet_name,
         pic_save] = (input_df.iloc)[i, :]
        print(number, description)

        spec_name, t_col, col, mol0, vol0, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, \
        sub_cont_rate, intercept, lol_method, path_length, sg_win, breakpoint_lim, diffusion_delay, zero, win, inc \
            = map(input_sort, [spec_name, t_col, col, mol0, vol0, add_sol_conc,
                               add_cont_rate, t_cont, add_one_shot, t_one_shot, sub_cont_rate, intercept, lol_method,
                               path_length, sg_win, breakpoint_lim, diffusion_delay, zero, win, inc])

        df = cc.raw_import(filename, sheet_name=sheet_name, t_col=t_col, col=col)
        starttime = timeit.default_timer()

        # cProfile.run('print(
        data = cc.gen(df, spec_name=spec_name, t_col=t_col, col=col, mol0=mol0, vol0=vol0, add_sol_conc=add_sol_conc,
                      add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot, t_one_shot=t_one_shot,
                      sub_cont_rate=sub_cont_rate, path_length=path_length, fit_eq=fit_eq, intercept=intercept,
                      lol_method=lol_method, sg_win=sg_win, breakpoint_lim=breakpoint_lim,
                      diffusion_delay=diffusion_delay, zero=zero, win=win, inc=inc, time_unit=time_unit,
                      conc_unit=conc_unit, intensity_unit=intensity_unit, path_length_unit=path_length_unit)

        with open("fit_output.pkl", 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        time_taken = timeit.default_timer() - starttime

        if plot_org:
            data.plot_intensity_vs_time(f_format='png', save_to=pic_save.replace('.png', '_intensity_vs_time.png'))

        data.plot_intensity_vs_conc(plot_resid=True, f_format='png',
                                    save_to=pic_save.replace('.png', '_intensity_vs_conc.png'))

        if fit_eq and "lin" in fit_eq.lower() and lol_method and plot_lol_tests:
            data.plot_lol_tests(f_format='png', save_to=pic_save.replace('.png', '_lol_tests.png'))

        export_df = data.all_df
        if 'y' in export_fit_csv:
            export_df.to_csv(gen_folder + 'CC_Fit_Results_' + str(number) + '.txt', index=False)
        if 'y' in export_fit_excel:
            cc.export_xlsx(export_df, gen_folder + 'CC_Fit_Results.xlsx', 'a', 'replace', str(number))

        total[i] = [number, description, data.est_t_cont, data.params, data.lol, data.mec, data.mol0_fit,
                    data.rss, data.rmse, data.mae, data.r2, data.r2_adj, data.aic, data.bic, time_taken]

    if 'y' in export_param:
        export_df = pd.DataFrame(total, columns=["Number", "Description", "Estimated t_cont", "Coefficients", "LOL",
                                                 "MEC", "mol0_fit",
                                                 "RSS", "RMSE", "MAE", "R2", "R2_adj", "AIC", "BIC", "script_runtime"])
        cc.export_xlsx(export_df, gen_folder + 'CC_Results.xlsx', 'a', 'new', date.today().strftime("%y%m%d"))
