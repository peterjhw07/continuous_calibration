"""CC Limit of Linearity Calculation"""

import numpy as np
import pandas as pd
from continuous_calibration.fitting import linearity_tests, regression


def get_lol(conc, intensity, spec_name, intercept=False, lol_test="Rainbow", get_lol="min",
            threshold=0.05, path_length=None, intensity_unit="AU", conc_unit="moles_unit volume_unit$^{-1}$",
            path_length_unit="path_length_unit"):
    if "rain" in lol_test.lower():
        lol_test = "Rainbow"
    elif "run" in lol_test.lower():
        lol_test = "Runs"
    elif "harvey" in lol_test.lower() or "hc" in lol_test.lower():
        lol_test = "Harvey-Collier"

    indices = np.arange(conc.shape[0]).reshape(-1, 1)

    # Perform optimization to find the number of data points that maximizes R^2
    if len(spec_name) > 1:
        species_alt = [spec + ' ' for spec in spec_name]
    else:
        species_alt = ['']
    test_columns = ["Limit"] + [f"{spec}{test}" for spec in species_alt for test in
                                ["RMSE", "MAE", "Rainbow Test", "Runs Test", "Harvey-Collier Test"]]
    tests_df = pd.DataFrame(index=range(len(conc)), columns=test_columns)
    tests_df["Limit"] = range(len(conc))
    gradient_unit = intensity_unit + ' (' + conc_unit + ")$^{-1}$"
    mec_unit = gradient_unit + ' ' + path_length_unit + "$^{-1}$"
    lol_columns = ["Species", "Rainbow", "Runs", "Harvey-Collier", "LOL", 'Gradient / ' + gradient_unit]
    if path_length:
        lol_columns += ['Molar Extinction Coefficient Lower / ' + mec_unit,
                        'Molar Extinction Coefficient Upper / ' + mec_unit]
    elif intercept:
        lol_columns += ['Intercept / ' + intensity_unit]
    lol_df = pd.DataFrame(index=spec_name, columns=lol_columns)
    fit_lines = np.full((len(conc), len(spec_name)), None, dtype=object)
    fit_resid = np.full((len(conc), len(spec_name)), None, dtype=object)
    lol_df["Species"] = spec_name
    for i in range(len(species_alt)):
        for limit in range(3, len(conc)):
            fit_res, _ = regression.lin_regress(conc[:, i], intensity[:, i], limit, intercept=intercept)
            tests_df.at[limit, species_alt[i] + 'RMSE'] = fit_res.rmse
            tests_df.at[limit, species_alt[i] + 'MAE'] = fit_res.mae
            tests_df.at[limit, species_alt[i] + 'Rainbow Test'] = linearity_tests.rainbow_test(fit_res)
            tests_df.at[limit, species_alt[i] + 'Runs Test'] = linearity_tests.runs_test(fit_res)
            tests_df.at[limit, species_alt[i] + 'Harvey-Collier Test'] = linearity_tests.harvey_collier_test(fit_res)

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        lol_df.at[spec_name[i], "Rainbow"] = try_except_test(indices, tests_df, species_alt[i],
                                                             'Rainbow Test', get_lol, threshold)
        lol_df.at[spec_name[i], "Runs"] = try_except_test(indices, tests_df, species_alt[i],
                                                          'Runs Test', get_lol, threshold)
        lol_df.at[spec_name[i], "Harvey-Collier"] = try_except_test(indices, tests_df, species_alt[i],
                                                                    'Harvey-Collier Test', get_lol, threshold)
        lol_df.at[spec_name[i], "LOL"] = lol_df.at[spec_name[i], lol_test]

        fit_res, fit_line = regression.lin_regress(conc[:, i], intensity[:, i],
                                                   lol_df.at[spec_name[i], "LOL"], intercept=intercept)
        fit_lines[:len(fit_line), i:i+1] = fit_line
        fit_resid[:len(fit_res.resid), i:i+1] = fit_res.resid.reshape(-1, 1)
        if intercept:
            lol_df.at[spec_name[i], 'Gradient / ' + gradient_unit] = fit_res.params[1]
            lol_df.at[spec_name[i], 'Intercept / ' + intensity_unit] = fit_res.params[0]
        else:
            lol_df.at[spec_name[i], 'Gradient / ' + gradient_unit] = fit_res.params[0]
            if path_length:
                lol_df.at[spec_name[i], 'Molar Extinction Coefficient / ' + mec_unit] = \
                    lol_df.at[spec_name[i], 'Gradient / ' + gradient_unit] / path_length

    if intercept:
        coeff = {"m": lol_df['Gradient / ' + gradient_unit].to_list(),
                 "c": lol_df['Intercept / ' + intensity_unit].to_list()}
        mec = None
    else:
        coeff = {"m": lol_df['Gradient / ' + gradient_unit].to_list()}
        if path_length:
            mec = lol_df['Molar Extinction Coefficient / ' + mec_unit].to_list()
        else:
            mec = None

    if len(species_alt) == 1:
        lol_df = lol_df.drop(columns="Species")

    return tests_df, lol_df, fit_lines, fit_resid, coeff, mec


def try_except_test(indices, tests_df, species_alt, test, criterion, threshold=0.05):
    try:
        if isinstance(criterion, (int, float)):
            return int(criterion)
        elif "max" in criterion.lower():
            return max(indices[tests_df[species_alt + test] == max(tests_df[species_alt + test].dropna())])[0]
        elif "first" in criterion.lower():
            return max(0, min(indices[tests_df[species_alt + test] < threshold])[0] - 1)
        elif "last" in criterion.lower():
            return max(indices[tests_df[species_alt + test] >= threshold])[0]
        else:
            return max(indices[tests_df[species_alt + test] >= threshold])[0]
    except:
        return 1
