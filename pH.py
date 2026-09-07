"""CC pH Calculation"""

from chempy.chemistry import Equilibrium, Substance
from chempy.equilibria import EqSystem
import numpy as np
import pandas as pd
import continuous_calibration as cc

# Define species
H = Substance.from_formula('H+')
Cl = Substance.from_formula('Cl-')
Na = Substance.from_formula('Na+')
OH = Substance.from_formula('OH-')
BPBH = Substance.from_formula('BPBH')
BPB = Substance.from_formula('BPB-')
H2O = Substance.from_formula('H2O')

# Define equilibria
wa_eq = Equilibrium({'BPBH': 1}, {'BPB-': 1, 'H+': 1}, 7.94e-5)
water_eq = Equilibrium({'H2O': 1}, {'H+': 1, 'OH-': 1}, 1e-14)

# Create equilibrium system
eqsys = EqSystem([wa_eq, water_eq], substances=[H, Cl, Na, OH, BPBH, BPB, H2O])

# Initial moles / mol
tot_mol = {
    'H+': 0,          # from 0.01 M HCl
    'Cl-': 0,         # spectator ion
    'Na+': 2.5E-5 * 5E-3,        # spectator ion
    'OH-': 2.5E-5 * 5E-3,        # from 0.005 M NaOH
    'BPBH': 2.5E-5 * 5E-3,     # weak acid
    'BPB-': 0.0 * 5E-3,      # conjugate base starts at 0
    'H2O': 55.35 * 5E-3,  # conjugate base starts at 0
}

add_rate = [10E-6, 50E-6]
t_add = [1, 11]
vol0 = 5E-3
HCl_conc = 0.025

# Solve equilibrium
df_import = cc.raw_import(r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\Case studies\Other UV-Vis\PJHW250807_Bromophenol_blue_CC_acid_10_50.xlsx', t_col=1)
array = np.empty((df_import.shape[0], 4))
t_add = list(t_add)
t_add.append(df_import.iloc[-1, 1])
for i, t in enumerate(df_import.iloc[:, 1]):
    add_vol = 0
    for j in range(len(t_add) - 1):
        if t - t_add[j] > 0:
            add_vol += min(t - t_add[j], t_add[j + 1]) * add_rate[j]
    total_vol = vol0 + add_vol
    tot_mol['H+'] = add_vol * HCl_conc
    tot_mol['Cl-'] = add_vol * HCl_conc
    tot_conc = {key: value / total_vol for key, value in tot_mol.items()}
    result = eqsys.solve(tot_conc, solver='nleq2')

    # Results
    pH = -np.log10(result.conc[0])
    array[i, :] = [total_vol, tot_mol['H+'], result.conc[0], pH]
cc.export_xlsx(pd.DataFrame(array, columns=['Total volume', '[H+]added', '[H+]', 'pH']),
               r'C:\Users\Peter\Documents\Postdoctorate_McIndoe\Work\CC\Case studies\Other UV-Vis\PJHW250807_Bromophenol_blue_CC_acid_10_50_pH.xlsx')
