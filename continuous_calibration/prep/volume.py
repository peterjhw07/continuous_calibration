import copy
import numpy as np


# Unpack time events
def get_events(lst):
    event_times = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                if isinstance(sub_item, int) or isinstance(sub_item, float):
                    event_times.append(sub_item)
        elif isinstance(item, int) or isinstance(item, float):
            event_times.append(item)
    return sorted(set(event_times))


# Prepare addition events
def prepare_add(num_spec, add, t):
    if add and t:
        event_t = np.array(get_events(t))
        dvols = np.zeros((len(event_t), num_spec))
        for i, j in enumerate(add):
            if j:
                for k in range(len(j)):
                    found_rows = [index for index, value in enumerate(event_t) if value == t[i][k]]
                    dvols[found_rows[0]:, i] = j[k]
        dvol = np.sum(dvols, axis=1)
    else:
        event_t = np.zeros(0)
        dvols = np.zeros((0, num_spec))
        dvol = np.zeros(0)
    return event_t, dvols, dvol


# Get continuous events and add discrete events
def get_cont_add_disc(num_spec, add_cont_rate, t_cont, event_t_disc, dmol_disc, dvol_disc, sub_cont_rate):
    t_cont, dvols_cont, dvol_cont = prepare_add(num_spec, add_cont_rate, t_cont)
    t_cont_add, dvols_cont_add, dvol_cont_add = t_cont, dvols_cont, dvol_cont
    mol_cont_add, vol_cont_add = np.zeros((len(t_cont), num_spec)), np.zeros(len(t_cont))

    for i, j in enumerate(event_t_disc):
        if j in t_cont:
            mol_cont_add[t_cont_add == j] += dmol_disc[i]
            vol_cont_add[t_cont_add == j] += dvol_disc[i]
        else:
            if len(t_cont[t_cont < j]) > 0:
                row_find = t_cont < j
                t_cont_add = np.append(t_cont_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, dvols_cont[row_find][-1]))
                dvol_cont_add = np.append(dvol_cont_add, dvol_cont[row_find][-1])
                mol_cont_add = np.vstack((mol_cont_add, dmol_disc[i]))
                vol_cont_add = np.append(vol_cont_add, dvol_disc[i])
            else:
                t_cont_add = np.append(t_cont_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, np.zeros((1, dvols_cont.shape[1]))))
                dvol_cont_add = np.append(dvol_cont_add, 0)
                mol_cont_add = np.vstack((mol_cont_add, dmol_disc[i]))
                vol_cont_add = np.append(vol_cont_add, dvol_disc[i])
    dvol_cont_add -= sub_cont_rate
    row_sort = t_cont_add.argsort()
    return t_cont_add[row_sort], dvols_cont_add[row_sort], dvol_cont_add[row_sort], \
           mol_cont_add[row_sort], vol_cont_add[row_sort]


# Add row to the start or end
def add_start_end_row(t, dvols, dvol, mol, vol, t_end, sub_cont_rate):
    if 0 not in t:
        t = np.append(0, t)
        dvols = np.vstack((np.zeros((1, dvols.shape[1])), dvols))
        dvol = np.append(-sub_cont_rate, dvol)
        mol = np.vstack((0, mol))
        vol = np.append(0, vol)
    if t_end not in t:
        t = np.append(t, t_end)
        dvols = np.vstack((dvols, dvols[-1]))
        dvol = np.append(dvol, dvol[-1])
        mol = np.vstack((mol, 0))
        vol = np.append(vol, 0)
    return t, dvols, dvol, mol, vol


# Finally calculate volume
def calc_mol_vol(t, dmol, dvol, dsub, mol, vol, mol0, vol0):
    mol[0] += mol0
    vol[0] += vol0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        mol[i] += mol[i - 1] + (dmol[i - 1] + mol[i - 1] * dsub / vol[i - 1]) * dt
        vol[i] += vol[i - 1] + dvol[i - 1] * dt
    return mol, vol


class EventData:
    def __init__(self, t, dmol, dvol, dsub, mol, vol):
        self.t = t
        self.dmol = dmol
        self.dvol = dvol
        self.dsub = dsub
        self.mol = mol
        self.vol = vol


def get_mol_vol(t, cont_event):
    num_spec = np.shape(cont_event.mol)[1]
    mol, vol = np.empty((len(t), num_spec)), np.empty(len(t))
    for i in range(len(cont_event.t) - 1):
        row_get = (t >= cont_event.t[i]) & (t < cont_event.t[i + 1])
        dt = t[row_get] - t[row_get][0]
        mol[row_get] = (cont_event.mol[i] + (cont_event.dmol[i] + (cont_event.mol[i] * cont_event.dsub / cont_event.vol[i])) * dt)[:, None]
        vol[row_get] = cont_event.vol[i] + cont_event.dvol[i] * dt
    mol[-1] = cont_event.mol[-1]
    vol[-1] = cont_event.vol[-1]
    return mol, vol


# Calculate additions and subtractions of species
def get_conc_events(t, num_spec, vol0, mol0, add_sol_conc, add_cont_rate, t_cont,
                     add_one_shot, t_one_shot, sub_cont_rate):

    sub_cont_rate = abs(sub_cont_rate) if sub_cont_rate else 0
    t_disc, dvols_disc, dvol_disc = prepare_add(num_spec, add_one_shot, t_one_shot)

    dmol_disc = dvols_disc
    for i, conc in enumerate(add_sol_conc):
        if conc:
            dmol_disc[:, i] *= conc

    t_cont, dvols_cont, dvol_cont, mol_cont, vol_cont = get_cont_add_disc(num_spec, add_cont_rate, t_cont, t_disc,
                                                                          dmol_disc, dvol_disc, sub_cont_rate)
    dsub_cont = -sub_cont_rate

    t_event, dvols_cont, dvol_cont, mol_cont, vol_cont = add_start_end_row(t_cont, dvols_cont, dvol_cont, mol_cont, vol_cont, t[-1], sub_cont_rate)

    dmol_cont = dvols_cont
    for i, conc in enumerate(add_sol_conc):
        if conc:
            dmol_cont[:, i] *= conc

    mol_cont, vol_cont = calc_mol_vol(t_event, dmol_cont, dvol_cont, dsub_cont, mol_cont, vol_cont, mol0, vol0)
    cont_event = EventData(t_event, dmol_cont, dvol_cont, dsub_cont, mol_cont, vol_cont)

    mol, vol = get_mol_vol(t, cont_event)
    conc = mol / vol[:, None]

    return conc, mol, vol
