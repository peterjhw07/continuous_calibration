"""CC Limit of Detection Calculation"""

# Calculate limit of detection
def get_lod(conc, intensity, std, stds=3.0, model=None, params=None):
    lod_idx, lod_conc = [], []
    for spec in range(conc.shape[1]):
        int_lod = intensity[:, spec][0] + stds * std[:, spec][0]
        if model and params:
            lod_idx.append(None)
            lod_conc.append(model(int_lod, *list(params[spec].values())))
        else:
            lod_idx.append([intensity[:, spec] - int_lod >= 0][0])
            lod_conc.append(conc[lod_idx[spec], spec])

    return lod_idx, lod_conc
