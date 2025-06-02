import numpy as np
from pygam import LinearGAM, s
import scipy.optimize as optimize
from scipy.signal import savgol_filter
from continuous_calibration import prep
from continuous_calibration.fitting import regression


def savgol_smooth(x, y, window_size='len/5', poly_order=1):
    if isinstance(window_size, str):
        window_size = int(len(x) / 5)
    return x[:, 0], savgol_filter(y[:, 0], window_size, poly_order)


# Smooth data
def smooth(data, smooth_eq=None, intercept=False, sg_win=1):
    data.smooth_model, data.smooth_intercept, data.smooth_model_inverse = [], [], []
    if smooth_eq is not None:
        data.smooth_intensity = np.zeros(data.avg_intensity.shape)
        if 'mono' in smooth_eq.lower() or 'ton' in smooth_eq.lower() or 'con' in smooth_eq.lower():
            if 'mono' in smooth_eq.lower() or 'ton' in smooth_eq.lower():
                constraints = 'monotonic_inc'
            else:
                constraints = 'concave'
            for i in range(data.avg_intensity.shape[1]):
                gam = LinearGAM(s(0, constraints=constraints), fit_intercept=intercept).fit(data.avg_conc[:, i],
                                                                                     data.avg_intensity[:, i])
                data.smooth_intensity[:, i] = gam.predict(data.avg_conc[:, i])
                data.smooth_intercept.append(data.smooth_intensity[0, i])
                data.smooth_model.append(gam)
        else:
            for i in range(data.avg_intensity.shape[1]):
                data.smooth_intensity[:, i] = savgol_filter(data.avg_intensity[:, i], sg_win, 1)
    return data