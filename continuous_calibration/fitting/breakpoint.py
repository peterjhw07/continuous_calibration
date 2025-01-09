import numpy as np
import pwlf
import ruptures as rpt


def get_breakpoints(t, intensity, t_cont, guesses=None, bounds=None):
    breaks = []
    for spec in range(intensity.shape[1]):
        for i in range(len(guesses[spec])):
            guess_lower_index = np.where(t >= guesses[spec][i] - bounds)[0][0]
            for limit in np.where((t <= guesses[spec][i] + bounds) & (t >= guesses[spec][i]))[0]:
                model = pwlf.PiecewiseLinFit(t[guess_lower_index:limit].tolist(), intensity[guess_lower_index:limit, spec].tolist())
                # Fit the model with 2 breakpoints (3 segments)
                breaks.append(model.fit_guess([guesses[spec][i]])[1])
            t_cont[spec][i] = max(breaks)

    # Predict
    # intensity_hat = model.predict(t)

    return t_cont
