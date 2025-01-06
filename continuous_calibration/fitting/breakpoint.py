import numpy as np
import pwlf


def get_breakpoints(x, y, guess=None, bounds=None):
    bounds_lower_index = x[x <= bounds[0]]
    model = pwlf.PiecewiseLinFit(x[].tolist(), y[].tolist())

    # Fit the model with 2 breakpoints (3 segments)
    if guess:
        breaks = model.fit_guess([guess])
    else:
        breaks = model.fit(2)


    # Predict
    y_hat = model.predict(x)

    # Breakpoints
    print("Breakpoints:", breaks)
