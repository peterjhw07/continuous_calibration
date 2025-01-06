import numpy as np
import pwlf

# Example data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5])

# Create the model
model = pwlf.PiecewiseLinFit(x, y)

# Fit the model with 2 breakpoints (3 segments)
breaks = model.fit(3)

# Predict
y_hat = model.predict(x)

# Breakpoints
print("Breakpoints:", breaks)