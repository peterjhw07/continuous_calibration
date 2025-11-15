# Continuous Calibration

Continuous Calibration (continuous_calibration) is a Pythonic code for processing experimental continuous calibration data. The code is also implemented in a web application: http://www.catacycle.com/cc.

A manuscript describing the code and its applications is pre-printed at DOI:10.26434/chemrxiv-2025-409m2. Please cite this manuscript should this code be used in your work.

## Installation

The code can be downloaded as a package for offline use using the conventional github download tools.

## Usage

Various examples uses of continuous_calibration are featured under "examples". Tutorial videos on usage are a work in progress.

Use help(function) for documentation for the principle functions below.

### Principle functions

```python
# Import package
import continuous_calibration as cc

# Generate calibration curve
cc.gen(**kwargs)

# Apply calibration curve
cc.apply(**kwargs)
```
