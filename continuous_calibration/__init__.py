"""
Continuous Calibration
Continuous Calibration is a method for obtaining a calibration curve
using a continuous addition of analyte to the measured solution.
"""

# Imports
from continuous_calibration.calc import run
from continuous_calibration.raw_import import raw_import


# Handle versioneer
#from ._version import get_versions
#versions = get_versions()
#__version__ = versions['version']
#__git_revision__ = versions['full-revisionid']
#del get_versions, versions
#
#from ._version import get_versions
#__version__ = get_versions()['version']
#del get_versions