"""
Continuous Calibration
A method for obtaining a calibration curve using a continuous addition of analyte to the monitored solution.
"""

# Imports
from continuous_calibration.gen import gen
from continuous_calibration.apply import apply
from continuous_calibration.prep.raw_import import raw_import
from continuous_calibration.prep.export import export_xlsx


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