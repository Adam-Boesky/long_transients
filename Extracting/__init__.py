import warnings
from astropy.wcs import FITSFixedWarning

# Ignore FITSFixedWarnings
warnings.simplefilter('ignore', FITSFixedWarning)