import sys
import matplotlib.pyplot as plt

sys.path.append('/Users/adamboesky/Research/long_transients')

from Source_Analysis.Sources import Source

src = Source(18.64737035466559, 9.25191865650902, max_arcsec=10)
src.plot_all_cutouts()
plt.show()
