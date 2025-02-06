import sys

sys.path.append('/Users/adamboesky/Research/long_transients')

from Source_Analysis.Light_Curve import Light_Curve
lc = Light_Curve(352.55095852, -24.80405223, query_in_parallel=False)
lc.lc

breakpoint()

# lc.lc['ztf_r_mag'][~lc.lc['ztf_r_mag'].mask]
