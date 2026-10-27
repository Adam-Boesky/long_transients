"""Survey constants shared across the pipeline, the API and the figures.

A leaf module on purpose: it imports nothing, so the places that need these
numbers can take them without dragging in matplotlib (`store_src_plots` sets
up usetex at module scope) or the serving stack.
"""
from __future__ import annotations

# Pan-STARRS 3pi 5-sigma point-source depth per band, substituted for a
# non-detection when computing ZTF - PanSTARRS. Nominal survey depths; the
# real per-skycell depth varies with coverage and seeing.
PSTARR_UPPER_LIM = {'g': 23.3, 'r': 23.2, 'i': 23.1}

# |Δmag sigma| beyond what an envelope grid resolves is reported as this, with a flag.
SIGMA_CAP = 8.0

# Stern et al. (2012, ApJ 753, 30): W1 - W2 >= 0.8 (Vega) selects AGN. The
# colour is only part of the criterion -- it also requires W2 SNR >= 10,
# blend flag nb <= 2 and clean cc_flags; see `source_db.unified_class`.
STERN_W1W2_CUT = 0.8
