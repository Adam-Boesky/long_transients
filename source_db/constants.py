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

# The association radius every per-source lookup shares: the light-curve
# pulls (ZTF, NEOWISE, PanSTARRS, SDSS) and the catalog crossmatches (WISE,
# SDSS, DESI, TNS, SIMBAD, AGN-DB). Each of those modules declares its own
# copy, which `tests/api/test_api_sources.py` holds to this value.
MATCH_RADIUS_ARCSEC = 1.5

# The radii within which data is gathered for a source, in the order the
# stamps list them. The numbers are copies of the constants the pipeline
# modules own (named in `test_every_region_matches_its_module`), kept here so
# the API can hand them to the page without importing those modules.
QUERY_REGIONS = (
    {'key': 'lightcurve', 'label': 'light curves',
     'help': 'ZTF, NEOWISE, PanSTARRS and SDSS epochs are matched within this radius',
     'radius_arcsec': MATCH_RADIUS_ARCSEC},
    {'key': 'catalog', 'label': 'catalog crossmatch',
     'help': 'WISE, SDSS, DESI, TNS, SIMBAD and AGN-DB are matched within this radius',
     'radius_arcsec': MATCH_RADIUS_ARCSEC},
    {'key': 'gaia', 'label': 'Gaia',
     'help': 'the Gaia match used for the parallax and proper-motion cuts',
     'radius_arcsec': 5.0},
    {'key': 'wise_beam', 'label': 'WISE beam',
     'help': 'neighbours inside this radius blend into the same WISE source',
     'radius_arcsec': 8.0},
    {'key': 'erosita', 'label': 'eROSITA',
     'help': 'the eROSITA DR2 crossmatch radius',
     'radius_arcsec': 10.0},
)

# Stern et al. (2012, ApJ 753, 30): W1 - W2 >= 0.8 (Vega) selects AGN. The
# colour is only part of the criterion -- it also requires W2 SNR >= 10,
# blend flag nb <= 2 and clean cc_flags; see `source_db.unified_class`.
STERN_W1W2_CUT = 0.8
