from ztfquery import lightcurve
from astropy.table import Table

lightcurve_tab = Table.from_pandas(
    lightcurve.LCQuery.from_position(
        3.38776,
        9.22086,
        1.5,  # example query radius in arcsec, adjust as needed
        BAD_CATFLAGS_MASK=6141,
    ).data
)

breakpoint()
print(lightcurve_tab)
