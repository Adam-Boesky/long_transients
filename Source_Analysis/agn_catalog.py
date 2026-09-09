"""Query helpers for the master-merged AGN catalog.

The catalog (~8.1M rows, 564 columns) is stored as Parquet
(`Data/master-merged-all-store_v1p0.parquet`, converted from the 10GB source
CSV via DuckDB — see `Data/master-merged-all-store_v1p0.csv`). Parquet is
columnar and row-group-pruned, so DuckDB queries against it only read the
columns/rows a query actually touches instead of loading the whole catalog
into memory. Regenerate the Parquet file if the source CSV changes:

    duckdb -c "COPY (SELECT * FROM read_csv_auto('Data/master-merged-all-store_v1p0.csv'))
                TO 'Data/master-merged-all-store_v1p0.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)"
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

CATALOG_PARQUET = Path(__file__).resolve().parents[1] / 'Data' / 'master-merged-all-store_v1p0.parquet'


def query_box(
    ra_min: float, ra_max: float, dec_min: float, dec_max: float,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return all catalog rows with RA in [ra_min, ra_max] and Dec in [dec_min, dec_max]."""
    cols = ', '.join(f'"{c}"' for c in columns) if columns else '*'
    return duckdb.sql(f"""
        SELECT {cols} FROM read_parquet('{CATALOG_PARQUET}')
        WHERE RA BETWEEN {ra_min} AND {ra_max} AND DEC BETWEEN {dec_min} AND {dec_max}
    """).df()


def cone_search(
    ra: float, dec: float, radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return catalog rows within `radius_arcsec` of (ra, dec), sorted by separation.

    Pre-filters with a padded RA/Dec box (cheap, lets DuckDB skip most of the
    catalog) then applies an exact angular-separation cut on the small
    remainder.
    """
    radius_deg = radius_arcsec / 3600.0
    dec_pad = radius_deg
    ra_pad = radius_deg / max(math.cos(math.radians(dec)), 1e-6)

    box_cols = list(columns) if columns else None
    if box_cols is not None and 'RA' not in box_cols:
        box_cols = ['RA', 'DEC'] + box_cols
    box = query_box(ra - ra_pad, ra + ra_pad, dec - dec_pad, dec + dec_pad, box_cols)
    if box.empty:
        return box

    sep = SkyCoord(ra=box['RA'].values * u.deg, dec=box['DEC'].values * u.deg).separation(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ).arcsec
    box = box.loc[sep <= radius_arcsec].copy()
    box['sep_arcsec'] = sep[sep <= radius_arcsec]
    return box.sort_values('sep_arcsec').reset_index(drop=True)
