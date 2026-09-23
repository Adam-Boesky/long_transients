"""Per-source filter reasons, reduced from the per-field reject tables.

Why this exists
---------------
The analysis page shows a 3x3 table of *why* a source failed each
(catalog, band) branch. That comes from `{cat}_{band}_filtered_out.ecsv`, which
`filter_fields.py` writes per field -- roughly 535 MB per field, ~200 GB across
a full run. Those files only ever live on the cluster, and they are deleted
once `combined/` is built.

So the reduction has to happen while the rejects and the survivors are still
side by side: 200 GB of reject rows collapse to nine short strings per
surviving source, about 20 MB in total. After that the columns ride through
`enrich_combined_tabs` into `sources.parquet` and the serving database with no
further plumbing.

Matching mirrors `Source.get_filtered_out_info`: nearest reject within
`max_arcsec` (1.5" by default, `Source.max_arcsec`), `'-'` when the source is
absent from a table, `'N/A'` when the table itself is missing.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

# Must match Source_Analysis.Sources.CATALOG_INT_MAP. Duplicated rather than
# imported because Sources pulls in matplotlib, astroquery and sparcl, which
# are not worth loading inside a cluster combine job.
CATALOG_INT_MAP = {'in_both': 0, 'in_ztf': 1, 'in_pstarr': 2}
BANDS = ('g', 'r', 'i')

# Source.max_arcsec
DEFAULT_MATCH_ARCSEC = 1.5

NOT_FILTERED = '-'
NO_TABLE = 'N/A'

# Only these are needed; the tables also carry fieldid/ccdid/qid.
_USE_COLUMNS = ['ra', 'dec', 'filter']


def column_name(cat_name: str, band: str) -> str:
    return f'filtered_out_{cat_name}_{band}'


ALL_COLUMNS = [column_name(c, b) for c in CATALOG_INT_MAP for b in BANDS]


def _read_rejects(path: str) -> Optional[pd.DataFrame]:
    """Load `ra`, `dec`, `filter` from a reject table.

    Read with pandas rather than `Table.read`: these are 0.9-1.5M row ECSV
    files and the full astropy path costs ~3x more, which matters when it runs
    9 times per field across several hundred fields. The ECSV header is a
    comment block followed by a plain space-separated table, so the column
    names are taken from the first non-comment line rather than assumed.
    """
    if not os.path.exists(path):
        return None
    header = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                header = line.split()
                break
    if header is None:
        return None                      # header only, no data
    missing = [c for c in _USE_COLUMNS if c not in header]
    if missing:
        raise ValueError(f'{path} is missing {missing}; got {header}')
    return pd.read_csv(path, comment='#', sep=r'\s+',
                       usecols=[header.index(c) for c in _USE_COLUMNS],
                       names=_USE_COLUMNS, header=0)


def reasons_for(
    ra: Iterable[float],
    dec: Iterable[float],
    field_dir: str,
    max_arcsec: float = DEFAULT_MATCH_ARCSEC,
) -> dict[str, np.ndarray]:
    """The nine filter-reason columns for one field's surviving sources."""
    ra = np.asarray(ra, dtype='float64')
    dec = np.asarray(dec, dtype='float64')
    out = {c: np.full(len(ra), NOT_FILTERED, dtype=object) for c in ALL_COLUMNS}
    if len(ra) == 0:
        return out
    survivors = SkyCoord(ra, dec, unit='deg')

    for cat_name, cat_idx in CATALOG_INT_MAP.items():
        for band in BANDS:
            col = column_name(cat_name, band)
            rejects = _read_rejects(
                os.path.join(field_dir, f'{cat_idx}_{band}_filtered_out.ecsv'))
            if rejects is None:
                out[col][:] = NO_TABLE
                continue
            if len(rejects) == 0:
                continue
            idx, sep, _ = match_coordinates_sky(
                survivors,
                SkyCoord(rejects['ra'].to_numpy(), rejects['dec'].to_numpy(),
                         unit='deg'))
            hit = sep < max_arcsec * u.arcsec
            if hit.any():
                out[col][hit] = rejects['filter'].to_numpy()[idx[hit]]
    return out


def attach(table, field_dir: str,
           max_arcsec: float = DEFAULT_MATCH_ARCSEC):
    """Add the nine columns to one field's survivor table, in place."""
    cols = reasons_for(table['ra'], table['dec'], field_dir, max_arcsec)
    for name, values in cols.items():
        table[name] = values.astype(str)
    return table
