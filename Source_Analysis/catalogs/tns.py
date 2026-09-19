"""Query helpers for the local TNS (Transient Name Server) object dump.

`tns_public_objects.csv` is only ~150k rows; most of its 78 MB is the free-text
`reporters` column. Dropping that and the bibcode columns leaves a few MB of
parquet. Matching lives in `_base.LocalCatalog`, shared with the SDSS and DESI
modules.

Unlike SDSS DR17 and DESI DR1, which are frozen data releases, TNS grows daily
-- so this is the one local catalog that goes stale. Refreshing it means
re-downloading the dump into Data/ and re-running `build_parquet()`. A TNS
`tns_marker` user-agent is required for the download, but a *user* marker needs
no API key:

    curl -X POST -H 'user-agent: tns_marker{"tns_id":ID,"type":"user","name":"NAME"}' \
      https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip \
      > tns_public_objects.csv.zip
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from astropy.time import Time

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CATALOG_CSV = _DATA_DIR / 'raw' / 'tns_public_objects.csv'
CATALOG_PARQUET = _DATA_DIR / 'tns_objects_slim.parquet'

# `reporters` and the ADS bibcode columns are dropped -- free text, and most of
# the CSV's size.
_COLUMNS: list[tuple[str, str]] = [
    ('objid', 'tns_objid'),
    ('name_prefix', 'tns_prefix'),
    ('name', 'tns_designation'),
    ('ra', 'ra'),
    ('declination', 'dec'),
    ('redshift', 'tns_redshift'),
    ('type', 'tns_type'),
    ('discoverydate', 'tns_discoverydate'),
    ('discoverymag', 'tns_discoverymag'),
    ('filter', 'tns_discovery_filter'),
    ('source_group', 'tns_source_group'),
    ('reporting_group', 'tns_reporting_group'),
    ('internal_names', 'tns_internal_names'),
    ('lastmodified', 'tns_lastmodified'),
]

_CATEGORICAL = {'tns_prefix', 'tns_type', 'tns_source_group',
                'tns_reporting_group', 'tns_discovery_filter'}

# Inverse of the build-time rename. Columns are prefixed in the parquet so they
# survive being joined onto a source table, but callers wanting a single object's
# record generally want TNS's own field names.
NATIVE_NAMES: dict[str, str] = {dst: src for src, dst in _COLUMNS}

# Every TNS row is a real object, so there is nothing to filter down to.
_CATALOG = LocalCatalog(CATALOG_PARQUET, prefix='tns')


def _read_dump(csv_path: Path) -> tuple[pd.DataFrame, Optional[str]]:
    """Read a staged TNS dump, tolerating the leading creation-timestamp line.

    Dumps downloaded from TNS carry the creation date/time on the first line and
    the column headers on the second. Some older copies here had that line
    stripped before the file was saved, so sniff for it rather than assume.
    """
    with open(csv_path, encoding='utf-8', errors='replace') as fh:
        first_line = fh.readline()
    stamped = not first_line.lstrip().startswith('"objid"')
    df = pd.read_csv(csv_path, skiprows=1 if stamped else 0,
                     usecols=[src for src, _ in _COLUMNS], low_memory=False)
    return df, first_line.strip() if stamped else None


def build_parquet(csv_path: Path = CATALOG_CSV, out_path: Path = CATALOG_PARQUET) -> Path:
    """Convert the TNS CSV dump into the slim parquet used for crossmatching."""
    df, created = _read_dump(csv_path)
    if created is not None:
        print(f'  dump created {created}')
    df = df.rename(columns=dict(_COLUMNS))[[dst for _, dst in _COLUMNS]]

    n_before = len(df)
    df = df[np.isfinite(df['ra']) & np.isfinite(df['dec'])].reset_index(drop=True)
    if len(df) != n_before:
        print(f'  dropped {n_before - len(df)} rows with missing coordinates')

    # 'AT2024yhx' rather than prefix and designation in separate columns, since
    # that is how TNS objects are actually named.
    df['tns_name'] = (df['tns_prefix'].fillna('').astype(str)
                      + df['tns_designation'].fillna('').astype(str)).replace('', None)

    # Discovery epoch as MJD, to match the units used everywhere else in the repo.
    dt = pd.to_datetime(df['tns_discoverydate'], format='mixed', errors='coerce')
    mjd = np.full(len(df), np.nan)
    ok = dt.notna().values
    mjd[ok] = Time(dt[ok].values).mjd
    df['tns_discovery_mjd'] = mjd

    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')

    df.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def load_catalog() -> pd.DataFrame:
    """The slim TNS catalog, cached. Read-only -- see `_local_catalog`."""
    return _CATALOG.frame()


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest TNS object within `radius_arcsec` for each input position."""
    return _CATALOG.match_nearest(ra, dec, radius_arcsec, columns=columns)


def cone_search(ra: float, dec: float, radius_arcsec: float = 2.0) -> pd.DataFrame:
    """All TNS objects within `radius_arcsec` of one position, nearest first."""
    return _CATALOG.within(ra, dec, radius_arcsec)


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name} from {CATALOG_CSV.name}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)')
    print(f'Newest TNS entry in this dump: {load_catalog()["tns_lastmodified"].max()}')
