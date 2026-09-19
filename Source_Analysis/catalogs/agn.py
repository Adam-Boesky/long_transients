"""Query helpers for the master-merged AGN catalog.

The catalog (~8.1M rows, 564 columns) is stored as Parquet
(`Data/catalogs/master-merged-all-store_v1p0.parquet`, converted from the 10GB
source CSV via DuckDB). Parquet is
columnar and row-group-pruned, so DuckDB queries against it only read the
columns/rows a query actually touches instead of loading the whole catalog
into memory. Regenerate the Parquet file if the source CSV changes:

    duckdb -c "COPY (SELECT * FROM read_csv_auto('Data/master-merged-all-store_v1p0.csv'))
                TO 'Data/master-merged-all-store_v1p0.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)"
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from Source_Analysis.catalogs._base import LocalCatalog

CATALOG_PARQUET = (Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
                   / 'master-merged-all-store_v1p0.parquet')

# Maps the integer catalog IDs stored in AGN-DB's `*_origin` columns (e.g.
# best_class_origin) to the contributing catalog's name, scraped from the "Source
# Catalogs" reference table at https://alessandropeca.com/agndb_pages/source_table_ref.html
# (AGN-DB ships this mapping as index2ds.pickle alongside the encrypted catalog download;
# this is the same mapping, just pulled from the public site instead).
CATALOG_ID_TO_NAME: dict[str, str] = {
    '0': '2MRS', '1': '2QZ', '2': '2SLAQ', '3': '3FGL Fermi cleanups 2', '4': 'belladitta19',
    '5': '3LAC', '6': '3LAC', '7': '3LAC', '8': '4XMM_DR10', '9': 'AGNELA', '10': 'ALMA_decarli',
    '11': 'ATLAS', '12': 'BAHM', '13': 'BASS', '14': 'BAT-105M', '15': 'BGGFC', '16': 'BGGFC',
    '17': 'C-COSM', '18': 'CDFS7', '19': 'ChaMP', '20': 'DEEP', '21': 'DR16Q', '22': 'DUHIZ',
    '23': 'DUHIZ', '24': 'DUz6', '25': 'DUz6', '26': 'ELQS-N', '27': 'ELQS-S', '28': 'FISCBA',
    '29': '4FGL3', '30': 'GL-DB', '31': 'GLIKMAN', '32': 'GaiaUnwise', '33': 'HELLAS2XMM',
    '34': 'HELLAS2XMM', '35': 'IANTEG', '36': 'IBIS', '37': 'J1030', '38': 'IKEDA',
    '39': 'belladitta20', '40': 'LSSA', '41': 'MFJC', '42': 'MFJC', '43': 'MHH', '44': 'MZZ',
    '45': 'NBCKDE', '46': 'NBCKv3', '47': 'OVRLAP', '48': 'OzDES', '49': '4LAC', '50': 'PHILLI',
    '51': 'PS1', '52': 'PS1', '53': 'PS1MAZ', '54': 'QUBNIR', '55': 'PSO', '56': 'RLQ',
    '57': 'S82X', '58': 'QUBRF', '59': 'SDLENS', '60': 'SDSSHI', '61': 'SIX',
    '62': 'SPIDERS DR14', '63': 'SPIDERS DR14', '64': 'SPIN19', '65': 'SQLS', '66': 'SQLS',
    '67': 'SUV', '68': 'SUV', '69': 'SXDF', '70': 'SXDS', '71': 'SXDS', '72': 'UFS', '73': 'UFS',
    '74': 'ULTRA', '75': 'VAQL', '76': 'VDES2', '77': 'VIKING', '78': 'VIPERS', '79': 'VMC',
    '80': 'WARSAW', '81': 'WISEA', '82': 'WOLF1', '83': 'XLSS', '84': 'XMM-XXL', '85': 'XMMSMC',
    '86': 'XMSS', '87': 'XWAS', '88': 'YQLF', '89': 'eHAQ', '90': 'eHAQ', '91': 'AllBRICQS',
    '92': 'z6.51', '93': 'Dart', '94': 'COMP2CAT', '95': 'eFEDS', '96': 'AT20G', '97': 'REDt',
    '98': 'YSZ', '99': 'CSC2.0', '100': 'Quaia', '101': 'HETDEX_HDR4', '102': '4FGL_DR4',
    '103': 'NLSy1_DR17', '104': 'HETDEX_LOFAR', '105': 'LoTSS_DR2', '106': 'Swift_BAT_157',
    '107': 'BASS_DR2', '108': '6dFGS_AGN', '109': 'CatGlobe', '110': 'NuSTAR_NSS80',
    '111': 'SHELLQs', '112': 'Stripe82X_DR3', '113': 'Stripe82XL', '114': 'SDSS_DR19Q',
    '115': 'LAMOST_DR10', '116': 'DESI_DR1', '117': 'eRASS1_merged',
}


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


# ---------------------------------------------------------------------------
# Bulk path
#
# The DuckDB functions above stay the right tool for per-source lookups: they
# read only the columns a query touches, so `Sources.agn_match` can run inside
# a multiprocessing Pool without each worker holding the catalog. For bulk
# crossmatching a whole source table that is ~200x too slow, so the nine
# columns worth matching on are projected into a slim parquet that fits in
# memory behind the same `LocalCatalog` the SDSS/DESI/TNS modules use.
# ---------------------------------------------------------------------------

SLIM_PARQUET = CATALOG_PARQUET.parent / 'agn_db_slim.parquet'

# (source column, output column). Covers what Sources.agn_match asks for, plus
# the redshift provenance companions to best_Z_merged.
_SLIM_COLUMNS: list[tuple[str, str]] = [
    ('RA', 'ra'),
    ('DEC', 'dec'),
    ('best_class_all', 'agn_db_best_class_all'),
    ('best_class_origin', 'agn_db_best_class_origin'),
    ('best_class_sub_all', 'agn_db_best_class_sub_all'),
    ('best_Z_merged', 'agn_db_best_z_merged'),
    ('best_Z_all', 'agn_db_best_z_all'),
    ('best_Z_origin', 'agn_db_best_z_origin'),
    ('star_flag', 'agn_db_star_flag'),
]

_CATEGORICAL = {'agn_db_best_class_all', 'agn_db_best_class_origin',
                'agn_db_best_class_sub_all', 'agn_db_star_flag'}

_CATALOG = LocalCatalog(SLIM_PARQUET, prefix='agn_db')


def build_parquet(out_path: Path = SLIM_PARQUET) -> Path:
    """Project the nine crossmatching columns out of the 564-column catalog.

    DuckDB streams this, so it never holds the 1.6 GB source file in memory.
    """
    selects = ', '.join(f'"{src}" AS "{dst}"' for src, dst in _SLIM_COLUMNS)
    df = duckdb.sql(
        f"SELECT {selects} FROM read_parquet('{CATALOG_PARQUET}') "
        f'WHERE RA IS NOT NULL AND DEC IS NOT NULL'
    ).df()
    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')
    df.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def load_catalog() -> pd.DataFrame:
    """The slim catalog, cached. Read-only -- see `_base`."""
    return _CATALOG.frame()


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest AGN-DB row within `radius_arcsec` of each input position.

    One row per input, in input order. Requires the slim parquet from
    `build_parquet()`; for a single position prefer `cone_search`, which reads
    from disk instead of holding the catalog in memory.
    """
    return _CATALOG.match_nearest(ra, dec, radius_arcsec, columns=columns)


if __name__ == '__main__':
    print(f'Building {SLIM_PARQUET.name} from {CATALOG_PARQUET.name}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
