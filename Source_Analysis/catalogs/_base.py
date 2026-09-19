"""Shared crossmatch machinery for the local catalog modules.

`sdss_catalog`, `desi_catalog` and `tns_catalog` all do the same thing: hold a
slim parquet in memory, build a KD-tree over its coordinates once, and match
source positions against it. This holds that logic so the three stay
semantically identical -- they previously carried three copies of it, and
three copies of the same bugs.

Catalog frames are cached and handed back live rather than copied: DESI's is
~8.5 GB, so copying per call costs more than the mutation hazard. Treat frames
returned by `frame()` as read-only; mutating one corrupts every later query and
desynchronises row order from the cached KD-tree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, MaskedColumn, Table


def _nullable(df: pd.DataFrame) -> pd.DataFrame:
    """Recast columns so unmatched rows can hold nulls without changing dtype.

    Plain int64 upcasts to float64 under `mask`, which silently corrupts values
    above 2**53 -- 98.9% of DESI TARGETIDs. Nullable Int64/boolean hold pd.NA
    directly. Applied unconditionally so the output schema is the same whether
    or not anything went unmatched.
    """
    out = {}
    for name, col in df.items():
        if isinstance(col.dtype, pd.CategoricalDtype):
            out[name] = col.astype(object)
        elif pd.api.types.is_integer_dtype(col.dtype):
            out[name] = col.astype('Int64')
        elif pd.api.types.is_bool_dtype(col.dtype):
            out[name] = col.astype('boolean')
        else:
            out[name] = col
    return pd.DataFrame(out, index=df.index)


def to_astropy_table(df: pd.DataFrame) -> Table:
    """Convert a crossmatch frame to an astropy Table with explicit dtypes.

    `Table.from_pandas` infers, and infers badly here. A pandas `string` column,
    or an object column holding pd.NA, becomes an astropy *object* column, which
    ECSV serialises as the repr of an empty masked array -- so a batch in which
    a catalog matched nothing writes garbage to the file. Integers matter too: a
    float64 detour silently rounds DESI TARGETIDs, 98.9% of which exceed 2**53.

    Empty strings are kept unmasked and distinct from nulls. ECSV cannot tell
    those apart on its own, so the writer must pass
    ``serialize_method={col: 'data_mask'}`` for the columns `text_column_names`
    reports.

    Float columns are written as plain NaN rather than masked. ECSV stores a
    masked value as an empty field, and on read astropy refills it with the
    dtype default -- 0.0 for a float -- so `np.asarray` on a masked magnitude
    column silently reports missing data as a real zero, and `np.isfinite`
    calls it finite. NaN has none of that ambiguity and reads back correctly
    without the caller needing `.filled()`. Integer and boolean columns have no
    NaN to fall back on, so they stay masked and do need `.filled()`.
    """
    cols = []
    for name in df.columns:
        s = df[name]
        na = s.isna().to_numpy()
        if pd.api.types.is_integer_dtype(s.dtype):
            cols.append(MaskedColumn(s.fillna(-1).to_numpy(dtype=np.int64), mask=na, name=name))
        elif pd.api.types.is_bool_dtype(s.dtype):
            cols.append(MaskedColumn(s.fillna(False).to_numpy(dtype=bool), mask=na, name=name))
        elif pd.api.types.is_float_dtype(s.dtype):
            cols.append(Column(s.to_numpy(dtype=float), name=name))
        else:
            lengths = s.dropna().astype(str).str.len()
            width = max(1, int(lengths.max())) if len(lengths) else 1
            vals = s.fillna('').astype(str).to_numpy(dtype=f'U{width}')
            cols.append(MaskedColumn(vals, mask=na, name=name))
    return Table(cols)


def text_column_names(table: Table) -> list[str]:
    """Columns needing ``serialize_method='data_mask'`` to survive an ECSV round
    trip with empty strings intact."""
    return [c for c in table.colnames if table[c].dtype.kind in ('U', 'S')]


class LocalCatalog:
    """A slim parquet catalog with cached coordinates for cone searching.

    `row_filter` is applied when `filtered=True` (the default) to restrict the
    catalog to its preferred rows -- primary spectra, science targets and so on.
    """

    def __init__(
        self,
        parquet_path: Path,
        prefix: str,
        row_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    ) -> None:
        self.parquet_path = parquet_path
        self.prefix = prefix
        self.row_filter = row_filter
        self._frames: dict[bool, pd.DataFrame] = {}
        self._coords: dict[bool, SkyCoord] = {}

    def frame(self, filtered: bool = True) -> pd.DataFrame:
        """The catalog as a DataFrame. Cached; read-only by convention."""
        if filtered not in self._frames:
            df = pd.read_parquet(self.parquet_path)
            if filtered and self.row_filter is not None:
                df = df[self.row_filter(df)].reset_index(drop=True)
            self._frames[filtered] = df
        return self._frames[filtered]

    def coords(self, filtered: bool = True) -> SkyCoord:
        """Cached SkyCoord; building it also builds the KD-tree astropy reuses."""
        if filtered not in self._coords:
            df = self.frame(filtered)
            self._coords[filtered] = SkyCoord(df['ra'].values * u.deg, df['dec'].values * u.deg)
        return self._coords[filtered]

    def _payload_columns(self, df: pd.DataFrame, columns: Optional[list[str]]) -> list[str]:
        available = [c for c in df.columns if c not in ('ra', 'dec')]
        if columns is None:
            return available
        if isinstance(columns, str):
            raise TypeError("columns must be a list of names, not a single string")
        unknown = set(columns) - set(available)
        if unknown:
            raise KeyError(f'unknown column(s) for {self.prefix}: {sorted(unknown)}')
        return [c for c in available if c in columns]

    def match_nearest(
        self,
        ra: Sequence[float],
        dec: Sequence[float],
        radius_arcsec: float,
        filtered: bool = True,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Nearest catalog row within `radius_arcsec` of each input position.

        One row per input, in input order. Inputs with no match -- or with
        non-finite coordinates -- come back with `<prefix>_matched` False and a
        null payload, so a missing value is never mistaken for a real one.
        """
        ra = np.atleast_1d(np.asarray(ra, dtype=float))
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        if ra.shape != dec.shape:
            raise ValueError(f'ra and dec must be the same length, got {ra.shape} and {dec.shape}')
        p = self.prefix

        cat = self.frame(filtered)
        keep = self._payload_columns(cat, columns)

        n = len(ra)
        idx = np.zeros(n, dtype=int)
        sep = np.full(n, np.nan)
        # astropy raises on NaN coordinates, which would lose the whole batch, so
        # non-finite inputs are held out and reported as unmatched.
        finite = np.isfinite(ra) & np.isfinite(dec)
        if finite.any():
            f_idx, f_sep, _ = SkyCoord(ra[finite] * u.deg, dec[finite] * u.deg).match_to_catalog_sky(
                self.coords(filtered)
            )
            idx[finite] = f_idx
            sep[finite] = f_sep.arcsec
        matched = np.isfinite(sep) & (sep <= radius_arcsec)

        out = _nullable(cat.iloc[idx][keep].reset_index(drop=True))
        out = out.mask(pd.Series(~matched, index=out.index), axis=0)
        out.insert(0, f'{p}_matched', matched)
        out.insert(1, f'{p}_sep_arcsec', np.where(matched, sep, np.nan))
        # The matched catalog position, prefixed so it can't collide with the
        # caller's own ra/dec when this is joined onto a source table.
        out.insert(2, f'{p}_ra', np.where(matched, cat['ra'].values[idx], np.nan))
        out.insert(3, f'{p}_dec', np.where(matched, cat['dec'].values[idx], np.nan))
        return out

    def within(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        filtered: bool = True,
    ) -> pd.DataFrame:
        """Every catalog row within `radius_arcsec` of one position, nearest first."""
        p = self.prefix
        cat = self.frame(filtered)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return cat.iloc[:0].assign(**{f'{p}_sep_arcsec': []})

        i_cat, _, sep2d, _ = SkyCoord([ra] * u.deg, [dec] * u.deg).search_around_sky(
            self.coords(filtered), radius_arcsec * u.arcsec
        )
        out = cat.iloc[i_cat].copy()
        out[f'{p}_sep_arcsec'] = sep2d.arcsec
        return out.sort_values(f'{p}_sep_arcsec').reset_index(drop=True)
