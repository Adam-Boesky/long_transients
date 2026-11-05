"""Read a `Source`'s data from the local database instead of the live APIs.

A `Source` normally assembles itself from ~8 network services: IRSA for ZTF,
WISE, NEOWISE, PTF and Gaia, MAST for Pan-STARRS, SDSS's archive, SPARCL for
DESI, plus cone searches against SIMBAD and AGN-DB. Every one of those is
already mirrored in `Data/db` by the source-database build, so a source that
is in a release can be assembled without touching the network at all.

    from Source_Analysis.local_store import LocalStore
    store = LocalStore()
    src = Source(ra, dec, store=store)

Injected rather than switched on with a flag: the store is one decision made
once, at construction, instead of an `if` inside every property. It also makes
the mixed case expressible -- local light curves with a live TNS check --
without inventing a second flag.

Where it intercepts, and why
----------------------------
Only at *fetching and naming*. `Light_Curve.get_catalog_lc` is split so that
everything from the NaN mask onward -- the Pan-STARRS same-night collapse and
the WISE/NEOWISE visit binning, including the inverse-variance flux
combination -- lives in `_finish_catalog_lc`, which both paths call. The store
hands over a shaped table and re-enters there.

That split is the whole design. An audit of this codebase against the website
found six places where the same quantity was computed twice and the two copies
had drifted: WISE binned by averaging magnitudes in one place and combining
fluxes in another, a colour histogram pairing bands two different ways, a
magnitude error averaged where it should have been combined. Every one came
from a second implementation of shared logic. A local mode that reimplemented
the binning would be the seventh.

The wide column names are derived from `Light_Curve.ALL_BAND_DF`, the same
table the remote path renames into, so the naming cannot drift either.

What is not covered
-------------------
`cutouts` returns None: the database stores stamp *addresses* and rendered
PNGs, not FITS pixels, and `plot_cutout` needs an array plus a WCS. `Source`
falls back to the remote path and says so rather than silently drawing
nothing. Spectra are likewise left remote for now -- the local store has the
arrays but `plot_spectrum` reaches into HDUList internals, so wiring it up
means changing that accessor rather than faking the structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.table import Table

from Source_Analysis.Light_Curve import ALL_BAND_DF
from source_db.paths import DEFAULT_OUT_DIR

# WISE and NEOWISE are not in `ALL_BAND_DF` -- their columns are already
# `w1_mag`..`w4_mag` before any renaming, so there is nothing to map.
_WISE_SURVEYS = ('wise', 'neowise')

# Surveys whose epochs the light-curve table holds. `ztforce`, `zubercal` and
# `custom` are followup products that are not part of a release, so they keep
# going through their existing paths.
LOCAL_LC_SURVEYS = ('ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia')


def catalog_columns(survey: str) -> list[str]:
    """Every `*_mag` column the remote path creates for a catalog.

    The remote path always builds the full set and leaves absent bands NaN
    (`Light_Curve` line ~592 for Pan-STARRS, ~602 for ZTF), and
    `_finish_catalog_lc` then indexes them unconditionally -- the Pan-STARRS
    collapse walks all five bands, `_bin_visit_clusters` walks w1-w4. So the
    shaped table has to carry the whole set whether or not this source has
    data in each, or the shared tail raises `KeyError` on the first band that
    happens to be missing.

    Derived from `ALL_BAND_DF` so it tracks the remote path automatically.
    """
    if survey == 'wise':
        return [f'w{n}_mag' for n in range(1, 5)]
    if survey == 'neowise':
        return [f'w{n}_mag' for n in range(1, 3)]
    return [c for c in ALL_BAND_DF.columns if ALL_BAND_DF[c]['survey'] == survey]


def _wide_column(survey: str, band: str) -> Optional[str]:
    """(survey, band) -> the column `get_catalog_lc` would have produced.

    Inverted from `ALL_BAND_DF` rather than written out again, so a rename on
    the remote side carries over here for free. Gaia is the one that would
    otherwise catch you out: the database stores bands `r` and `b` while the
    analysis columns are `gaia_rp_mag` and `gaia_bp_mag`.
    """
    if survey in _WISE_SURVEYS:
        return f'{band}_mag'
    alias = {('gaia', 'r'): 'rp', ('gaia', 'b'): 'bp'}.get((survey, band), band)
    # Case-insensitive on the band. PTF reports Mould R as `R` while
    # `ALL_BAND_DF` keys it `['r', 'ptf'] -> 'R_mag'`, so an exact match drops
    # every PTF epoch silently -- 15 of them on the first source I tried, and
    # 3.6M across the catalog. This is the same mismatch that had the website
    # drawing PTF grey; no survey has both `r` and `R`, so folding case is
    # safe as well as necessary.
    for column in ALL_BAND_DF.columns:
        col_band, col_survey = ALL_BAND_DF[column]
        if col_survey == survey and str(col_band).lower() == str(alias).lower():
            return column
    # `ALL_BAND_DF` keys Gaia by its own band letters, so try those directly.
    for column in ALL_BAND_DF.columns:
        if column == f'{survey}_{alias}_mag':
            return column
    return None


def shape_catalog_lc(rows: pd.DataFrame, catalog: str, source_uuid: str,
                     ra: Optional[float] = None,
                     dec: Optional[float] = None) -> Table:
    """One catalog's light-curve rows (`band, mjd, mag, magerr, snr`) as the
    table `get_catalog_lc` builds, ready for `Light_Curve._finish_catalog_lc`.

    Split out of `LocalStore.catalog_lc` so bulk builds shape rows through the
    same code rather than a copy of it.
    """
    # One row per epoch, one column pair per band -- the shape the remote
    # path arrives at after its per-catalog pivot. Every band the catalog
    # can have gets a column, NaN where this source has no data, because
    # that is what the remote path produces and what the shared tail
    # assumes.
    frame = pd.DataFrame({'mjd': rows['mjd'].astype(float)})
    nan = np.full(len(rows), np.nan)
    for column in catalog_columns(catalog):
        frame[column] = nan
        frame[column.replace('_mag', '_magerr')] = nan
    if catalog == 'neowise':
        for n in (1, 2):
            frame[f'w{n}_snr'] = nan

    for band in rows['band'].unique():
        column = _wide_column(catalog, str(band))
        if column is None or column not in frame.columns:
            continue
        hit = (rows['band'] == band).to_numpy()
        frame.loc[hit, column] = rows['mag'].to_numpy()[hit]
        frame.loc[hit, column.replace('_mag', '_magerr')] = (
            rows['magerr'].to_numpy()[hit])
        if catalog == 'neowise':
            frame.loc[hit, f'{band}_snr'] = rows['snr'].to_numpy()[hit]

    frame['ra'] = ra if ra is not None else np.nan
    frame['dec'] = dec if dec is not None else np.nan

    # `get_lc` narrows each catalog to its closest object with
    #     mask = cat_lc[f'{catalog}_id'] == cat_lc[f'{catalog}_id'][0]
    # so the column has to exist. A constant, not the real `catalog_id`:
    # the store selected these rows *by source_uuid*, so they are already
    # one source and the filter must be a no-op. The real id varies per
    # epoch for the single-exposure catalogs -- which is exactly why
    # `get_lc` special-cases NEOWISE -- and emitting it would make this
    # mask silently drop most of the light curve.
    frame[f'{catalog}_id'] = str(source_uuid)
    return Table.from_pandas(frame)


class LocalStore:
    """Serves a `Source` from `Data/db` instead of the network."""

    def __init__(self, out_dir: Path | str = DEFAULT_OUT_DIR):
        self.out_dir = Path(out_dir)
        self.lightcurve_dir = self.out_dir / 'light_curves'
        if not self.lightcurve_dir.exists():
            raise FileNotFoundError(
                f'{self.lightcurve_dir} not found -- run '
                f'`python -m source_db.build_lightcurve_table` first')
        self._con = None
        self._release = None

    # `store_src_plots` ships whole `Source` objects to worker processes, which
    # pickles whatever they hold. A DuckDB connection is not picklable, and it
    # should not be shared across a fork anyway -- each worker opens its own on
    # first use.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_con'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._con = None

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def release_id(self) -> Optional[int]:
        """Which release this store is a snapshot of.

        Worth recording on any figure built from it: a source read locally
        reflects the crossmatch as it stood at build time, while a live cone
        search reflects the catalogue today.
        """
        if self._release is None:
            import json
            path = self.out_dir / 'release.json'
            self._release = (json.loads(path.read_text()).get('release_id')
                             if path.exists() else -1)
        return None if self._release == -1 else self._release

    def _duckdb(self):
        """A cursor, not the connection.

        `Light_Curve.get_lc` fans the catalogs out across a ThreadPoolExecutor,
        and a DuckDB connection used from two threads at once does not raise --
        it returns the other query's result, or None. `cursor()` is the
        supported way to fan out, and it is cheap.

        This is the third place in this codebase where one connection shared
        across threads produced silently wrong results rather than an error;
        the API's SQLite handles and the catalog filter engine were the others.
        """
        if self._con is None:
            import duckdb
            self._con = duckdb.connect()
            self._con.execute('SET threads = 4')
        return self._con.cursor()

    # ── light curves ──────────────────────────────────────────────────────

    def catalog_lc(self, catalog: str, source_uuid: Optional[str] = None,
                   ra: Optional[float] = None, dec: Optional[float] = None,
                   radius_arcsec: float = 1.5) -> Optional[Table]:
        """One catalog's epochs, shaped as `get_catalog_lc` shapes them.

        Returns the table *before* `_finish_catalog_lc`, so the caller still
        applies the NaN mask, the Pan-STARRS collapse and the WISE/NEOWISE
        binning. `None` means "this store cannot answer", which the caller
        treats as a reason to fall back rather than as an empty light curve.
        """
        if catalog not in LOCAL_LC_SURVEYS:
            return None
        if source_uuid is None:
            source_uuid = self.resolve_uuid(ra, dec, radius_arcsec)
            if source_uuid is None:
                return None

        con = self._duckdb()
        rows = con.execute(
            f"SELECT band, mjd, mag, magerr, snr "
            f"FROM read_parquet('{self.lightcurve_dir}/uuid_prefix=*/part-0.parquet') "
            f"WHERE source_uuid = ? AND survey = ? ORDER BY mjd",
            [source_uuid, catalog]).fetchdf()
        if rows.empty:
            return Table()
        return shape_catalog_lc(rows, catalog, source_uuid, ra, dec)

    def resolve_uuid(self, ra: float, dec: float,
                     radius_arcsec: float = 1.5) -> Optional[str]:
        """The release's uuid for a position, or None if it holds no such source."""
        if ra is None or dec is None:
            return None
        con = self._duckdb()
        # A box first so the scan is bounded, then the true angular separation.
        half = radius_arcsec / 3600.0
        cos_dec = max(np.cos(np.radians(dec)), 1e-6)
        got = con.execute(
            f"SELECT source_uuid, ra, dec FROM "
            f"read_parquet('{self.out_dir}/sources.parquet') "
            f"WHERE dec BETWEEN ? AND ? AND ra BETWEEN ? AND ?",
            [dec - half, dec + half,
             ra - half / cos_dec, ra + half / cos_dec]).fetchall()
        if not got:
            return None
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        here = SkyCoord(ra, dec, unit='deg')
        there = SkyCoord([g[1] for g in got], [g[2] for g in got], unit='deg')
        seps = here.separation(there).arcsec
        best = int(np.argmin(seps))
        return got[best][0] if seps[best] <= radius_arcsec else None

    # ── the catalog row ───────────────────────────────────────────────────

    def catalog_row(self, source_uuid: Optional[str] = None,
                    ra: Optional[float] = None, dec: Optional[float] = None,
                    radius_arcsec: float = 1.5) -> Optional[Table]:
        """The source's catalog row, as a one-row table.

        This is what makes local mode actually offline. Without it `Source.data`
        falls back to reading the per-field HDF5 catalogs under
        `Data/catalog_results/field_results` -- 301 GB that a laptop running
        off a copy of `Data/db` will not have -- and, on the way, fires an IRSA
        metadata query to work out which field to load.

        The same row, with the same columns, is already in `sources.parquet`.
        """
        if source_uuid is None:
            source_uuid = self.resolve_uuid(ra, dec, radius_arcsec)
            if source_uuid is None:
                return None
        frame = self._duckdb().execute(
            f"SELECT * FROM read_parquet('{self.out_dir}/sources.parquet') "
            f"WHERE source_uuid = ?", [source_uuid]).fetchdf()
        if frame.empty:
            return None
        return Table.from_pandas(frame)

    # ── not covered ───────────────────────────────────────────────────────

    def cutout(self, survey: str, band: str, ra: float, dec: float):
        """Always None: the release stores stamp addresses and PNGs, not FITS."""
        return None

    def sdss_spectrum(self, *args, **kwargs):
        """Always None for now -- see the module docstring."""
        return None

    def desi_spectrum(self, *args, **kwargs):
        """Always None for now -- see the module docstring."""
        return None
