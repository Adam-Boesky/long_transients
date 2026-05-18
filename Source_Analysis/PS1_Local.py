import sys
import logging
import numpy as np
import pandas as pd
import healpy as hp
import duckdb

from pathlib import Path
from typing import List, Optional, Tuple
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.append('/n/home04/aboesky/berger/long_transients')

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default path to the HEALPix-partitioned Parquet dataset built by
# panstarrs_ingest.py.  Each subdirectory hpix32=N/ holds one or more
# Parquet files for pixel N (nested scheme, nside=32).
DEFAULT_PARQUET_DIR = Path(
    '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/panstarrs/parquet'
)

# Columns returned by every query method (band-specific names filled at call time)
_BASE_OUT_COLS = ['objID', 'raMean', 'decMean', 'qualityFlag', 'objInfoFlag']
_BAND_OUT_COLS = ['KronMag', 'KronMagErr', 'PSFMag', 'PSFMagErr',
                  'psfLikelihood', 'infoFlag', 'infoFlag2']

VALID_BANDS = frozenset('grizy')

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PanSTARRSLocal
# ---------------------------------------------------------------------------

class PanSTARRSLocal:
    """Fast sky-region queries against a local PS1 DR2 stacked catalog.

    The catalog is stored as a Hive-partitioned Parquet dataset where each
    directory ``hpix32=N/`` holds objects whose raMean/decMean maps to HEALPix
    pixel N (nside=32, nested scheme).  DuckDB is used as the query engine;
    before any SQL scan it filters on ``hpix32 IN (...)`` so only the relevant
    Parquet partitions are opened — typically 4–20 files out of 12 288.

    Build the dataset first with::

        python panstarrs_download.py   # fetch from MAST (~3–8 h, full sky)
        python panstarrs_ingest.py     # convert to partitioned Parquet

    Parameters
    ----------
    parquet_dir:
        Root directory of the Hive-partitioned Parquet dataset.
    nside:
        HEALPix nside used when the dataset was partitioned (default 32).
        Must match the nside used during ingestion.
    """

    def __init__(
        self,
        parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
        nside: int = 32,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.nside = nside

        if not self.parquet_dir.exists():
            raise FileNotFoundError(
                f'Parquet dataset not found at {self.parquet_dir}.  '
                'Run panstarrs_download.py then panstarrs_ingest.py first.'
            )

        # Persistent in-memory DuckDB connection — one connection per instance.
        # Using ':memory:' avoids stale on-disk state across runs.
        self._con = duckdb.connect(database=':memory:')
        self._build_view()
        log.info('PanSTARRSLocal ready: %s (nside=%d)', self.parquet_dir, nside)

    def __del__(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_view(self) -> None:
        """Register the Parquet dataset as a DuckDB view named ``ps1``.

        hive_partitioning=true tells DuckDB to parse ``hpix32=N`` from the
        directory names and expose it as a filterable integer column.  When a
        query contains ``WHERE hpix32 IN (...)``, DuckDB skips all other
        partition directories without opening them.
        """
        glob = str(self.parquet_dir / '**' / '*.parquet')
        self._con.execute(f"""
            CREATE OR REPLACE VIEW ps1 AS
            SELECT * FROM read_parquet('{glob}', hive_partitioning = true)
        """)

    def _tile_pixels(
        self,
        ra_min: float, ra_max: float,
        dec_min: float, dec_max: float,
        buffer: float,
    ) -> np.ndarray:
        """Return HEALPix pixel indices that overlap a buffered RA/Dec tile.

        HEALPix partition pruning strategy
        -----------------------------------
        For non-wrapping tiles (ra_min < ra_max) we use ``query_polygon``
        with the four buffered corners of the rectangle.  This is exact to
        within one pixel width and avoids over-fetching.

        For tiles that cross RA = 0 (ra_min > ra_max after applying the
        requested buffer) ``query_polygon`` cannot span the RA = 0 boundary
        cleanly — we fall back to ``query_disc`` centred on the midpoint with
        a radius equal to the tile's half-diagonal plus buffer.  The disc
        over-selects by at most a few pixels at the corners, which is
        acceptable because the exact RA/Dec box filter is always applied
        downstream in SQL.

        Parameters
        ----------
        ra_min, ra_max : float
            RA bounds in degrees.  ra_min > ra_max signals a wrap-around tile.
        dec_min, dec_max : float
            Dec bounds in degrees.
        buffer : float
            Extra margin (degrees) added to all edges before computing pixels.
        """
        dec_lo = max(-90.0, dec_min - buffer)
        dec_hi = min( 90.0, dec_max + buffer)
        wraps  = (ra_min > ra_max)  # tile crosses RA = 0

        if not wraps:
            ra_lo = max(  0.0, ra_min - buffer)
            ra_hi = min(360.0, ra_max + buffer)
            # Unit-sphere vectors for the four buffered rectangle corners
            vecs = np.array([
                hp.ang2vec(ra_lo, dec_lo, lonlat=True),
                hp.ang2vec(ra_hi, dec_lo, lonlat=True),
                hp.ang2vec(ra_hi, dec_hi, lonlat=True),
                hp.ang2vec(ra_lo, dec_hi, lonlat=True),
            ])
            return hp.query_polygon(self.nside, vecs, nest=True, inclusive=True)

        # Wrap-around: use a bounding disc centred on the tile midpoint
        ra_span    = ra_max + 360.0 - ra_min   # total RA extent (degrees)
        ra_center  = (ra_min + ra_span / 2.0) % 360.0
        dec_center = (dec_min + dec_max) / 2.0
        # Half-diagonal of the tile in angular distance, accounting for
        # cos(dec) compression of RA differences at the tile centre
        half_diag = np.hypot(
            ra_span / 2.0 * np.cos(np.radians(dec_center)),
            (dec_max - dec_min) / 2.0,
        )
        radius_deg = half_diag + buffer
        vec = hp.ang2vec(ra_center, dec_center, lonlat=True)
        return hp.query_disc(self.nside, vec, np.radians(radius_deg),
                             nest=True, inclusive=True)

    def _disc_pixels(self, ra: float, dec: float, radius_deg: float) -> np.ndarray:
        """Return HEALPix pixels overlapping a disc centred at (ra, dec).

        Uses ``query_disc`` with ``inclusive=True`` so border pixels that
        partially overlap the disc are included.  The exact angular separation
        filter is applied afterwards in Python.
        """
        vec = hp.ang2vec(ra, dec, lonlat=True)
        return hp.query_disc(self.nside, vec, np.radians(radius_deg),
                             nest=True, inclusive=True)

    def _output_cols(self, band: str) -> List[str]:
        """Build the ordered list of output column names for a given band."""
        return _BASE_OUT_COLS + [f'{band}{c}' for c in _BAND_OUT_COLS]

    def _build_sql(
        self,
        pixels: np.ndarray,
        band: str,
        ra_filter_sql: str,
        dec_filter_sql: str,
        extra_where: str = '',
    ) -> str:
        """Return a DuckDB SQL query that prunes partitions, filters, and deduplicates.

        Deduplication pattern
        ----------------------
        ROW_NUMBER() OVER (PARTITION BY objID ORDER BY primaryDetection DESC)
        mirrors a SQL "keep the primary detection" pattern: if any row has
        primaryDetection = 1, that row gets rn = 1 and is the one selected.
        If none have primaryDetection = 1, the row with the highest value is
        kept.  This handles edge cases in the PS1 stacked catalog where some
        objects have no flagged primary detection.
        """
        pixel_list = ', '.join(map(str, pixels.tolist()))
        select_cols = ', '.join(self._output_cols(band))
        return f"""
            WITH ranked AS (
                SELECT
                    {select_cols},
                    primaryDetection,
                    nStackDetections,
                    nDetections,
                    ROW_NUMBER() OVER (
                        PARTITION BY objID
                        ORDER BY primaryDetection DESC
                    ) AS _rn
                FROM ps1
                WHERE hpix32 IN ({pixel_list})
                  AND (nStackDetections > 0 OR nDetections > 1)
                  AND ({band}infoFlag2 & 4) = 0
                  AND {ra_filter_sql}
                  AND {dec_filter_sql}
                  {extra_where}
            )
            SELECT {select_cols}
            FROM ranked
            WHERE _rn = 1
        """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query_tile(
        self,
        ra_range: Tuple[float, float],
        dec_range: Tuple[float, float],
        band: str,
        query_buffer: float = 0.1,
    ) -> pd.DataFrame:
        """Return PS1 stacked catalog objects within a RA/Dec tile.

        Parameters
        ----------
        ra_range : (ra_min, ra_max)
            RA bounds in degrees.  Set ra_min > ra_max to indicate a
            tile that crosses RA = 0 (e.g. ``(358.0, 2.0)``).
        dec_range : (dec_min, dec_max)
            Dec bounds in degrees.
        band : str
            One of ``'g'``, ``'r'``, ``'i'``, ``'z'``, ``'y'``.
        query_buffer : float
            Extra HEALPix margin in degrees to add when pre-computing
            candidate pixels (default 0.1°).  The exact RA/Dec box filter
            is always applied after partition pruning, so this only affects
            which Parquet files are opened, not which rows are returned.

        Returns
        -------
        pd.DataFrame
            Columns: objID, raMean, decMean, qualityFlag, objInfoFlag,
            {band}KronMag, {band}KronMagErr, {band}PSFMag, {band}PSFMagErr,
            {band}psfLikelihood, {band}infoFlag, {band}infoFlag2.
            One row per unique objID (primary detection preferred).
        """
        if band not in VALID_BANDS:
            raise ValueError(f'band must be one of {sorted(VALID_BANDS)}, got {band!r}')

        ra_min, ra_max = ra_range
        dec_min, dec_max = dec_range
        wraps = (ra_min > ra_max)

        pixels = self._tile_pixels(ra_min, ra_max, dec_min, dec_max, query_buffer)
        log.debug('query_tile: %d candidate pixels', len(pixels))

        # Build RA filter that handles the wrap-around case
        if not wraps:
            ra_sql = f'raMean BETWEEN {ra_min} AND {ra_max}'
        else:
            ra_sql = f'(raMean >= {ra_min} OR raMean <= {ra_max})'
        dec_sql = f'decMean BETWEEN {dec_min} AND {dec_max}'

        sql = self._build_sql(pixels, band, ra_sql, dec_sql)
        return self._con.execute(sql).df()

    def cone_search(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
        band: str,
    ) -> pd.DataFrame:
        """Return PS1 stacked catalog objects within a cone.

        Two-phase approach
        ------------------
        1. HEALPix partition pruning: ``query_disc`` identifies candidate
           pixels; only those Parquet files are read by DuckDB.
        2. Exact separation filter: astropy ``SkyCoord.separation`` computes
           the true angular distance for each candidate row and removes objects
           outside the requested radius.

        Parameters
        ----------
        ra, dec : float
            Cone centre in degrees (ICRS).
        radius_deg : float
            Search radius in degrees.
        band : str
            One of ``'g'``, ``'r'``, ``'i'``, ``'z'``, ``'y'``.

        Returns
        -------
        pd.DataFrame
            Same columns as ``query_tile``, plus a ``sep_deg`` column
            giving the angular separation from the cone centre.
            One row per unique objID (primary detection preferred).
        """
        if band not in VALID_BANDS:
            raise ValueError(f'band must be one of {sorted(VALID_BANDS)}, got {band!r}')

        pixels = self._disc_pixels(ra, dec, radius_deg)
        log.debug('cone_search: %d candidate pixels', len(pixels))

        # Broad SQL bounding box shrinks the candidate set before the exact
        # separation computation — avoids loading the full disc's Parquet rows
        # into Python when the disc is large.
        margin  = radius_deg + 0.05
        ra_lo   = ra - margin
        ra_hi   = ra + margin
        dec_sql = f'decMean BETWEEN {max(-90, dec - margin)} AND {min(90, dec + margin)}'
        if ra_lo < 0 or ra_hi > 360:
            # Cone wraps around RA = 0; use the two-sided OR form
            ra_lo_w = (ra_lo % 360)
            ra_hi_w = (ra_hi % 360)
            ra_sql = f'(raMean >= {ra_lo_w} OR raMean <= {ra_hi_w})'
        else:
            ra_sql = f'raMean BETWEEN {ra_lo} AND {ra_hi}'

        sql = self._build_sql(pixels, band, ra_sql, dec_sql)
        df  = self._con.execute(sql).df()

        if df.empty:
            df['sep_deg'] = pd.Series(dtype=float)
            return df

        # Exact angular separation filter using astropy
        target  = SkyCoord(ra * u.deg, dec * u.deg)
        sources = SkyCoord(df['raMean'].to_numpy() * u.deg,
                           df['decMean'].to_numpy() * u.deg)
        sep_deg = target.separation(sources).deg
        mask    = sep_deg <= radius_deg

        result = df[mask].copy()
        result['sep_deg'] = sep_deg[mask]
        result = result.sort_values('sep_deg').reset_index(drop=True)
        return result


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    ps1 = PanSTARRSLocal()

    # --- query_tile ---
    print('\n=== query_tile (RA 179.9–180.1, Dec −0.1–0.1, g band) ===')
    tile = ps1.query_tile(
        ra_range=(179.9, 180.1),
        dec_range=(-0.1, 0.1),
        band='g',
    )
    print(f'{len(tile)} rows returned')
    print(tile.head())

    # --- cone_search ---
    print('\n=== cone_search (RA 180, Dec 0, r=0.2 deg, r band) ===')
    cone = ps1.cone_search(ra=180.0, dec=0.0, radius_deg=0.2, band='r')
    print(f'{len(cone)} rows returned')
    print(cone.head())

    # --- wrap-around tile ---
    print('\n=== query_tile wrap-around (RA 359–1, Dec −1–1, i band) ===')
    wrap = ps1.query_tile(
        ra_range=(359.0, 1.0),
        dec_range=(-1.0, 1.0),
        band='i',
    )
    print(f'{len(wrap)} rows returned')
    print(wrap.head())
