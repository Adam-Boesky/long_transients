"""Build the manifest that drives the classification app.

One row per analysis-page PDF, pre-joined to the enriched filter table so the
app can filter on crossmatch properties without opening a 40 MB ECSV or
building a KD-tree at request time.

    python -m scripts.build_classify_manifest \
        --pages-dir Data/kde_analysis_pages_9_9_2026_gaia_propagation \
        --combined-dir Data/filter_results_7_30_2026_gaia_propagation/combined \
        --workspace Data/classify_workspace

Writes into <workspace>/:
    manifest.parquet   one row per PDF (the app's only data input)
    config.json        where the PDFs live, so the app can find them
    buckets.csv        seeded empty -- buckets are added from inside the app
    members.csv        seeded with one member, edit to add collaborators
    labels.csv         seeded empty, append-only

Re-running is safe and idempotent: manifest.parquet and config.json are
rewritten, while buckets/members/labels are left alone once they exist.

Joining PDFs to table rows
--------------------------
Filenames encode position via Source_Analysis.coord_utils at 4 decimal places.
The join tries an exact coord-stem string match first, then falls back to a
1.5" sky match (= MATCH_RADIUS_ARCSEC in enrich_combined_tabs, itself
Source.max_arcsec). Both counts are reported, along with any PDF that matched
nothing -- a stale page directory shows up here rather than silently producing
a manifest full of nulls.

Derived columns
---------------
`spec_z` takes the first available redshift in the order desi -> sdss -> tns ->
agn_db, and only accepts desi/sdss values whose zwarn/zwarning flag is 0.
`spec_z_source` records which catalog it came from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Extracting.utils import load_ecsv  # noqa: E402
from Source_Analysis.coord_utils import coord_stem, parse_coord_stem  # noqa: E402

# Same radius the enrichment step used, so a page can never be joined to a row
# the crossmatch itself would have considered a different object.
MATCH_RADIUS_ARCSEC = 1.5

# Catalog subdirectory -> the combined table index it was generated from.
# 0 = in both surveys, 1 = ZTF only, 2 = PanSTARRS only (see combine_filtered_tabs).
CATALOG_TABLE_INDEX = {'in_both': 0, 'in_ztf': 1, 'in_pstarr': 2}

CROSSMATCH_COLUMNS: dict[str, list[str]] = {
    'sdss': ['sdss_matched', 'sdss_sep_arcsec', 'sdss_class', 'sdss_subclass',
             'sdss_z', 'sdss_zwarning'],
    'desi': ['desi_matched', 'desi_sep_arcsec', 'desi_spectype', 'desi_subtype',
             'desi_z', 'desi_zwarn'],
    'tns': ['tns_matched', 'tns_sep_arcsec', 'tns_designation', 'tns_type',
            'tns_redshift', 'tns_discoverydate'],
    'wise': ['wise_matched', 'wise_sep_arcsec', 'wise_w1mpro', 'wise_w2mpro',
             'wise_w3mpro', 'wise_var_flg', 'wise_ph_qual', 'wise_cc_flags'],
    'agn_db': ['agn_db_matched', 'agn_db_sep_arcsec', 'agn_db_best_class_all',
               'agn_db_best_class_sub_all', 'agn_db_best_z_merged', 'agn_db_star_flag'],
}

PHOTOMETRY_COLUMNS = [
    'ZTF_gPSFMag', 'ZTF_rPSFMag', 'ZTF_iPSFMag',
    'PSTARR_gPSFMag', 'PSTARR_rPSFMag', 'PSTARR_iPSFMag',
]

DMAG_COLUMNS = ['rank', 'g_dmag', 'r_dmag', 'i_dmag', 'mean_abs_dmag']

MEMBERS_HEADER = ['member_id', 'display_name', 'active']
BUCKETS_HEADER = ['bucket_id', 'display_name', 'shortcut', 'sort_order',
                  'created_by', 'created_at', 'active', 'description']
LABELS_HEADER = ['timestamp', 'member', 'page_id', 'catalog', 'filename',
                 'ra', 'dec', 'bucket', 'comment', 'app_version']


def log(msg: str) -> None:
    print(msg, flush=True)


# ── astropy -> pandas ─────────────────────────────────────────────────────────

# `to_pandas_array` moved to catalogs/_base.py, next to `to_astropy_table` whose
# inverse it is, so this script and `source_db.catalog_tier` share one copy.
# Re-exported here because the rest of this module refers to it unqualified.
from Source_Analysis.catalogs._base import to_pandas_array  # noqa: E402,F401


def column_or_null(table: Table, name: str, n: int) -> pd.Series:
    """Column `name` as a pandas array, or an all-null float column if absent.

    Keeps the manifest schema stable even when a catalog block is missing from
    one of the combined tables.
    """
    if name not in table.colnames:
        return pd.Series(np.full(n, np.nan))
    return to_pandas_array(table[name])


# ── inputs ────────────────────────────────────────────────────────────────────

def discover_pages(pages_dir: Path) -> pd.DataFrame:
    """Every PDF under `pages_dir`, with its catalog subdirectory and coordinates."""
    rows = []
    for pdf in sorted(pages_dir.rglob('*.pdf')):
        parts = pdf.relative_to(pages_dir).parts
        catalog = parts[0] if len(parts) > 1 else '(root)'
        coord = parse_coord_stem(pdf.stem)
        rows.append({
            'page_id': f'{catalog}/{pdf.name}',
            'catalog': catalog,
            'filename': pdf.name,
            'stem': pdf.stem,
            'ra': coord[0] if coord else np.nan,
            'dec': coord[1] if coord else np.nan,
        })
    return pd.DataFrame(rows)


def load_dmag_sidecars(pages_dir: Path, catalog: str) -> pd.DataFrame:
    """Merge every `dmag_order*.csv` in a catalog subdirectory into one frame.

    Filenames are not unique in the sidecars. `coord_stem` rounds to 4 decimal
    places, i.e. a grid of roughly 0.36", and sources closer together than that
    collide onto one name -- which also means one source's PDF has already
    overwritten the other's on disk. Only one row can survive here, so the
    collisions are reported rather than dropped quietly: the value kept may
    describe the source whose page was overwritten.
    """
    catalog_dir = pages_dir if catalog == '(root)' else pages_dir / catalog
    frames = []
    for csv_path in sorted(catalog_dir.glob('dmag_order*.csv')):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(columns=['filename'] + DMAG_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    keep = ['filename'] + [c for c in DMAG_COLUMNS if c in merged.columns]
    merged = merged[keep]

    clashing = merged['filename'].duplicated(keep=False)
    if clashing.any():
        names = merged.loc[clashing, 'filename']
        log(f'  WARNING: {names.nunique()} filename(s) cover '
            f'{int(clashing.sum())} sidecar rows -- coordinates within ~0.36" '
            f'share a filename, so their pages overwrote each other')
        if 'mean_abs_dmag' in merged.columns:
            spread = (merged.loc[clashing]
                      .groupby('filename')['mean_abs_dmag']
                      .agg(lambda s: s.max() - s.min()))
            log(f'           their mean_abs_dmag differs by up to '
                f'{spread.max():.3f} (median {spread.median():.3f})')

    return merged.drop_duplicates(subset='filename', keep='last')


# ── the join ──────────────────────────────────────────────────────────────────

def join_to_table(pages: pd.DataFrame, table: Table) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach enriched-table columns to each page.

    Exact coord-stem match first, then a sky match within MATCH_RADIUS_ARCSEC.
    Returns the selected columns indexed like `pages`, plus a tally of how each
    page was matched.
    """
    n_rows = len(table)
    table_ra = np.asarray(table['ra'], dtype=float)
    table_dec = np.asarray(table['dec'], dtype=float)

    # Exact stem match. Duplicate stems (two rows rounding to the same 4 d.p.
    # position) would be ambiguous, so they are dropped from the exact index and
    # left to the sky match.
    stems = pd.Series([coord_stem(r, d) for r, d in zip(table_ra, table_dec)])
    unique_stems = stems[~stems.duplicated(keep=False)]
    stem_to_row = {s: i for i, s in unique_stems.items()}

    row_for_page = np.full(len(pages), -1, dtype=int)
    how = np.array(['none'] * len(pages), dtype=object)

    for i, stem in enumerate(pages['stem']):
        # `wide_` and other prefixes are stripped by re-encoding the parsed
        # coordinates rather than by string surgery.
        coord = parse_coord_stem(stem)
        if coord is None:
            continue
        canonical = coord_stem(*coord)
        row = stem_to_row.get(canonical)
        if row is not None:
            row_for_page[i] = row
            how[i] = 'exact'

    # Sky match for whatever is left.
    todo = np.flatnonzero((row_for_page < 0) & np.isfinite(pages['ra'].to_numpy(dtype=float)))
    if len(todo) and n_rows:
        page_sky = SkyCoord(ra=pages['ra'].to_numpy(dtype=float)[todo] * u.deg,
                            dec=pages['dec'].to_numpy(dtype=float)[todo] * u.deg)
        table_sky = SkyCoord(ra=table_ra * u.deg, dec=table_dec * u.deg)
        idx, sep, _ = page_sky.match_to_catalog_sky(table_sky)
        ok = sep.arcsec <= MATCH_RADIUS_ARCSEC
        row_for_page[todo[ok]] = idx[ok]
        how[todo[ok]] = 'sky'

    # Gather the columns we care about, NA where a page matched nothing.
    wanted = (
        [c for block in CROSSMATCH_COLUMNS.values() for c in block]
        + PHOTOMETRY_COLUMNS
    )
    matched_rows = row_for_page.copy()
    unmatched = matched_rows < 0
    matched_rows[unmatched] = 0  # placeholder; blanked out below

    out = {}
    for name in wanted:
        full = column_or_null(table, name, n_rows)
        if len(full) == 0:
            out[name] = pd.Series(np.full(len(pages), np.nan), index=pages.index)
            continue
        taken = full.iloc[matched_rows].reset_index(drop=True)
        if unmatched.any():
            taken[unmatched] = None
        taken.index = pages.index
        out[name] = taken

    tally = {
        'exact': int((how == 'exact').sum()),
        'sky': int((how == 'sky').sum()),
        'none': int((how == 'none').sum()),
    }
    return pd.DataFrame(out, index=pages.index), tally


# ── derived columns ───────────────────────────────────────────────────────────

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """WISE colors, a single best spectroscopic redshift, and match counts."""
    df = df.copy()

    df['wise_w1_w2'] = df['wise_w1mpro'] - df['wise_w2mpro']
    df['wise_w2_w3'] = df['wise_w2mpro'] - df['wise_w3mpro']

    # Only trust desi/sdss redshifts with a clean warning flag; tns and agn_db
    # carry no equivalent, so they are taken as published.
    candidates = [
        ('desi', df['desi_z'].where(df['desi_zwarn'].fillna(-1) == 0)),
        ('sdss', df['sdss_z'].where(df['sdss_zwarning'].fillna(-1) == 0)),
        ('tns', df['tns_redshift']),
        ('agn_db', df['agn_db_best_z_merged']),
    ]
    spec_z = pd.Series(np.nan, index=df.index, dtype='float64')
    spec_z_source = pd.Series(pd.array([None] * len(df), dtype='string'), index=df.index)
    for source, values in candidates:
        fill = spec_z.isna() & values.notna()
        spec_z[fill] = values[fill].astype('float64')
        spec_z_source[fill] = source
    df['spec_z'] = spec_z
    df['spec_z_source'] = spec_z_source

    matched_cols = [f'{p}_matched' for p in CROSSMATCH_COLUMNS]
    df['n_catalog_matches'] = (
        df[matched_cols].fillna(False).astype(bool).sum(axis=1).astype('Int64')
    )
    return df


# ── workspace seeding ─────────────────────────────────────────────────────────

def seed_csv(path: Path, header: list[str], rows: list[list] | None = None) -> None:
    """Create a CSV with `header` if it does not exist. Never overwrites."""
    if path.exists():
        log(f'  {path.name} already exists, leaving it alone')
        return
    pd.DataFrame(rows or [], columns=header).to_csv(path, index=False)
    log(f'  seeded {path.name}')


# ── main ──────────────────────────────────────────────────────────────────────

def build(pages_dir: Path, combined_dir: Path, workspace: Path, member: str) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)

    pages = discover_pages(pages_dir)
    if pages.empty:
        raise SystemExit(f'No PDFs found under {pages_dir}')
    log(f'{len(pages):,} PDFs under {pages_dir.name}')
    for catalog, count in pages['catalog'].value_counts().items():
        log(f'  {catalog}: {count:,}')

    unparseable = pages['ra'].isna().sum()
    if unparseable:
        log(f'  WARNING: {unparseable} filename(s) have no parseable coordinates')

    per_catalog = []
    for catalog, group in pages.groupby('catalog', sort=True):
        log(f'\n{catalog}:')

        dmag = load_dmag_sidecars(pages_dir, catalog)
        merged = group.merge(dmag, on='filename', how='left')
        merged.index = group.index
        missing_dmag = merged['mean_abs_dmag'].isna().sum() if 'mean_abs_dmag' in merged else len(merged)
        log(f'  dmag sidecar: {len(merged) - missing_dmag}/{len(merged)} pages have an entry')

        index = CATALOG_TABLE_INDEX.get(catalog)
        table_path = combined_dir / f'{index}_enriched.ecsv' if index is not None else None
        if table_path is None or not table_path.exists():
            log(f'  WARNING: no enriched table for this catalog '
                f'({table_path.name if table_path else "unmapped"}) -- '
                f'crossmatch columns will be null')
            table = Table()
            tally = {'exact': 0, 'sky': 0, 'none': len(merged)}
            joined = pd.DataFrame(index=merged.index)
            for name in ([c for b in CROSSMATCH_COLUMNS.values() for c in b] + PHOTOMETRY_COLUMNS):
                joined[name] = pd.array([pd.NA] * len(merged), dtype='float64')
        else:
            table = load_ecsv(str(table_path))
            log(f'  {table_path.name}: {len(table):,} rows')
            joined, tally = join_to_table(merged, table)

        log(f'  join: {tally["exact"]:,} exact, {tally["sky"]:,} sky-matched, '
            f'{tally["none"]:,} unmatched')
        if tally['none']:
            orphans = merged.loc[joined.index[np.asarray(
                joined[[f'{p}_matched' for p in CROSSMATCH_COLUMNS]].isna().all(axis=1)
            )], 'filename'].tolist()
            log(f'  WARNING: {tally["none"]} page(s) matched no table row, e.g. '
                f'{", ".join(orphans[:5])}')

        per_catalog.append(pd.concat([merged, joined], axis=1))

    manifest = pd.concat(per_catalog, ignore_index=True)
    manifest = add_derived(manifest)
    manifest = manifest.drop(columns=['stem'])

    # Triage order: biggest dmag first, pages with no dmag entry last.
    manifest = manifest.sort_values(
        ['catalog', 'mean_abs_dmag'], ascending=[True, False], na_position='last'
    ).reset_index(drop=True)

    out_path = workspace / 'manifest.parquet'
    manifest.to_parquet(out_path, index=False)
    log(f'\nwrote {out_path} ({len(manifest):,} rows x {len(manifest.columns)} cols, '
        f'{out_path.stat().st_size / 1e3:.0f} KB)')

    config = {
        'pages_root': str(pages_dir.resolve()),
        'pages_dirname': pages_dir.name,
        'combined_dir': str(combined_dir.resolve()),
        'backend': 'local',
    }
    (workspace / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    log(f'wrote {workspace / "config.json"}')

    log('\nseeding stores:')
    seed_csv(workspace / 'buckets.csv', BUCKETS_HEADER)
    seed_csv(workspace / 'members.csv', MEMBERS_HEADER, [[member, member, True]])
    seed_csv(workspace / 'labels.csv', LABELS_HEADER)

    summary(manifest)
    return out_path


def summary(manifest: pd.DataFrame) -> None:
    """Print what the app will be able to filter on, so gaps are obvious."""
    log('\nfilterable coverage:')
    for prefix in CROSSMATCH_COLUMNS:
        col = f'{prefix}_matched'
        n = int(manifest[col].fillna(False).astype(bool).sum())
        log(f'  {prefix:7s} {n:5,} / {len(manifest):,} matched')
    n_z = int(manifest['spec_z'].notna().sum())
    log(f'  {"spec_z":7s} {n_z:5,} / {len(manifest):,} have a redshift')
    if n_z:
        by_source = manifest['spec_z_source'].value_counts()
        log('           ' + ', '.join(f'{k}: {v}' for k, v in by_source.items()))
    n_color = int(manifest['wise_w1_w2'].notna().sum())
    log(f'  {"W1-W2":7s} {n_color:5,} / {len(manifest):,} have a color')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pages-dir', required=True,
                        help='analysis-page directory holding the catalog subdirs of PDFs')
    parser.add_argument('--combined-dir', required=True,
                        help='filter_results .../combined directory holding *_enriched.ecsv')
    parser.add_argument('--workspace', default='Data/classify_workspace',
                        help='output directory for the manifest and stores')
    parser.add_argument('--member', default=os.environ.get('USER', 'me'),
                        help='member_id to seed members.csv with')
    args = parser.parse_args()

    build(
        Path(args.pages_dir).expanduser().resolve(),
        Path(args.combined_dir).expanduser().resolve(),
        Path(args.workspace).expanduser().resolve(),
        args.member,
    )


if __name__ == '__main__':
    main()
