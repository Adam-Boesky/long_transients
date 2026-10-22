"""Build the catalog tier of the source database.

Reads the three `*_enriched.ecsv` tables produced by `enrich_combined_tabs.py`,
assigns a stable `source_uuid` to every source, and writes the artifacts. No
network access -- the crossmatch results are already in those tables.

Usage
-----
    python -m source_db.build_catalog_tier --dry-run   # row counts, writes nothing
    python -m source_db.build_catalog_tier             # parquet + identity + release
    python -m source_db.build_catalog_tier --sqlite    # also an indexed SQLite

Writes into Data/db/:

    sources.parquet    one row per source, keys + ~300 science columns
    identity.parquet   source_uuid <-> position, carried across rebuilds
    release.json       release id, build time, git sha, inputs, counts

Rebuilding
----------
Re-running against a *different* `filter_results_*` directory is the normal
case. UUIDs are carried forward by positional crossmatch, so a source whose
centroid moved slightly between pipeline versions keeps its identity. Sources
absent from the new build are retired rather than deleted.

`identity.parquet` is the only artifact that cannot be regenerated. Until the
publish path exists it lives under the gitignored `Data/`, so do not start
annotating against a build whose identity file is not backed up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source_db import catalog_tier  # noqa: E402
from source_db.paths import DEFAULT_COMBINED, DEFAULT_OUT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--combined-dir', type=Path, default=DEFAULT_COMBINED,
                    help=f'directory of enriched tables (default {DEFAULT_COMBINED})')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help=f'where to write the artifacts (default {DEFAULT_OUT_DIR})')
    ap.add_argument('--sqlite', action='store_true',
                    help='also write an indexed sources.sqlite (~290 MB)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen without writing anything')
    args = ap.parse_args()

    if not args.combined_dir.is_dir():
        sys.exit(f'No such directory: {args.combined_dir}')

    summary = catalog_tier.build(
        args.combined_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        write_sqlite=args.sqlite,
    )

    print('\n' + '-' * 60)
    for key in ('release_id', 'n_sources', 'n_columns', 'n_new_uuids',
                'n_carried_forward', 'n_retired', 'n_identity'):
        v = summary[key]
        print(f'  {key:20} {v:,}' if isinstance(v, int) else f'  {key:20} {v}')


if __name__ == '__main__':
    main()
