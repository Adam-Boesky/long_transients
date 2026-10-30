"""Download the eROSITA-DE DR2 main catalogue and rebuild the slim crossmatch parquet.

One 1.4 GB FITS file (eRASS3_Main_v1.3.fits.gz, 1.98M sources) from MPE,
anonymous. Written to a temporary name and renamed when complete, so an
interrupted download is retried rather than half-read; an existing file is
kept unless --force is given.

Usage
-----
    python scripts/build_erosita_catalog.py                # download + rebuild
    python scripts/build_erosita_catalog.py --rebuild-only  # parquet from the FITS on disk
    python scripts/build_erosita_catalog.py --force         # re-download

Then append the columns to existing enriched tables with

    python Source_Analysis/enrich_combined_tabs.py <table> --in-place --catalogs erosita
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Source_Analysis.catalogs import erosita  # noqa: E402

TIMEOUT_SECONDS = (30, 300)


def download(dest: Path = erosita.CATALOG_FITS, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f'{dest.name} already on disk ({dest.stat().st_size / 1e9:.2f} GB); skipping download')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix('.incoming')
    with requests.get(erosita.CATALOG_URL, stream=True, timeout=TIMEOUT_SECONDS) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        done = 0
        with open(staging, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB', end='', flush=True)
    print()
    if total and staging.stat().st_size != total:
        staging.unlink(missing_ok=True)
        sys.exit(f'download truncated: got {done} of {total} bytes')
    staging.replace(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rebuild-only', action='store_true', help='skip the download')
    ap.add_argument('--force', action='store_true', help='re-download even if present')
    args = ap.parse_args()

    if not args.rebuild_only:
        download(force=args.force)
    print(f'Building {erosita.CATALOG_PARQUET.name}...')
    path = erosita.build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
    print(f'  {len(erosita.load_catalog()):,} sources')


if __name__ == '__main__':
    main()
