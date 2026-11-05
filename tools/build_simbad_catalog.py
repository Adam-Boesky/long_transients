"""Download the SIMBAD `basic` table and rebuild the slim crossmatch parquet.

SIMBAD is queried through the CDS TAP service. A single response is capped at
2M rows, so the table is pulled in `oid` pages (see `simbad.OID_STRIDE`) --
roughly 21 pages of ~1.05M rows for the current 22.2M-object catalog, about
half an hour end to end.

Pages are written one CSV each under Data/catalogs/raw/simbad/ and skipped if
already present, so an interrupted download resumes instead of restarting. Pass
--force to re-fetch pages that are already on disk.

Usage
-----
    python scripts/build_simbad_catalog.py                # download + rebuild
    python scripts/build_simbad_catalog.py --rebuild-only  # parquet from pages on disk
    python scripts/build_simbad_catalog.py --force         # re-fetch every page

Notes
-----
No credentials are needed; CDS serves this anonymously. It does throttle
clients issuing many queries per second, which this never approaches -- 21
queries spread over half an hour, with a short pause between pages.

Like TNS, SIMBAD grows continuously, so re-running occasionally is worthwhile.
Unlike TNS there is no archiving here: the parquet is a pure function of the
pages on disk, and the pages are re-downloadable.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Source_Analysis.catalogs import simbad  # noqa: E402

PAUSE_SECONDS = 2.0
TIMEOUT_SECONDS = 1200


def _tap_csv(adql: str, dest: Path | None = None, maxrec: int | None = None) -> str | None:
    """Run one ADQL query, returning CSV text or streaming it to `dest`.

    TAP reports query errors as a 200 response holding a VOTable error document
    rather than an error status, so the payload has to be inspected: a good CSV
    response opens with the header line, never with XML.
    """
    data = {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': adql}
    if maxrec is not None:
        data['MAXREC'] = str(maxrec)

    resp = requests.post(simbad.TAP_SYNC_URL, data=data,
                         stream=dest is not None, timeout=TIMEOUT_SECONDS)
    if resp.status_code != 200:
        sys.exit(f'TAP returned HTTP {resp.status_code}\n{resp.text[:2000]}')

    if dest is None:
        if resp.text.lstrip().startswith('<'):
            sys.exit(f'TAP returned an error document:\n{resp.text[:2000]}')
        return resp.text

    # Written to a temp name so an interrupted transfer cannot leave a truncated
    # page that the next run would happily skip.
    staging = dest.with_suffix('.csv.incoming')
    first = True
    with open(staging, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if first and chunk.lstrip()[:1] == b'<':
                staging.unlink(missing_ok=True)
                sys.exit(f'TAP returned an error document:\n{chunk[:2000].decode(errors="replace")}')
            first = False
            fh.write(chunk)
    staging.replace(dest)
    return None


def download(raw_dir: Path, stride: int, force: bool) -> None:
    """Fetch `otypedef` and every `oid` page of `basic` into `raw_dir`."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    otypedef = raw_dir / 'otypedef.csv'
    if force or not otypedef.exists():
        print('Downloading otypedef (object-type hierarchy)')
        _tap_csv(simbad.OTYPEDEF_ADQL, dest=otypedef)
        print(f'  wrote {otypedef.name}')

    total = int(pd.read_csv(io.StringIO(_tap_csv(simbad.COUNT_ADQL)))['n'].iloc[0])
    span = pd.read_csv(io.StringIO(_tap_csv(simbad.OID_RANGE_ADQL)))
    oid_lo, oid_hi = int(span['lo'].iloc[0]), int(span['hi'].iloc[0])
    n_pages = (oid_hi - oid_lo) // stride + 1
    print(f'SIMBAD basic: {total:,} objects, oid {oid_lo:,}-{oid_hi:,} '
          f'-> {n_pages} pages of {stride:,} ids')

    for page, lo in enumerate(range(oid_lo, oid_hi + 1, stride)):
        hi = min(lo + stride - 1, oid_hi)
        dest = raw_dir / f'basic_{page:03d}.csv'
        if dest.exists() and not force:
            print(f'  [{page + 1}/{n_pages}] {dest.name} present, skipping')
            continue
        t0 = time.time()
        _tap_csv(simbad.basic_adql(lo, hi), dest=dest, maxrec=simbad.TAP_MAXREC)
        rows = sum(1 for _ in open(dest, encoding='utf-8', errors='replace')) - 1
        print(f'  [{page + 1}/{n_pages}] {dest.name} oid {lo:,}-{hi:,}: '
              f'{rows:,} rows, {dest.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s')
        if rows >= simbad.TAP_MAXREC:
            sys.exit(f'{dest.name} hit the {simbad.TAP_MAXREC:,}-row cap and is '
                     f'therefore truncated -- re-run with a smaller --stride')
        time.sleep(PAUSE_SECONDS)

    on_disk = sum(
        sum(1 for _ in open(p, encoding='utf-8', errors='replace')) - 1
        for p in sorted(raw_dir.glob('basic_*.csv')))
    print(f'\n{on_disk:,} rows across {len(list(raw_dir.glob("basic_*.csv")))} pages '
          f'(TAP reported {total:,})')
    if on_disk != total:
        print(f'  NOTE: differs from the reported count by {on_disk - total:,}; '
              f'SIMBAD updates continuously, so a small drift mid-download is normal')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rebuild-only', action='store_true',
                    help='skip the download and rebuild from the pages on disk')
    ap.add_argument('--force', action='store_true',
                    help='re-fetch pages that are already present')
    ap.add_argument('--stride', type=int, default=simbad.OID_STRIDE,
                    help=f'oid ids per page (default {simbad.OID_STRIDE:,})')
    ap.add_argument('--raw-dir', type=Path, default=simbad.CATALOG_RAW_DIR,
                    help='where the downloaded pages live')
    args = ap.parse_args()

    if not args.rebuild_only:
        download(args.raw_dir, args.stride, args.force)

    print(f'\nBuilding {simbad.CATALOG_PARQUET.name}...')
    path = simbad.build_parquet(args.raw_dir)
    frame = pd.read_parquet(path, columns=['ra', 'simbad_otype'])
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
    print(f'  {len(frame):,} objects, {frame["simbad_otype"].nunique()} distinct object types')


if __name__ == '__main__':
    main()
