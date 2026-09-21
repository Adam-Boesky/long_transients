"""Refresh the local TNS object catalog from the daily staged dump.

TNS is the one local catalog that goes stale -- SDSS DR17 and DESI DR1 are
frozen releases, but TNS grows by ~25k objects a year. This downloads the
staged full-catalog CSV and rebuilds the slim parquet that
`Source_Analysis.catalogs.tns` crossmatches against.

One-time setup
--------------
The download requires a TNS `tns_marker` user-agent identifying you. A *user*
marker needs no API key (unlike a bot marker), so all that is needed is your
TNS user id and name, in ~/vault/tns_marker.txt as two lines:

    4306
    Adam Pearce Boesky

Or set TNS_USER_ID and TNS_USER_NAME in the environment, which take precedence.

Usage
-----
    python scripts/refresh_tns_catalog.py           # download + rebuild
    python scripts/refresh_tns_catalog.py --rebuild-only
    python scripts/refresh_tns_catalog.py --no-archive

The previous CSV and parquet are moved into Data/catalogs/raw/archive/, tagged
with their data cutoff date, so a refresh can be backed out and so the
snapshot a given analysis ran against stays recoverable.

Notes
-----
TNS asks that large data sets be pulled from these staged files rather than by
paging the search page or issuing bulk cone searches, so this is the sanctioned
path. Re-running monthly is plenty; the dump is regenerated daily after UT
midnight.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Source_Analysis.catalogs import tns  # noqa: E402

STAGED_URL = ('https://www.wis-tns.org/system/files/tns_public_objects/'
              'tns_public_objects.csv.zip')
MARKER_FILE = Path.home() / 'vault' / 'tns_marker.txt'
ARCHIVE_DIR = tns.CATALOG_CSV.parent / 'archive'


def _marker() -> str:
    """The tns_marker user-agent string identifying this download to TNS."""
    user_id = os.environ.get('TNS_USER_ID')
    user_name = os.environ.get('TNS_USER_NAME')

    if not (user_id and user_name):
        if not MARKER_FILE.exists():
            sys.exit(f'No TNS credentials: set TNS_USER_ID and TNS_USER_NAME, '
                     f'or write your id and name as two lines in {MARKER_FILE}')
        lines = [ln.strip() for ln in MARKER_FILE.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            sys.exit(f'{MARKER_FILE} should hold your TNS user id on line 1 '
                     f'and your name on line 2')
        user_id, user_name = lines[0], lines[1]

    # json.dumps so a name containing quotes cannot break the header.
    return 'tns_marker' + json.dumps(
        {'tns_id': int(user_id), 'type': 'user', 'name': user_name})


def download(dest_zip: Path) -> Path:
    """Fetch the staged full-catalog zip. The endpoint is POST-only."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {STAGED_URL}')

    resp = requests.post(STAGED_URL, headers={'user-agent': _marker()},
                         stream=True, timeout=600)
    # A bad marker returns a 403 HTML error page, not an error status we can
    # rely on alone, so check the content type too.
    if resp.status_code != 200:
        sys.exit(f'TNS returned HTTP {resp.status_code} -- check your marker '
                 f'credentials (or the rate limit, which is per 60s window)')
    if 'zip' not in resp.headers.get('content-type', ''):
        sys.exit(f'Expected a zip, got {resp.headers.get("content-type")!r} -- '
                 f'this is usually an auth failure page')

    with open(dest_zip, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    print(f'  wrote {dest_zip.name} ({dest_zip.stat().st_size / 1e6:.1f} MB)')
    return dest_zip


def extract(src_zip: Path, dest_csv: Path) -> Path:
    """Unpack the single CSV out of the staged zip."""
    with zipfile.ZipFile(src_zip) as zf:
        names = [n for n in zf.namelist() if n.endswith('.csv')]
        if len(names) != 1:
            sys.exit(f'Expected one CSV in {src_zip.name}, found {names}')
        with zf.open(names[0]) as fsrc, open(dest_csv, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
    print(f'  extracted {dest_csv.name} ({dest_csv.stat().st_size / 1e6:.1f} MB)')
    return dest_csv


def _cutoff_tag(parquet: Path) -> str:
    """Data cutoff of an existing parquet, as YYYYMMDD, for archive naming."""
    try:
        stamp = pd.read_parquet(parquet, columns=['tns_lastmodified'])
        return str(pd.to_datetime(stamp['tns_lastmodified'].max()).date()).replace('-', '')
    except Exception:
        return 'unknown'


def archive_existing() -> None:
    """Move the current CSV and parquet aside, tagged with their data cutoff."""
    if not tns.CATALOG_PARQUET.exists():
        return
    tag = _cutoff_tag(tns.CATALOG_PARQUET)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for src, name in ((tns.CATALOG_CSV, f'tns_public_objects_cutoff{tag}.csv'),
                      (tns.CATALOG_PARQUET, f'tns_objects_slim_cutoff{tag}.parquet')):
        if src.exists():
            shutil.move(str(src), ARCHIVE_DIR / name)
            print(f'  archived {name}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rebuild-only', action='store_true',
                    help='skip the download and rebuild from the CSV on disk')
    ap.add_argument('--no-archive', action='store_true',
                    help='overwrite the current CSV/parquet instead of archiving them')
    args = ap.parse_args()

    if args.rebuild_only:
        if not tns.CATALOG_CSV.exists():
            sys.exit(f'No CSV at {tns.CATALOG_CSV} to rebuild from')
    else:
        # Download to a temp name first so a failure partway cannot leave a
        # truncated CSV in place of a good one.
        staging = tns.CATALOG_CSV.with_suffix('.csv.incoming')
        staging_zip = tns.CATALOG_CSV.with_suffix('.csv.zip.incoming')
        download(staging_zip)
        extract(staging_zip, staging)

        if not args.no_archive:
            archive_existing()
        staging.replace(tns.CATALOG_CSV)
        staging_zip.unlink(missing_ok=True)

    print(f'Building {tns.CATALOG_PARQUET.name}...')
    path = tns.build_parquet()
    frame = pd.read_parquet(path)
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)')
    print(f'  {len(frame):,} objects, newest entry {frame["tns_lastmodified"].max()}')


if __name__ == '__main__':
    main()
