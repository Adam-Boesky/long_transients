import os
import re
import linecache
import numpy as np
import pandas as pd
import requests

from typing import List

from Extracting.utils import get_data_path
from ztf_fp_query.constants import HTTP_AUTH, DOWNLOAD_BASE_URL
from ztf_fp_query.submit import query_recent_jobs

_MAP_FNAME = 'source_map.csv'
_LC_PATTERN = re.compile(r'.+/(.+\.txt)$')
_RA_PATTERN = re.compile(r'Requested input R\.A\. = ([\d.]+) degrees')
_DEC_PATTERN = re.compile(r'Requested input Dec\. = (-?[\d.]+) degrees')


def get_data_dir() -> str:
    return os.path.join(get_data_path(), 'followup', 'ztf_forced_photometry')


def get_map_path() -> str:
    return os.path.join(get_data_dir(), _MAP_FNAME)


def load_map() -> pd.DataFrame:
    """Load the local (ra, dec) -> filename index, creating it if absent."""
    path = get_map_path()
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({'ra': [], 'dec': [], 'fname': []})


def _save_map(map_df: pd.DataFrame) -> None:
    map_df.to_csv(get_map_path(), index=False)


def _parse_ra_dec(fpath: str) -> tuple[float, float]:
    """Read ra and dec from the header of a ZFPS ASCII lightcurve file."""
    ra_line = linecache.getline(fpath, 4)
    dec_line = linecache.getline(fpath, 5)
    ra_match = _RA_PATTERN.search(ra_line)
    dec_match = _DEC_PATTERN.search(dec_line)
    ra = float(ra_match.group(1)) if ra_match else None
    dec = float(dec_match.group(1)) if dec_match else None
    return ra, dec


def rebuild_map() -> pd.DataFrame:
    """Rebuild the index from all .txt files currently in the data directory."""
    data_dir = get_data_dir()
    rows = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            ra, dec = _parse_ra_dec(os.path.join(data_dir, fname))
            rows.append({'ra': ra, 'dec': dec, 'fname': fname})
    map_df = pd.DataFrame(rows)
    _save_map(map_df)
    return map_df


def sync(outdir: str = None) -> List[str]:
    """Download all completed jobs from the ZTF service and update the local index.

    Returns the list of file paths that were downloaded (skips already-present files).
    """
    if outdir is None:
        outdir = get_data_dir()
    os.makedirs(outdir, exist_ok=True)

    # Get completed jobs from the service
    recent_df = query_recent_jobs()
    lc_paths = recent_df['lightcurve'].dropna().tolist()

    map_df = load_map()
    downloaded = []

    for lc_path in lc_paths:
        match = _LC_PATTERN.match(lc_path)
        if not match:
            print(f'Could not parse filename from: {lc_path}')
            continue
        fname = match.group(1)
        fpath = os.path.join(outdir, fname)

        if os.path.exists(fpath):
            continue

        url = DOWNLOAD_BASE_URL + lc_path
        response = requests.get(url, auth=HTTP_AUTH)
        if response.status_code == 200:
            with open(fpath, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded: {fpath}')
            downloaded.append(fpath)

            # Add to map immediately after download
            ra, dec = _parse_ra_dec(fpath)
            new_row = pd.DataFrame([{'ra': ra, 'dec': dec, 'fname': fname}])
            map_df = pd.concat([map_df.astype(new_row.dtypes), new_row], ignore_index=True)
        else:
            print(f'Failed to download {url}: {response.status_code} {response.reason}')

    _save_map(map_df)
    return downloaded
