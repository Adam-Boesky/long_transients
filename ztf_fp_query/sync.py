import os
import re
import linecache
import tempfile
import pandas as pd
import requests

from typing import List

from Source_Analysis import followup
from ztf_fp_query.constants import HTTP_AUTH, DOWNLOAD_BASE_URL
from ztf_fp_query.submit import query_recent_jobs

_LC_PATTERN = re.compile(r'.+/(.+\.txt)$')
_RA_PATTERN = re.compile(r'Requested input R\.A\. = ([\d.]+) degrees')
_DEC_PATTERN = re.compile(r'Requested input Dec\. = (-?[\d.]+) degrees')


def _parse_ra_dec(fpath: str) -> tuple[float, float]:
    """Read ra and dec from the header of a ZFPS ASCII lightcurve file."""
    linecache.checkcache(fpath)
    ra_match = _RA_PATTERN.search(linecache.getline(fpath, 4))
    dec_match = _DEC_PATTERN.search(linecache.getline(fpath, 5))
    ra = float(ra_match.group(1)) if ra_match else None
    dec = float(dec_match.group(1)) if dec_match else None
    return ra, dec


def load_map() -> pd.DataFrame:
    """Every ZFPS file in the target directories: requested ra, dec, fname, path, slug."""
    rows = []
    for t in followup.iter_targets():
        for p in t.zfps_files:
            ra, dec = _parse_ra_dec(str(p))
            rows.append({'ra': ra, 'dec': dec, 'fname': p.name, 'path': str(p), 'slug': t.slug})
    return pd.DataFrame(rows, columns=['ra', 'dec', 'fname', 'path', 'slug'])


def sync() -> List[str]:
    """Download completed jobs not already on disk into their follow-up target's zfps/.

    Only jobs requested at an existing target's position are kept: make the target
    (`followup.create_target`) first. Returns the paths of the files downloaded.
    """
    recent_df = query_recent_jobs().dropna(subset=['lightcurve'])
    have = set(load_map()['fname'])

    downloaded = []
    for _, job in recent_df.iterrows():
        lc_path = job['lightcurve']
        match = _LC_PATTERN.match(lc_path)
        if not match:
            print(f'Could not parse filename from: {lc_path}')
            continue
        fname = match.group(1)
        if fname in have:
            continue
        target = followup.nearest_target(float(job['ra']), float(job['dec']))
        if target is None:
            print(f'Skipping {fname}: no follow-up target at ({job["ra"]}, {job["dec"]})')
            continue

        url = DOWNLOAD_BASE_URL + lc_path
        response = requests.get(url, auth=HTTP_AUTH)
        if response.status_code != 200:
            print(f'Failed to download {url}: {response.status_code} {response.reason}')
            continue
        dest = target.path / followup.ZFPS_DIR / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first, so a failed write never leaves a partial file.
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.part', delete=False) as f:
            f.write(response.content)
        os.replace(f.name, dest)
        print(f'Downloaded: {dest}')
        downloaded.append(str(dest))
        have.add(fname)

    return downloaded
