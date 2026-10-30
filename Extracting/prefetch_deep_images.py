"""Download ZTF deep reference images into ztf_data/ ahead of a running list extraction.

Extraction workers that hit IRSA's intermittent 503s sleep through backoff while holding a CPU slot.
This fetches the same files (same URL and path as `ZTF_Catalog.download_image`) so extraction finds
them cached. Lists are walked from the end (or, with --forward, from --skip onwards) to stay ahead of
extraction, and each file is
written to a temp name and renamed into place only once valid.

    python -m Extracting.prefetch_deep_images part1.csv part2.csv part3.csv --threads 6
"""
import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

import pandas as pd
import requests

try:
    from utils import get_data_path, get_credentials, fitscheck_valid
except ModuleNotFoundError:
    from .utils import get_data_path, get_credentials, fitscheck_valid

MAX_ATTEMPTS = 40
_lock = threading.Lock()
_counts = {'ok': 0, 'cached': 0, 'extracted': 0, '404': 0, 'fail': 0}


def image_url_and_name(fieldid: int, ccdid: int, qid: int, band: str):
    pf, cc = str(fieldid).zfill(6), str(ccdid).zfill(2)
    name = f'ztf_{pf}_z{band}_c{cc}_q{qid}_refimg.fits'
    url = f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep/{pf[:3]}/field{pf}/z{band}/ccd{cc}/q{qid}/{name}'
    return url, name


def _log(key: str, msg: str, total: int):
    with _lock:
        _counts[key] += 1
        done = sum(_counts.values())
        print(f'PREFETCH_{key.upper()} {msg} [{done}/{total}] {_counts}', flush=True)


def fetch(job, out_dir: str, auth, total: int):
    fieldid, ccdid, qid, band = job
    quad = f'{str(fieldid).zfill(6)}_{str(ccdid).zfill(2)}_{qid}'
    if os.path.exists(os.path.join(get_data_path(), 'catalog_results', quad, f'ZTF_{band}.hdf5')):
        return _log('extracted', f'{quad} {band}', total)
    url, name = image_url_and_name(fieldid, ccdid, qid, band)
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and fitscheck_valid(path):
        return _log('cached', name, total)

    tmp = os.path.join(out_dir, f'.{name[:-5]}.part{os.getpid()}.fits')
    for attempt in range(MAX_ATTEMPTS):
        try:
            # Deep stacks are public; log in only if IRSA asks
            r = requests.get(url, stream=True, timeout=(10, 60))
            if r.status_code in (401, 403):
                r = requests.get(url, auth=auth, stream=True, timeout=(10, 60))
            if r.status_code == 404:
                return _log('404', url, total)
            if r.status_code == 200:
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                if fitscheck_valid(tmp):
                    os.replace(tmp, path)
                    return _log('ok', name, total)
        except requests.exceptions.RequestException:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(min(10 * 1.5 ** attempt, 120) * random.uniform(0.8, 1.2))
    _log('fail', url, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lists', nargs='+', help='quadrant-list CSVs (fieldid, ccdid, qid, bands_to_extract)')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--forward', action='store_true', help='walk each list from the start instead of the end')
    ap.add_argument('--skip', type=int, default=0, help='with --forward, skip this many quadrants of each list')
    args = ap.parse_args()

    # Each list walked from the end (default) or from --skip forwards, then interleaved across lists
    per_list = []
    for path in args.lists:
        q = pd.read_csv(path)
        q = q.iloc[args.skip:] if args.forward else q.iloc[::-1]
        per_list.append([(int(r.fieldid), int(r.ccdid), int(r.qid), b) for r in q.itertuples() for b in r.bands_to_extract])
    jobs = [j for group in zip_longest(*per_list) for j in group if j is not None]

    out_dir = os.path.join(get_data_path(), 'ztf_data')
    auth = tuple(map(str, get_credentials('irsa_login.txt')))
    print(f'Prefetching {len(jobs)} images into {out_dir} with {args.threads} threads', flush=True)
    def safe_fetch(j):
        try:
            fetch(j, out_dir, auth, len(jobs))
        except Exception as e:  # one bad file must not stop the rest
            _log('fail', f'{j} {type(e).__name__}: {e}', len(jobs))

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(safe_fetch, jobs))
    print(f'Done: {_counts}', flush=True)


if __name__ == '__main__':
    main()
