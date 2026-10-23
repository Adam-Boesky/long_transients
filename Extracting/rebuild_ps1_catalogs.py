"""Rebuild each quadrant's PSTARR.hdf5 from the local PS1 parquet over the union of its band images.

Extraction used to query PS1 from the first band's image only, so strips of the other bands' (offset) deep stacks
had no PS1 to match and their detections were flagged ZTF-only (plan_filter_review.md #12). ZTF catalogs are left
untouched. The old catalog is kept as PSTARR_pre_rebuild.hdf5. Re-run association + merge afterwards.

    python -m Extracting.rebuild_ps1_catalogs --quad-list Data/final_footprint_quadrants_10_20_2026.csv --workers 8
    python -m Extracting.rebuild_ps1_catalogs --quads 000683_06_4 --out-name PSTARR_rebuilt.hdf5   # side copy
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from Extracting.Catalogs import PSTARR_Local_Catalog
from Extracting.utils import coord_range_from_wcs, get_data_path, prepare_table_for_write, union_coord_range

BANDS = ('g', 'r', 'i')
DONE_MARKER = 'ps1_rebuilt.txt'


def quadrant_box(qdir: str):
    """Bands extracted in a quadrant and the padded union of their image boxes."""
    bands, ranges = [], []
    for band in BANDS:
        wcs_path = os.path.join(qdir, 'WCSs', f'ZTF_{band}_wcs.pkl')
        mask_path = os.path.join(qdir, 'nan_masks', f'ZTF_{band}_nan_mask.npy')
        if not (os.path.exists(os.path.join(qdir, f'ZTF_{band}.hdf5')) and os.path.exists(wcs_path) and os.path.exists(mask_path)):
            continue
        with open(wcs_path, 'rb') as f:
            wcs = pickle.load(f)
        shape = np.load(mask_path, mmap_mode='r').shape  # nan mask has the image's shape
        bands.append(band)
        ranges.append(coord_range_from_wcs(wcs, shape))
    if not bands:
        return bands, None, None
    ra_range, dec_range = union_coord_range(ranges)
    return bands, ra_range, dec_range


def rebuild_quadrant(qdir: str, out_name: str = 'PSTARR.hdf5', overwrite: bool = False) -> str:
    name = os.path.basename(qdir.rstrip('/'))
    in_place = out_name == 'PSTARR.hdf5'
    if in_place and not overwrite and os.path.exists(os.path.join(qdir, DONE_MARKER)):
        return f'PS1_SKIP {name} already rebuilt'
    try:
        t0 = time.time()
        bands, ra_range, dec_range = quadrant_box(qdir)
        if not bands:
            return f'PS1_SKIP {name} no extracted bands'
        data = PSTARR_Local_Catalog(ra_range, dec_range, catalog_bands=bands).data

        # Write to a temp file first so an interrupted run never leaves a truncated catalog
        tmp = os.path.join(qdir, f'.{out_name}.part{os.getpid()}')
        prepare_table_for_write(data).write(tmp, format='hdf5', path='data', serialize_meta=True, overwrite=True)
        if in_place:
            old, backup = os.path.join(qdir, 'PSTARR.hdf5'), os.path.join(qdir, 'PSTARR_pre_rebuild.hdf5')
            if os.path.exists(old) and not os.path.exists(backup):
                os.replace(old, backup)
        os.replace(tmp, os.path.join(qdir, out_name))
        if in_place:
            with open(os.path.join(qdir, 'ps1_source.txt'), 'w') as f:
                f.write('local\n')
            with open(os.path.join(qdir, DONE_MARKER), 'w') as f:
                f.write(f'bands={"".join(bands)} ra={ra_range[0]:.6f},{ra_range[1]:.6f} dec={dec_range[0]:.6f},{dec_range[1]:.6f}\n')
        return f'PS1_OK {name} bands={"".join(bands)} rows={len(data)} {time.time() - t0:.0f}s'
    except Exception as e:
        return f'PS1_FAIL {name} {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quad-list', help='CSV with fieldid, ccdid, qid columns')
    ap.add_argument('--quads', nargs='+', help='quadrant directory names, e.g. 000683_06_4')
    ap.add_argument('--out-name', default='PSTARR.hdf5', help='write a side copy under another name instead of replacing')
    ap.add_argument('--overwrite', action='store_true', help='rebuild quadrants already marked as rebuilt')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    if args.quads:
        names = args.quads
    else:
        df = pd.read_csv(args.quad_list)
        names = [f'{int(r.fieldid):06d}_{int(r.ccdid):02d}_{int(r.qid)}' for r in df.itertuples()]
    catalog_dir = os.path.join(get_data_path(), 'catalog_results')
    qdirs = [os.path.join(catalog_dir, n) for n in names if os.path.isdir(os.path.join(catalog_dir, n))]
    print(f'Rebuilding PS1 for {len(qdirs)} / {len(names)} quadrants -> {args.out_name} with {args.workers} worker(s)', flush=True)

    def _run(q):
        return rebuild_quadrant(q, args.out_name, args.overwrite)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(_rebuild_star, [(q, args.out_name, args.overwrite) for q in qdirs], chunksize=1):
                print(msg, flush=True)
    else:
        for q in qdirs:
            print(_run(q), flush=True)


def _rebuild_star(args):
    return rebuild_quadrant(*args)


if __name__ == '__main__':
    main()
