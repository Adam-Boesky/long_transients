import sys, os
sys.path.insert(0, '/n/home04/aboesky/berger/long_transients')
import numpy as np
from astropy.table import Table
from Extracting.utils import get_data_path
from Source_Analysis import filtered_out_summary as FS

R = os.path.join(get_data_path(), 'filter_results_9_21_2026_full_catalog_fixed_pm')
OLD, NEW = os.path.join(R, 'combined'), os.path.join(R, 'combined_with_reasons')

overall = True
for fname in ('0.hdf5', '1.hdf5', '1_wide_association.hdf5'):
    po, pn = os.path.join(OLD, fname), os.path.join(NEW, fname)
    if not (os.path.exists(po) and os.path.exists(pn)):
        print(f'{fname}: MISSING on one side'); overall = False; continue
    o = Table.read(po, path='data'); n = Table.read(pn, path='data')
    print(f'\n=== {fname} ===')
    print(f'  rows      old {len(o):>8,}   new {len(n):>8,}   {"SAME" if len(o)==len(n) else "DIFFER"}')
    oc, nc = set(o.colnames), set(n.colnames)
    added, removed = nc - oc, oc - nc
    print(f'  columns   old {len(oc):>8}   new {len(nc):>8}   added {len(added)}  removed {len(removed)}')
    ok = len(o) == len(n) and not removed and added == set(FS.ALL_COLUMNS)
    if removed:
        print(f'    !! removed: {sorted(removed)}')
    if added != set(FS.ALL_COLUMNS):
        print(f'    !! unexpected added: {sorted(added - set(FS.ALL_COLUMNS))}')
        print(f'    !! missing expected: {sorted(set(FS.ALL_COLUMNS) - added)}')

    # every shared column byte-identical
    diffs = []
    for c in sorted(oc):
        a, b = o[c], n[c]
        if a.dtype.kind in 'fc':
            same = np.allclose(np.asarray(a, float), np.asarray(b, float),
                               equal_nan=True, rtol=0, atol=0)
        else:
            same = np.array_equal(np.asarray(a).astype(str), np.asarray(b).astype(str))
        if not same:
            diffs.append(c)
    print(f'  shared columns identical: {len(oc)-len(diffs)}/{len(oc)}'
          + (f'   !! DIFFER: {diffs[:6]}' if diffs else ''))
    ok &= not diffs

    # the new columns
    if added:
        filled = 0
        for c in sorted(added):
            vals = np.asarray(n[c]).astype(str)
            nb = int(np.sum((vals != FS.NOT_FILTERED) & (vals != FS.NO_TABLE)))
            na = int(np.sum(vals == FS.NO_TABLE))
            filled += nb
            print(f'    {c:30} {nb:>7,} with a reason, {na:>7,} N/A')
        print(f'  total cells with a reason: {filled:,} of {len(n)*9:,}')
        ok &= filled > 0
    print(f'  --> {"PASS" if ok else "FAIL"}')
    overall &= ok

print(f'\n{"="*56}\nOVERALL: {"PASS" if overall else "FAIL"}')
