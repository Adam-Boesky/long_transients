"""List the (field, ccdid, qid, band) quadrants that fed the merged field tables of the final filtering run.

merge_field() vstacks every {band}_associated.hdf5 in the field's quadrant dirs, so file presence is
enough; quadrant files newer than the merged table are flagged since merging doesn't overwrite.
"""
import os
import re
import sys

BASE = '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients'
RES = os.path.join(BASE, 'filter_results_9_21_2026_full_catalog_fixed_pm')
CATDIR = os.path.join(BASE, 'catalog_results')

fields = sorted(f for f in os.listdir(RES) if re.fullmatch(r'\d{6}', f)
                and os.path.exists(os.path.join(RES, f, '2_flowchart.pdf')))
fieldset = set(fields)
print(f'{len(fields)} finished fields', file=sys.stderr, flush=True)

quad_dirs = sorted(d for d in os.listdir(CATDIR)
                   if re.fullmatch(r'\d{6}_\d{2}_\d', d) and d[:6] in fieldset)
print(f'{len(quad_dirs)} quadrant dirs', file=sys.stderr, flush=True)

merged_mtime = {}
for f in fields:
    for band in 'gri':
        p = os.path.join(CATDIR, 'field_results', f'{f}_{band}.hdf5')
        merged_mtime[(f, band)] = os.path.getmtime(p) if os.path.exists(p) else None

with open(sys.argv[1], 'w') as out:
    out.write('fieldid,ccdid,qid,band,newer_than_merged\n')
    for d in quad_dirs:
        f, c, q = d.split('_')
        for band in 'gri':
            p = os.path.join(CATDIR, d, f'{band}_associated.hdf5')
            if not os.path.exists(p):
                continue
            mm = merged_mtime[(f, band)]
            newer = mm is None or os.path.getmtime(p) > mm
            out.write(f'{int(f)},{int(c)},{int(q)},{band},{int(newer)}\n')
