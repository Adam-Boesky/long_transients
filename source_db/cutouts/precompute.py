"""Render postage stamps ahead of time, so nobody waits on IRSA or STScI.

Warming only, now that `cutout_sources` records which stamps exist: this
renders exactly the rows that table lists and writes nothing back. It used
to double as the discovery step for ZTF band availability, which meant a
stamp had to be rendered before the app could know it existed, and that
knowledge lived only in the serving database. `cutouts.ztf` resolves it
offline instead.

**Warm a working set, not the whole catalog.** Rendering all 169k sources
costs ~9 h and ~27 GB, for stamps that mostly nobody opens; a cold stamp is
0.4-0.9 s and the page already prefetches one source ahead, which hides it
while paging. The first couple of thousand in queue order is the part worth
having warm:

    python -m source_db.cutouts.precompute --limit 2000

Resumable: a stamp already cached is skipped, so a re-run costs one `stat`
each. `--max-gb` evicts down to a budget when it finishes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sqlite3
import sys
import time
from typing import Optional

from source_db.cutouts import render


def _one_stamp(uuid: str, survey: str, band: str, ra: float, dec: float,
               origin: str, stretch: str, percent: float) -> bool:
    """Render and cache one stamp. True if it produced an image."""
    path = render.cache_path(uuid, survey, band, stretch, percent,
                             render.source_fingerprint(ra, dec, origin))
    if render.cached(path) is not None:
        return True
    if survey == 'ztf':
        field, ccd, qid = (int(x) for x in origin.split('/'))
        img = render.fetch_ztf(field, ccd, qid, band, ra, dec)
    else:
        img = render.fetch_ps1(ra, dec, origin)
    if img is None or img.size == 0:
        return False
    render.store(path, render.render_png(img, stretch, percent))
    return True


def _one_source(rows: list[sqlite3.Row], stretch: str, percent: float) -> int:
    """Every stamp for one source. Returns how many are now cached."""
    n = 0
    for r in rows:
        try:
            if _one_stamp(r['source_uuid'], r['survey'], r['band'],
                          r['ra'], r['dec'], r['origin'], stretch, percent):
                n += 1
        except Exception as e:                      # noqa: BLE001
            print(f"  {r['survey']} {r['band']} {r['source_uuid'][:8]}: {e}",
                  file=sys.stderr)
    return n


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--db', default=None, help='science sqlite (default: api config)')
    ap.add_argument('--limit', type=int, default=None,
                    help='warm this many sources in queue order')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--stretch', default=render.DEFAULT_STRETCH)
    ap.add_argument('--percent', type=float, default=render.DEFAULT_PERCENT)
    ap.add_argument('--max-gb', type=float, default=None,
                    help='evict coldest entries down to this budget afterwards')
    ap.add_argument('--evict-only', action='store_true',
                    help='evict and exit, rendering nothing')
    args = ap.parse_args(argv)

    if args.evict_only:
        if args.max_gb is None:
            ap.error('--evict-only needs --max-gb')
        return _evict(args.max_gb)

    db = args.db
    if db is None:
        from api.config import SCIENCE_PATH
        db = str(SCIENCE_PATH)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    # Queue order, so `--limit` warms the head of the list the app shows.
    sql = ('SELECT c.source_uuid, c.survey, c.band, c.origin, s.ra, s.dec '
           'FROM cutout_sources c JOIN sources s USING (source_uuid) '
           'WHERE s.ra IS NOT NULL AND s.dec IS NOT NULL')
    if args.limit:
        sql += (' AND c.source_uuid IN (SELECT source_uuid FROM sources '
                f'ORDER BY source_uuid LIMIT {int(args.limit)})')
    by_source: dict[str, list[sqlite3.Row]] = {}
    for r in con.execute(sql + ' ORDER BY c.source_uuid'):
        by_source.setdefault(r['source_uuid'], []).append(r)
    con.close()

    total = sum(len(v) for v in by_source.values())
    print(f'{len(by_source):,} sources, {total:,} stamps, {args.workers} workers')

    t0 = time.time()
    done = rendered = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_one_source, rows, args.stretch, args.percent)
                for rows in by_source.values()]
        for fut in cf.as_completed(futs):
            rendered += fut.result()
            done += 1
            if done % 25 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f'  {done}/{len(by_source)}  {rendered} stamps  '
                      f'{rate:.1f} src/s  eta '
                      f'{(len(by_source) - done) / max(rate, 1e-9) / 60:.1f} min',
                      flush=True)

    dt = time.time() - t0
    print(f'done: {done} sources, {rendered} stamps in {dt / 60:.1f} min '
          f'({done / max(dt, 1e-9):.1f} src/s)')
    if args.max_gb is not None:
        _evict(args.max_gb)
    return 0


def _evict(max_gb: float) -> int:
    n, total = render.cache_size()
    print(f'cache: {n} stamps, {total / 1e9:.2f} GB (limit {max_gb} GB)')
    r = render.evict(int(max_gb * 1e9))
    if r['removed']:
        print(f"  evicted {r['removed']} stamps, freed {r['freed'] / 1e9:.2f} GB "
              f"-> {r['bytes_after'] / 1e9:.2f} GB")
    else:
        print('  nothing to evict')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
