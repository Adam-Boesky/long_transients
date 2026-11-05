"""Adding sources later must not re-fetch what is already on disk.

The trap this guards: chunks for most catalogs are *row-positional*, so
inserting a source shifts every later row into a different chunk. Checking
coverage per chunk id then sees every chunk as changed. ZTF is exempt because
its chunk id is a sky partition, which insertion does not move.
"""
from __future__ import annotations

import pandas as pd
import pytest

from source_db.lightcurves import _base

CAT = 'demo'
SIZE = 10


def _positions(n, prefix='src'):
    return pd.DataFrame({
        'source_uuid': [f'{prefix}-{i:05d}' for i in range(n)],
        'ra': [10.0 + i for i in range(n)],
        'dec': [0.0] * n,
    })


@pytest.fixture
def fetched(tmp_path):
    """A catalog with 30 sources already fetched, in 3 chunks."""
    pos = _positions(30)
    for cid, sub in _base.row_chunks(pos, SIZE):
        _base.write_raw(CAT, cid, sub, raw_dir=tmp_path)
        _base.write_coverage(CAT, cid, sub['source_uuid'], raw_dir=tmp_path)
    return tmp_path, pos


def test_rerun_with_no_changes_fetches_nothing(fetched):
    raw, pos = fetched
    assert _base.pending_chunks(CAT, pos, SIZE, raw) == []


def test_all_covered_unions_every_chunk(fetched):
    raw, pos = fetched
    assert _base.all_covered(CAT, raw) == set(pos['source_uuid'])


def test_appended_sources_cost_one_new_chunk(fetched):
    raw, pos = fetched
    grown = pd.concat([pos, _positions(5, 'new')], ignore_index=True)
    todo = _base.pending_chunks(CAT, grown, SIZE, raw)
    assert len(todo) == 1
    assert sum(len(s) for _, s in todo) == 5


def test_inserting_at_the_front_does_not_refetch_everything(fetched):
    """The regression this whole mechanism exists to prevent."""
    raw, pos = fetched
    grown = pd.concat([_positions(5, 'new'), pos], ignore_index=True)
    todo = _base.pending_chunks(CAT, grown, SIZE, raw)
    assert sum(len(s) for _, s in todo) == 5, (
        'only the new sources should be fetched, however they are ordered')
    assert all(u.startswith('new-') for _, s in todo for u in s['source_uuid'])


def test_new_chunk_ids_do_not_collide_with_existing_parts(fetched):
    raw, pos = fetched
    grown = pd.concat([pos, _positions(5, 'new')], ignore_index=True)
    existing = {p.stem.removeprefix('part-')
                for p in (raw / CAT).glob('part-*.parquet')}
    new_ids = {cid for cid, _ in _base.pending_chunks(CAT, grown, SIZE, raw)}
    assert not (existing & new_ids), 'a reused id would overwrite a fetched part'
    assert _base.next_chunk_index(CAT, raw) == 3


def test_many_new_sources_split_across_several_chunks(fetched):
    raw, pos = fetched
    grown = pd.concat([pos, _positions(25, 'new')], ignore_index=True)
    todo = _base.pending_chunks(CAT, grown, SIZE, raw)
    assert [cid for cid, _ in todo] == ['0003', '0004', '0005']
    assert sum(len(s) for _, s in todo) == 25


def test_no_resume_refetches_from_scratch(fetched):
    raw, pos = fetched
    todo = _base.pending_chunks(CAT, pos, SIZE, raw, resume=False)
    assert sum(len(s) for _, s in todo) == len(pos)
    assert [cid for cid, _ in todo] == ['0000', '0001', '0002']


def test_partially_covered_source_set_only_fetches_the_gap(fetched):
    raw, pos = fetched
    # Simulate a chunk that was never written: drop its sidecar and part.
    (raw / CAT / 'part-0001.sources').unlink()
    (raw / CAT / 'part-0001.parquet').unlink()
    todo = _base.pending_chunks(CAT, pos, SIZE, raw)
    got = {u for _, s in todo for u in s['source_uuid']}
    assert got == set(pos['source_uuid'][10:20]), (
        'only the sources from the missing chunk should come back')
