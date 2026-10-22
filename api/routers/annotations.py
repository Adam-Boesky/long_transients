"""The only writable tier: who labelled what.

Writes go to `annotations.sqlite`, never to the science database, so a science
rebuild can replace that file wholesale without migrating anything. Every edit
is also appended to `history`, because the labelling process is itself data
you will want to analyse later.
"""
from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from datetime import datetime, timezone

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from api.db import get_annotations, get_science
from api.models import (Annotation, AnnotationIn, Disagreement, Label, LabelIn,
                        LabelOrder, LabelPatch, MemberProgress, Progress, Stats, Verdict)

router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_annotation(row) -> Annotation:
    return Annotation(
        source_uuid=row['source_uuid'], author=row['author'], label=row['label'],
        confidence=row['confidence'], note=row['note'],
        tags=json.loads(row['tags']) if row['tags'] else [],
        status=row['status'],
        created_at=row['created_at'], updated_at=row['updated_at'])


@router.get('/sources/{source_uuid}/annotations', response_model=list[Annotation])
def list_annotations(source_uuid: str,
                     ann: sqlite3.Connection = Depends(get_annotations)):
    return [_row_to_annotation(r) for r in ann.execute(
        'SELECT * FROM annotations WHERE source_uuid=? ORDER BY author',
        (source_uuid,))]


@router.put('/sources/{source_uuid}/annotations/{author}', response_model=Annotation)
def put_annotation(source_uuid: str, author: str, body: AnnotationIn,
                   sci: sqlite3.Connection = Depends(get_science),
                   con: sqlite3.Connection = Depends(get_annotations)):
    """Create or update one author's annotation for one source."""
    if sci.execute('SELECT 1 FROM sources WHERE source_uuid=?',
                   (source_uuid,)).fetchone() is None:
        raise HTTPException(404, f'no source {source_uuid}')

    now = _now()
    tags = json.dumps(body.tags)
    prior = con.execute(
        'SELECT created_at FROM annotations WHERE source_uuid=? AND author=?',
        (source_uuid, author)).fetchone()
    created = prior['created_at'] if prior else now

    con.execute(
        'INSERT INTO annotations '
        '(source_uuid, author, label, confidence, note, tags, status, '
        ' created_at, updated_at) '
        'VALUES (?,?,?,?,?,?,?,?,?) '
        'ON CONFLICT(source_uuid, author) DO UPDATE SET '
        '  label=excluded.label, confidence=excluded.confidence, '
        '  note=excluded.note, tags=excluded.tags, status=excluded.status, '
        '  updated_at=excluded.updated_at',
        (source_uuid, author, body.label, body.confidence, body.note, tags,
         body.status, created, now))
    con.execute(
        'INSERT INTO history '
        '(source_uuid, author, label, confidence, note, tags, status, recorded_at) '
        'VALUES (?,?,?,?,?,?,?,?)',
        (source_uuid, author, body.label, body.confidence, body.note, tags,
         body.status, now))
    con.commit()

    return _row_to_annotation(con.execute(
        'SELECT * FROM annotations WHERE source_uuid=? AND author=?',
        (source_uuid, author)).fetchone())


@router.delete('/sources/{source_uuid}/annotations/{author}')
def delete_annotation(source_uuid: str, author: str,
                      con: sqlite3.Connection = Depends(get_annotations)):
    """Retract a label. The `history` row stays -- that is the point of it."""
    cur = con.execute('DELETE FROM annotations WHERE source_uuid=? AND author=?',
                      (source_uuid, author))
    con.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, 'no annotation for that source and author')
    return {'deleted': cur.rowcount}


_LABELS_IN_ORDER = (
    'SELECT name, created_by, created_at, color, '
    '  (SELECT COUNT(DISTINCT source_uuid) FROM annotations a '
    '   WHERE a.label = labels.name COLLATE NOCASE) AS n_sources '
    'FROM labels ORDER BY position IS NULL, position, created_at, rowid')


@router.get('/labels', response_model=list[Label])
def list_labels(ann: sqlite3.Connection = Depends(get_annotations)):
    """The label vocabulary in the order the sidebar shows it."""
    return [Label(**dict(r)) for r in ann.execute(_LABELS_IN_ORDER)]


@router.patch('/labels/{name}', response_model=Label)
def patch_label(name: str, body: LabelPatch,
                ann: sqlite3.Connection = Depends(get_annotations)):
    """Recolour a label for everyone."""
    color = body.color.lower() if body.color else None
    cur = ann.execute('UPDATE labels SET color=? WHERE name=?', (color, name))
    if cur.rowcount == 0:
        raise HTTPException(404, f'no label {name!r}')
    ann.commit()
    return Label(**dict(ann.execute(
        'SELECT name, created_by, created_at, color FROM labels WHERE name=?',
        (name,)).fetchone()))


@router.delete('/labels/{name}')
def delete_label(name: str, ann: sqlite3.Connection = Depends(get_annotations)):
    """Remove a label. 409 while any source is still filed under it."""
    if ann.execute('SELECT 1 FROM annotations WHERE label=? COLLATE NOCASE',
                   (name,)).fetchone() is not None:
        raise HTTPException(409, f'label {name!r} still has sources')
    cur = ann.execute('DELETE FROM labels WHERE name=?', (name,))
    if cur.rowcount == 0:
        raise HTTPException(404, f'no label {name!r}')
    ann.commit()
    return {'deleted': name}


@router.put('/labels/order', response_model=list[Label])
def reorder_labels(body: LabelOrder,
                   ann: sqlite3.Connection = Depends(get_annotations)):
    """Put the labels in the order given, which must name every one of them.

    409 when it does not: someone else added a label since you loaded the
    list, and quietly dropping it to the end would move it out from under
    them. The client refetches and you drag again.
    """
    current = {r['name'].lower(): r['name']
               for r in ann.execute('SELECT name FROM labels')}
    given = [n.lower() for n in body.names]
    if len(set(given)) != len(given) or set(given) != set(current):
        raise HTTPException(409, 'order does not match the current labels')
    ann.executemany('UPDATE labels SET position=? WHERE name=?',
                    [(i, current[n]) for i, n in enumerate(given)])
    ann.commit()
    return [Label(**dict(r)) for r in ann.execute(_LABELS_IN_ORDER)]


@router.post('/labels', response_model=Label, status_code=201)
def add_label(body: LabelIn, ann: sqlite3.Connection = Depends(get_annotations)):
    """Add a label everyone can file under. 409 if it exists in any case."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, 'label name is empty')
    now = _now()
    try:
        ann.execute('INSERT INTO labels (name, created_by, created_at) '
                    'VALUES (?,?,?)', (name, body.author, now))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f'label {name!r} already exists')
    ann.commit()
    return Label(name=name, created_by=body.author, created_at=now)


@router.get('/stats', response_model=Stats)
def stats(sci: sqlite3.Connection = Depends(get_science),
          ann: sqlite3.Connection = Depends(get_annotations)):
    """Progress: what the three of you have got through."""
    meta = dict(sci.execute('SELECT key, value FROM meta'))
    by_label = {(r['label'] or '(none)'): r['n'] for r in ann.execute(
        'SELECT label, COUNT(*) AS n FROM annotations GROUP BY label')}
    by_author = {r['author']: r['n'] for r in ann.execute(
        'SELECT author, COUNT(*) AS n FROM annotations GROUP BY author')}
    n_annotated = ann.execute(
        'SELECT COUNT(DISTINCT source_uuid) FROM annotations').fetchone()[0]
    return Stats(
        n_sources=int(meta.get('n_sources', 0)),
        n_with_lightcurves=int(meta.get('n_sources_with_lc', 0)),
        n_epochs=int(meta.get('n_epochs', 0)),
        n_with_spectra=int(meta.get('n_sources_with_spectra', 0)),
        n_spectra=int(meta.get('n_spectra', 0)),
        n_annotated=n_annotated, by_label=by_label, by_author=by_author,
        built_at=meta.get('built_at'))


@router.get('/progress', response_model=Progress)
def progress(limit: int = Query(200, ge=1, le=2000),
             sci: sqlite3.Connection = Depends(get_science),
             ann: sqlite3.Connection = Depends(get_annotations)):
    """The Progress page in one request.

    Aggregated here rather than in the browser because the alternative is
    shipping every annotation row to compute four numbers from it, and the
    cross-tab, the per-author table and the contested list all read the same
    rows -- one scan, three answers.
    """
    rows = [r for r in ann.execute(
        'SELECT source_uuid, author, label, note FROM annotations '
        "WHERE label IS NOT NULL AND TRIM(label) != '' "
        'ORDER BY source_uuid, author')]

    by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sources_by_author: dict[str, set[str]] = defaultdict(set)
    labels_by_author: dict[str, set[str]] = defaultdict(set)
    comments_by_author: dict[str, int] = defaultdict(int)
    labels_by_source: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        label, author = r['label'].strip(), r['author']
        by_label[label][author] += 1
        sources_by_author[author].add(r['source_uuid'])
        labels_by_author[author].add(label)
        if (r['note'] or '').strip():
            comments_by_author[author] += 1
        labels_by_source[r['source_uuid']].add(label)

    by_member = sorted(
        (MemberProgress(author=a, n_sources=len(s),
                        n_labels=len(labels_by_author[a]),
                        n_comments=comments_by_author[a])
         for a, s in sources_by_author.items()),
        key=lambda m: (-m.n_sources, m.author))

    # Most-contested first: a three-way split is more interesting than a
    # two-way one, and the list is truncated, so the order decides what the
    # page actually shows.
    contested = sorted(
        (uuid for uuid, labels in labels_by_source.items() if len(labels) > 1),
        key=lambda u: (-len(labels_by_source[u]), u))

    shown = contested[:limit]
    coords = {r['source_uuid']: r for r in sci.execute(
        'SELECT source_uuid, coord_slug, ra, dec FROM sources '
        f"WHERE source_uuid IN ({','.join('?' * len(shown))})", shown)
    } if shown else {}
    verdicts: dict[str, list[Verdict]] = defaultdict(list)
    for r in rows:
        verdicts[r['source_uuid']].append(
            Verdict(author=r['author'], label=r['label'], note=r['note']))

    meta = dict(sci.execute('SELECT key, value FROM meta'))
    return Progress(
        n_sources=int(meta.get('n_sources', 0)),
        n_labelled=len(labels_by_source),
        authors=sorted(sources_by_author),
        by_label={k: dict(v) for k, v in by_label.items()},
        by_member=by_member,
        n_disagreements=len(contested),
        disagreements=[
            Disagreement(
                source_uuid=u,
                coord_slug=coords[u]['coord_slug'] if u in coords else None,
                ra=coords[u]['ra'] if u in coords else None,
                dec=coords[u]['dec'] if u in coords else None,
                labels=sorted(labels_by_source[u]),
                verdicts=verdicts[u])
            for u in shown],
    )


@router.get('/annotations/export')
def export_annotations(
    fmt: str = Query('json', pattern='^(json|csv)$'),
    author: str | None = None,
    include_history: bool = False,
    ann: sqlite3.Connection = Depends(get_annotations),
):
    """Every annotation, for the offline analysis this whole exercise is for.

    `include_history` returns each edit rather than the current state, which is
    what you want if the question is how the labelling itself behaved.
    """
    table = 'history' if include_history else 'annotations'
    order = 'recorded_at' if include_history else 'updated_at'
    sql = f'SELECT * FROM {table}'
    params: list = []
    if author:
        sql += ' WHERE author=?'
        params.append(author)
    sql += f' ORDER BY source_uuid, {order}'
    rows = [dict(r) for r in ann.execute(sql, params)]
    for r in rows:
        if r.get('tags'):
            r['tags'] = json.loads(r['tags'])

    if fmt == 'json':
        return {'n': len(rows), 'table': table, 'rows': rows}

    if not rows:
        return PlainTextResponse('', media_type='text/csv')
    buf = io.StringIO()
    cols = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow({k: (','.join(v) if isinstance(v, list) else v)
                    for k, v in r.items()})
    return PlainTextResponse(
        buf.getvalue(), media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{table}.csv"'})
