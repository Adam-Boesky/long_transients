# Follow-up status: one per object

## What changed

The follow-up status (`to queue` / `queued` / `observed`) used to be a column on
each author's row in `annotations`, so three people could hold three different
answers about the same source, and the same object seen in two overlapping
fields could be queued twice. It is now a fact about the object:

- **Table** `followup (source_uuid PRIMARY KEY, status, author, updated_at)` in
  `annotations.sqlite`, plus `followup_history` (one row per change). Schema in
  `source_db/serving/schema.py`; logic in `api/followup.py`.
- **Per object, not per source.** A `PUT /api/sources/{uuid}/followup` writes the
  row for every source sharing that uuid's `group_id` (via
  `CatalogFilters.group_members`), so reading any detection answers for the
  object. `author` is whoever set it last; the UI shows that under the control.
- **Last writer wins** at runtime. The ordering below is only for merging values
  that were set independently.
- `annotations.status` is no longer written or returned. The column stays in
  the table (SQLite cannot drop it cheaply, and `history` keeps the old trail).
  `/api/annotations/export` still dumps it because it does `SELECT *`.
- Filters (`status_in` / `status_out` on `/api/sources`, `/api/table`,
  `/api/sources-locate`, `/api/sources-next-unlabelled`) now read `followup`.
- `/api/table/annotations` carries a `followup` map beside `annotations`;
  `SourceDetail.followup` is the object's row or null.

## Merge order

When two statuses for one object have to be reconciled (the old per-author
values, or group-mates that drifted apart between releases), the later stage
wins: **observed > queued > to queue > nothing**. An object someone observed is
observed whatever anyone else said; one already in a queue must not be queued
again. Within a stage the most recent edit is credited as `author`.
`api.followup.merged` is the one place this lives.

## Migrating the deployed database

When the production `annotations.sqlite` comes back from the server:

1. **Back it up** before anything else (`cp annotations.sqlite annotations.pre-followup.sqlite`).
2. **Backfill from the per-author column.** Automatic: the first time the new
   API opens the database (`api.db._bootstrap_annotations`), it runs
   `followup.backfill_from_annotations`, which per source keeps the
   furthest-along status anyone set (order above), inserts it into `followup`,
   and clears `annotations.status` in the same transaction. One source of
   truth afterwards; a crash mid-way rolls both back; a later open finds the
   column empty and does nothing. `history` keeps every per-author value.
3. **Propagate across groups.** The backfill knows nothing about `group_id`, so
   two detections of one object can disagree after it. With the matching
   `science.sqlite` and catalog in place:
   ```
   python -m api.followup sync --db path/to/annotations.sqlite
   ```
   rewrites every group to its furthest-along member (and credits whoever set
   that). Re-run after any science rebuild that changes group membership.
4. **Check**: `SELECT status, COUNT(*) FROM followup GROUP BY status` against
   the backup's `SELECT status, COUNT(DISTINCT source_uuid) FROM annotations
   WHERE status IS NOT NULL GROUP BY status`. Merging only moves objects
   forward and group propagation only adds rows, so `observed` can only grow
   and `to queue` can only shrink.
5. Deploy the API and the rebuilt frontend together: the old frontend sends
   `status` inside the annotation PUT, which the new API ignores.

Dry run on 2026-11-08 against a copy of the local `Data/db/annotations.sqlite`:
one `to queue` carried over, no disagreements, nothing left in the old column,
history untouched. Note that the test suite's app lifespan opens the
configured annotations database, so the local copy was migrated the first
time the tests ran with this code.

## Still open

- `/api/annotations/export` dumps `annotations.status`, now always NULL; drop
  the column from the export once nobody needs the old shape.
- The Streamlit `classify_app` does not read `status`, so nothing to do there.
