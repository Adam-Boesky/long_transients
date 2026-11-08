# Follow-up status: one per object

Status (`to queue` / `queued` / `observed`) is a fact about the object, not an
author's opinion, so it moved out of `annotations.status` into its own table.

- `followup (source_uuid PRIMARY KEY, status, author, updated_at)` in
  `annotations.sqlite`, plus `followup_history`. Logic in `api/followup.py`.
- A `PUT /api/sources/{uuid}/followup` writes every detection sharing the
  source's `group_id`, so an object is never queued twice from two fields.
  Last writer wins; `author` is who.
- Merge order, used only when values set independently must be reconciled:
  **observed > queued > to queue > nothing**. Later stage wins; within a stage,
  the latest edit is credited.

## Migrating the deployed database

1. Back up `annotations.sqlite`.
2. Start the new API. On first open it folds the per-author column into
   `followup` (merge order above) and clears that column in the same
   transaction. Idempotent; `history` keeps the old values.
3. `python -m api.followup sync` so detections of one object agree. Re-run
   after any catalog rebuild that changes `group_id`.
4. Sanity check: `SELECT status, COUNT(*) FROM followup GROUP BY status`.
   `observed` can only have grown and `to queue` only shrunk versus the backup.
5. Deploy API and frontend together: the old frontend sends `status` inside
   the annotation PUT, which the new API ignores.
