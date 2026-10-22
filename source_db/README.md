# `source_db/` — building the source database

Everything here exists to **build `Data/db/`**, the per-source results database
that backs the website. Nothing here is analysis code.

The distinction that matters:

| | what it is | where it lives | ships to the server? |
|---|---|---|---|
| `Source_Analysis/catalogs/` | bulk reference data for matching (SIMBAD, DESI, …) | laptop, ~258 GB | **no** |
| `source_db/` (this package) | one flattened record per source | `Data/db/`, ~0.3 GB | **yes** |

Design and rationale: `../plan_source_database.md`.

## Layout

```
source_db/
  paths.py              shared locations
  identity.py           source_uuid assignment + the identity map
  catalog_tier.py       the ~300-column source rows, from the enriched tables
  lightcurves/          per-catalog bulk fetch + normalisation
  spectra/              per-catalog bulk fetch + normalisation
  build_catalog_tier.py entry point
```

Output:

```
Data/db/
  sources.parquet       169,052 rows x 305 columns
  identity.parquet      source_uuid <-> position, persisted across rebuilds
  release.json          release id, git sha, inputs, counts
  raw/<catalog>/        native-format pulls, fetched once
  light_curves/         normalised, bucketed by source_uuid prefix
```

## Running

```bash
python -m source_db.build_catalog_tier --dry-run
python -m source_db.build_catalog_tier
```

## Two rules

**`identity.parquet` cannot be regenerated.** Every other artifact is a pure
function of the pipeline output. This one is the only link between an
annotation and a source across releases, so it is never dropped on rebuild and
must be backed up once annotation begins.

**Fetch once, normalise many times.** Raw pulls land under `raw/` in each
service's native schema and are never re-fetched. Normalisation into the
light-curve and spectrum tiers is a separate, re-runnable step, so a bug in the
melt logic costs minutes rather than another overnight backfill.
