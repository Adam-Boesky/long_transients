# Plan: source database + dynamic web rendering

Replace 56 GB of pre-rendered analysis-page PDFs with a queryable database and a
web client that draws the panels from data.

A private tool for three people to sort and tag candidates. Not a public site.

All numbers measured on `filter_results_9_13_2026_full_catalog` unless marked
*estimated*.

---

## 1. Decisions

| | |
|---|---|
| **identity** | `uuid4` primary key + a persisted identity map; coordinate stem kept as a display slug |
| **images** | not stored -- HiPS in the browser |
| **PDFs** | generated on demand, never in bulk |
| **light curves** | store exactly what `Light_Curve.get_lc()` already returns |
| **catalogs** | matching infrastructure, stay on the laptop |
| **science data** | immutable on the server; annotations are the only writes |

Open: per-source light-curve bytes (§6.4), Streamlit vs React (§8).

---

## 2. Why

| | PDFs | database |
|---|---|---|
| storage | **55.8 GB** (169,052 x 0.33 MB) | **~6 GB**, 0.3 GB of it on the server |
| regenerate after a plot change | ~67 days of compute | re-run a build script |
| upstream services needed to view a source | 6, live | 0 |
| "SIMBAD type CV* and dmag > 3" | impossible | a `WHERE` clause |

Storage is the weakest of these. The decisive argument is that a rendered PDF
cannot be updated incrementally: adding the SIMBAD block to `get_info_string`
invalidated every page already on disk. Second is decoupling from six flaky
upstreams -- IRSA went down mid-development and made pages ungenerable at all.

---

## 3. The boundary: catalogs are not the database

**Catalogs** (SIMBAD, DESI, SDSS, AGN-DB, TNS, AllWISE) are matching
infrastructure. They exist so crossmatching is cheap and offline, and they never
leave the laptop. **The database** holds only what the matching produced: one
flattened record per source.

| | |
|---|---|
| slim catalogs 3.8 GB + raw downloads 29 GB + ZTF refimgs 225 GB | **~258 GB, laptop** |
| per-source results as SQLite | **288 MB, ships** |

Three consequences:

**Everything must be pre-flattened.** The server can never join against a
catalog, so any question the site can ask must already be a column. This is
exactly what `enrich_combined_tabs.py` produces. The cost is that new questions
need a rebuild and republish, not a query change -- so make republishing cheap
and boring rather than trying to anticipate every column.

**Science is read-only on the server.** Two schemas, different lifecycles,
enforced by grants rather than convention:

| | `science` | `annotations` |
|---|---|---|
| written by | publish, wholesale | users, continuously |
| app DB role | `SELECT` | `SELECT`, `INSERT` |
| on republish | dropped and reloaded | untouched |
| master copy | the laptop | **the server** -- back this up |

**Labels should record a `row_hash`.** UUIDs already keep a label attached to
the right object across rebuilds, so nothing breaks on republish. The one thing
they don't capture is whether the *data underneath* a label changed -- someone
tags a source at dmag 4.5, a photometry fix lands, and it is now dmag 0.3.
Since `science` is dropped and reloaded, the old values are gone, so a
`row_hash` stored on the label is the cheap way to detect this without keeping
any history.

One column, computed at build. It only matters if the annotations later feed
something quantitative (a recovery rate or purity versus dmag, where labels made
against two photometry versions would bias the result); for picking a shortlist
to observe, it is ceremony. It cannot be backfilled, which is the only reason to
decide early.

---

## 4. Identity

Coordinate stems are **not unique**, so they cannot be the key:

```
0.ecsv:               158,876 rows -> 154,970 unique     3,906 collisions (2.5%)
across all three tables                                  3,933 collisions
```

Two causes. *4-decimal rounding*: 0.0001 deg is ~0.36 arcsec here, so separated
sources merge (colliding pairs sit a median 0.094 arcsec apart). More precision
helps -- 5 dp -> 331, 6 dp -> 233 -- then plateaus. *Identical association
positions*: 431 rows in 215 groups share byte-identical `(ra, dec)` while
differing in ~89 measurement columns including `ZTF_rPSFMag`. No precision can
separate those; they are the plateau.

The UUID must also be **random, not derived**, because there is no stable input
to hash:

- the only ID column, `PSTARR_PanSTARR_ID`, is itself non-unique (151,551 /
  158,876) and absent for ZTF-only sources;
- position is not stable across pipeline changes -- `filter_results_kde`,
  `..._gaia_propagation`, `..._full_catalog` differ in astrometric handling, so
  centroids move;
- the measurement columns that *do* discriminate are the ones that change.

So: assign `uuid4` on first sight and persist it. On rebuild, carry UUIDs
forward by **positional crossmatch against the previous identity table** at
~0.5 arcsec -- which `_base.LocalCatalog.match_nearest` already does. New rows
get fresh UUIDs; unmatched old rows are retired, never deleted, since
annotations may point at them.

`science.identity` lives in the `science` schema but is **never dropped on
publish** -- it is the only link between annotations and sources across
releases. Back it up alongside `annotations`.

The 215 identical-position groups are handled in the UI by a duplicate view, not
forced apart in the schema. They may also be worth a look as a possible
`combine_filtered_tabs.py` issue.

---

## 5. Schema

```sql
-- persisted across releases, never dropped
science.identity(source_uuid PK, ra, dec, coord_slug,
                 first_release, last_release, retired_at)

-- dropped and reloaded on publish
science.release(release_id PK, built_at, git_sha, pipeline_dir, notes)
science.sources(
    source_uuid UUID PK REFERENCES science.identity,
    ra, dec, coord_slug, catalog_tier,   -- 0 | 1 | 1_wide_association
    release_id, row_hash,
    <~300 measurement + crossmatch columns>
)
science.light_curves(source_uuid FK, mjd, survey, band, mag, magerr, snr)
science.spectra(source_uuid FK, origin, wavelength BYTEA, flux BYTEA)

-- append-only, never dropped
annotations.members(user_id PK, email, display_name)
annotations.buckets(bucket_id PK, name, description)
annotations.labels(label_id PK, source_uuid, user_id, bucket_id,
                   row_hash, created_at, note)   -- row_hash: see §3
```

- The ~300 measurement columns stay **wide and denormalised** -- right for
  "filter on any column", and only 288 MB. A filter+sort+paginate query measured
  **33 ms** on untuned SQLite, so this tier will never be the bottleneck. Index
  the match flags and the default sort column.
- `spectra` arrays are packed float32 blobs; a spectrum is an array, not 3,800
  rows.
- `labels` is append-only -- a correction is a new row, not an `UPDATE`.
- The duplicate view is just `WHERE coord_slug = $1`.

**Publishing.** Build locally, ship one artifact, load into `science_new` and
swap schemas in one transaction. Never a partial publish; `annotations` is in
its own schema so the swap cannot touch it.

**Pulling annotations back.** Export the append-only rows to parquet/CSV and
analyse on the laptop. The export must be self-contained -- analysis should
never require the server to be alive.

---

## 6. Data tiers

### 6.1 Images -- not stored

All six panels are static functions of sky position (the ZTF files are
`*_refimg.fits`, reference imagery), and all six surveys already exist as HiPS,
verified against the CDS registry: `CDS/P/ZTF/DR7/{g,r,i,color}` and
`CDS/P/PanSTARRS/DR1/{g,r,i,y,color-i-r-g}`. Aladin Lite renders these
in-browser with zero bytes stored.

Rubin reached the same conclusion -- on-demand generation into an object store
with 30-day expiry, "to avoid unbounded storage requirements for cutouts that
are no longer of interest" (DMTN-208).

For when the UI phase arrives: measured on one 200x200 PanSTARRS cutout, FITS
float32 160-177 KB, RICE 47.8 KB, PNG8 25.7 KB, JPEG q85 10.5 KB, **WebP q80
8.8 KB** -- so ~31 KB for all six panels, 5.2 GB for the full catalog or
~155 MB for a 5,000-source shortlist. Note HiPS is not bit-identical to your
cutouts (ZTF DR7 HiPS is a different reference build at 0.805 vs 1.01
arcsec/px): fine for visual context, not for "the pixels the photometry came
from". You hold all 225 GB of refimgs, so an authoritative renderer stays
possible.

### 6.2 PDFs -- on demand

`Source.plot_everything()` + `store_src_plots.save_src_plot()` already render a
page; the only change is triggering it from a source page instead of a 169k
batch job. ~0.6 min/page occasionally, versus 67 days to pre-render everything.

Export needs the six upstreams live, so it should fail visibly rather than
silently emit blank panels. Once light curves and spectra are stored, only the
cutouts need the network -- which makes export much more robust, and is a reason
to do §6.4 before wiring up export.

### 6.3 Spectra

An SDSS `spec-lite` file is **214 KB**; 28,847 sources have an SDSS match and
42,342 a DESI match. Store downsampled `(wavelength, flux)` float32 arrays
binned to ~1,000 points -- ~8-10 KB each, **~710 MB for all ~71k** *(estimated)*
-- and keep the SAS/SPARCL URL for full resolution.

Worth it mainly to remove the SDSS skyserver and SPARCL dependencies, two of the
flakiest in the stack and the cause of the `ConnectTimeout` failures already
seen in `store_src_plots.py`.

### 6.4 Light curves

**What gets stored:** exactly what the pages already plot.
`Light_Curve.get_lc()` queries all eight catalogs in parallel, filters to the
closest object per catalog, `vstack`s them and returns one table per source.
Storing light curves means persisting that table -- nothing new to acquire.

Catalogs are the existing default in both `Light_Curve.__init__` and
`store_src_plots.DEFAULT_LC_CATALOGS`:

```
['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom']
```

Schema is already fixed by `ALL_LC_COLNAMES` and `ALL_BAND_DF`: `mjd`, `ra`,
`dec`, per-survey `*_id`, and `<survey>_<band>_mag` / `_magerr` / `_snr`.

**Acquisition:** one resumable batch pass calling that same code per source.
Far cheaper than the plotting run -- it skips cutouts, spectra and matplotlib,
which dominate the 0.6 min/page. Still 169k x eight cone searches, so
overnight-class, and the last place the pipeline touches IRSA. Run it once and
the website never touches IRSA again.

**Storage shape:** the native table is tall but sparse -- one row per epoch with
~30 magnitude columns, only one or two populated, since each row comes from a
single survey. Store it melted as `(source_uuid, survey, band, mjd, mag,
magerr, snr, catalog_id)`. Nothing is lost: `ALL_BAND_DF` is exactly the
`column -> (band, survey)` map needed to pivot back to the wide form the
plotting code wants.

**One file, not 256 buckets.** The earlier bucketing plan was sized against a
~5 GB guess. Measured, ZTF -- the dominant time series -- is **0.14 GB / ~4.9M
rows** for all 169,052 sources. Even with the other catalogs the tier stays
well under a gigabyte, so a single parquet sorted by `source_uuid` is simpler
and sufficient: row-group statistics on the sorted key give point lookup by
predicate pushdown, and it ships to the server as one artifact alongside
`sources.parquet`.

**Two layers on disk.** `raw/<catalog>/` holds the fetch-once native output
(for ZTF that is a HATS collection of a few thousand partition parquets, with
the nested `lightcurve` column intact); `light_curves.parquet` is the
normalised single file the app reads.

### 8.5 ZTF: measured

Bulk access is via the DR24 HATS collection on public S3, not the light-curve
API -- that API takes one position per request, and IRSA's own docs redirect
anything over ~1000 light curves to HATS.

```
full run          ~3.3 h, ~2 GB peak      (0.07 s/source, streamed)
output            ~0.14 GB normalised, ~4.9M rows
objectids/source  min 1, median 2, max 8  (n_neighbors=25, no truncation)
epochs/source     min 1, median 5, max 2,944
```

Three things that are load-bearing and easy to get wrong:

- **`write_catalog`, never `.compute()`.** The latter materialises every
  matched row; with the nested column that reached 17.9 GB before being killed.
- **All ten columns.** `catflags` drives the quality cut
  (`catflags & 6141`, ZTF Explanatory Supplement §10.3, see
  `Source_Analysis.ZTF_Local_LC`) and `clrcoeff` is carried through by the
  existing reader. Neither can be recovered later without re-running.
- **`n_neighbors` is a cap, not a target.** ZTF assigns one objectid per band
  *and* per field overlap; the observed max is 8. Too low silently truncates a
  source's light curve.

Verified against `Light_Curve.get_lc()`: where both return an objectid the
photometry is identical to 5e-08 (float32 precision), including a 2,940-epoch
source. Every difference is an objectid with 1-2 epochs that the old API did
not return at all, so the bulk path is a strict superset.

**Comparing the two paths correctly matters.** Pairing must be per-objectid and
positional after sorting. ZTF repeat exposures come as close as 39 s apart
while the hmjd-mjd offset is 80-200 s, so any nearest-in-time matching
cross-pairs adjacent epochs and manufactures ~0.3 mag "discrepancies" that are
not real.

**Runs in a separate conda env.** `hats` needs numpy>=2.3 and astropy>=7;
`long_transients` is on numpy 1.26 / astropy 6.1 and upgrading it would drag
the whole pipeline along. The backfill only emits parquet, so it gets `ztf_hats`:

    conda run -n ztf_hats python -m source_db.lightcurves.ztf

---

## 7. Budget

| tier | size | where |
|---|---|---|
| catalog | 288 MB | server |
| spectra | ~710 MB *(est.)* | server |
| light curves | ~5 GB *(est., unmeasured)* | object storage |
| images | 0 | HiPS / CDS |
| annotations | MB | server -- **master copy** |
| catalogs + refimgs | 258 GB | laptop only |

Server-side ~6 GB against 55.8 GB of PDFs, and every byte queryable and
re-renderable.

---

## 8. Stack

Three users means no scaling or CDN concerns, trivial auth (email allowlist),
annotation volume in the tens of thousands of rows, and comfortable free tiers.

| layer | choice |
|---|---|
| build | existing pipeline -> parquet (151 MB) |
| database | Postgres, two schemas (Neon / Supabase free tier) |
| API | PostgREST / Supabase auto-REST |
| frontend | Streamlit (existing) or React -- see below |
| bulk blobs | object storage (Cloudflare R2 / Backblaze B2) for light curves |

**Frontend.** At three users doing tag-and-sort, `Source_Analysis/classify_app/`
already *is* this application -- `filters.py`, `stores.py`, `identity.py`,
`pages_table.py`, a working multi-page UI. Repointing it from
`manifest.parquet` + PDFs to the database gets most of the benefit for a
fraction of the work. React is right if you want an interface you will keep for
years or Streamlit becomes limiting; it is not required by the scope. Phases 1-2
are shared either way, so this can be decided later.

---

## 9. Phases

1. **Identity + catalog tier.** `science.identity` with uuid4 and the
   crossmatch-forward rebuild step; build `sources.parquet` / `sources.sqlite`;
   point `classify_app/filters.py` at it.
   *Deferred:* `Source_Analysis/test_source_db.py` covering `assign_uuids` and
   `update_identity` -- coincident positions, jitter, retirement, NaN coords.
   Worth it because `identity.parquet` is the one artifact that cannot be
   regenerated and whose bugs fail silently.
2. **Publish path.** Two schemas, grants, release table, `row_hash`, scripted
   atomic publish.
3. **Measure light curves** (`profile_lc.py`, ~100 sources).
4. **Light-curve + spectrum tiers**, backfilled in dmag order.
5. **Frontend**, if React.

Phases 1-2 are the foundation and worth doing regardless of whether 5 happens.
The only piece of phase 2 with a deadline is `row_hash` on labels, which cannot
be backfilled once annotation starts.

---

## References

- Rubin cutout strategy, DMTN-208: <https://dmtn-208.lsst.io/>
- IVOA SODA experience, SQR-063: <https://sqr-063.lsst.io/>
- Aladin Lite v3: <https://aladin.cds.unistra.fr/AladinLite/doc/>
- HiPS: <https://aladin.cds.unistra.fr/hips/>
- hips2fits: <https://alasky.cds.unistra.fr/hips-image-services/hips2fits>
- ESO SODA: <http://archive.eso.org/cms/eso-data/programmatic-access.html>
