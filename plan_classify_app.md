# Plan: Multi-user hosted classification app

## Overview

Grow `Source_Analysis/classify_pages_app.py` from a single-user local Streamlit
script into a **private, hosted, multi-user** app for sorting analysis pages
into buckets.

Four requirements, and how each is met:

| # | Requirement | Approach |
|---|---|---|
| 1 | Accessible via a free URL | Streamlit Community Cloud, **private app** from a **private repo**, viewers by email invite (free tier allows one such app) |
| 2 | Users identify themselves; results stored per person | `members` tab in a Google Sheet + `st.selectbox`, mirrored to a query param |
| 3 | Results persist off-host | **Append-only** Google Sheet, written by a **service account** |
| 4 | Filter on enriched properties | Offline-built `manifest.parquet` pre-joined to `*_enriched.ecsv`, filtered in-app |

### Settled decisions

- **One bucket per page.** Single click, no multi-select, no tags. Clicking a
  bucket stays on the page; moving on is explicit (Next, or Next unbucketed).
- **Buckets are user-extensible and shared.** Adding one mid-session makes it
  available to everyone immediately.
- **Comments are per-person but visible to all.** You edit your own; you read
  everyone else's on the same page.
- **Private everything.** Private repo, invite-only app, and no
  "anyone-with-the-link" Drive permissions — the service account reads PDF bytes
  server-side. Retrofitting privacy later is far worse than starting with it:
  git history is permanent and shared Drive links are hard to recall.

---

## 1. Architecture

```
OFFLINE (laptop, re-run whenever new pages are generated)
  scripts/build_classify_manifest.py
    PDFs on disk ─┬─> upload new ones to Drive (private; service account can read)
                  └─> join to *_enriched.ecsv by coord_stem (1.5" sky-match fallback)
    → manifest.parquet   (~570 rows x ~45 cols, ~150 KB — committed to the app repo)

DEPLOYED (private Community Cloud app)
  streamlit_app.py
    manifest.parquet from the repo             (cached, cheap)
    one PDF at a time from Drive by file ID    (cached, ~500 KB each)
    buckets / members / labels via Sheets API  (service account, st.secrets)
```

The load-bearing idea is the **manifest**: a tiny pre-joined table carrying the
Drive file ID alongside every enriched property worth filtering on. The deployed
app never opens a 40 MB ECSV, never builds a KD-tree, and never lists a Drive
folder — which is what makes it fit in a free tier's 1 GB.

### Repo layout

The app lives in its own **private** repo (or a subtree pushed to one). Only app
code and the manifest go there — never images, ECSVs, or credentials.

```
streamlit_app.py            # entry point, st.navigation
app_pages/
  classify.py               # the flipbook
  progress.py               # per-member counts, disagreement list
lib/
  manifest.py               # load_manifest() -> DataFrame            (cached)
  pages_store.py            # PDF bytes: local dir | Drive file ID     (cached)
  labels_store.py           # append/read labels: Sheets | local CSV
  buckets.py                # bucket registry read/add
  members.py                # roster
  filters.py                # sidebar widgets -> boolean mask
manifest.parquet
requirements.txt            # pinned: streamlit==1.59.2, pymupdf, pyarrow, google-api-python-client
.streamlit/config.toml
```

Keep write access to this repo limited to Adam: on Community Cloud, **anyone with
repo write access can view and edit the app's secrets**. Collaborators are added
as *app viewers*, not repo collaborators.

---

## 2. Data sources

### Current inputs

| Path | Contents |
|---|---|
| `Data/kde_analysis_pages_9_9_2026_gaia_propagation/` | 570 PDFs, 308 MB (232 `in_both`, 338 `in_ztf`) |
| `.../in_*/dmag_order*.csv` | rank, filename, ra, dec, `{g,r,i}_dmag`, `mean_abs_dmag` |
| `Data/filter_results_7_30_2026_gaia_propagation/combined/0_enriched.ecsv` | 14,012 rows x 313 cols (`in_both`) |
| `.../combined/1_enriched.ecsv` | 426 rows x 313 cols (`in_ztf`) |

### Joining PDFs to table rows

Filenames encode position via `Source_Analysis/coord_utils.py` at 4 d.p. Primary
join is an exact `coord_stem` string match; fall back to a 1.5″ `SkyCoord`
match (= `MATCH_RADIUS_ARCSEC` in `enrich_combined_tabs.py`, itself
`Source.max_arcsec`) for any stem that doesn't hit exactly.

> **Known data gaps** — the builder must report these, not silently drop them:
> - `1_wide_association.ecsv` is 56 bytes (empty) and has no enriched counterpart.
>   No `wide_*.pdf` files exist yet, so nothing is lost today.
> - The 232 `in_both` PDFs are **stale**: that loop is commented out at
>   `Source_Analysis/store_src_plots.py:158`, so they came from an earlier filter
>   run and may not correspond to rows in the current `0_enriched.ecsv`.
>   Consider regenerating `in_both` before deploying.

---

## 3. Manifest schema

One row per PDF. Written by `scripts/build_classify_manifest.py`.

**Identity / access**
`page_id` (= `f"{catalog}/{filename}"`, the primary key), `catalog`, `filename`,
`ra`, `dec`, `drive_file_id`

**Triage ordering** (from the dmag sidecars)
`rank`, `g_dmag`, `r_dmag`, `i_dmag`, `mean_abs_dmag`

**Crossmatch columns** (from the enriched tables, five catalogs)

| Catalog | Columns carried |
|---|---|
| SDSS | `sdss_matched`, `sdss_sep_arcsec`, `sdss_class`, `sdss_subclass`, `sdss_z`, `sdss_zwarning` |
| DESI | `desi_matched`, `desi_sep_arcsec`, `desi_spectype`, `desi_subtype`, `desi_z`, `desi_zwarn` |
| TNS | `tns_matched`, `tns_designation`, `tns_type`, `tns_redshift`, `tns_discoverydate` |
| AllWISE | `wise_matched`, `wise_w1mpro`, `wise_w2mpro`, `wise_w3mpro`, `wise_var_flg`, `wise_ph_qual`, `wise_cc_flags` |
| AGN-DB | `agn_db_matched`, `agn_db_best_class_all`, `agn_db_best_class_sub_all`, `agn_db_best_z_merged`, `agn_db_star_flag` |

**Photometry**
`ZTF_{g,r,i}PSFMag`, `PSTARR_{g,r,i}PSFMag`

**Derived**
`wise_w1_w2`, `wise_w2_w3`, `spec_z` (first non-null of desi → sdss → tns →
agn_db), `spec_z_source`, `n_catalog_matches`

---

## 4. Google Sheet schema

One spreadsheet, three tabs, shared only with the service account and the team.

### `labels` — append-only

```
timestamp | member | page_id | catalog | filename | ra | dec | bucket | comment | app_version
```

Every row is a **complete snapshot** of one person's verdict on one page. Current
state is `groupby(['member','page_id']).last()`. Changing a bucket, editing a
comment, or both is the same operation, and the full revision history is free.

Append-only is the point: each action is one `values.append` call, so two people
labeling simultaneously can never clobber each other — which a read-modify-write
of a shared `labels.csv` absolutely would.

### `buckets` — the shared, extensible registry

```
bucket_id | display_name | shortcut | sort_order | created_by | created_at | active | description
```

`sort_order` is assigned **once at creation and never moves**, so a bucket
keeps its place in the list instead of drifting as counts change. (The old
`classify_pages_app.py:180` derived button order from `value_counts()`, which
reshuffled the list as you worked.)

`shortcut` is retained in the schema so existing registries stay readable, but
it is **no longer populated or read** — see the note in §7 on removing
keyboard bindings.

### `members`

```
member_id | display_name | active
```

---

## 5. UI

### Sidebar

- **Member** chosen on first visit from a row of clickable tiles, one per
  person, with a **New person** button at the end; the choice is mirrored to
  `?member=` so a refresh or bookmark keeps identity. Attribution, not
  authentication — the app itself is already invite-gated.
- **Filters** → boolean mask over the manifest (cheap at 570 rows):

  | Filter | Columns |
  |---|---|
  | Catalog | `catalog` |
  | Δmag range | `mean_abs_dmag`, `{g,r,i}_dmag` |
  | Has spectrum | `sdss_matched`, `desi_matched` |
  | Class | `sdss_class`, `desi_spectype`, `tns_type`, `agn_db_best_class_all` |
  | Known AGN | `agn_db_matched`, `agn_db_best_class_all`, `agn_db_star_flag` |
  | Known transient | `tns_matched`, `tns_type`, `tns_discoverydate` |
  | WISE colors | `wise_w1_w2`, `wise_w2_w3`, `wise_var_flg`, `wise_ph_qual` |
  | Label state | joined from the log: All / Unlabeled / Labeled by me / Unlabeled by me / Disagreements |

  All of these live in one **Filters** expander, with **Label state** kept
  outside it (the cut that changes most often) and a **reset** button sitting
  inside the expander's own header bar. There were once three blocks -- Presets, Catalog matches, Numeric cuts
  -- but that split followed how each control was implemented, not anything a
  user cares about: the presets held no state, they just wrote into the other
  two and reran. Merged, with the W1−W2 = 0.8 threshold kept as a **toggle**
  rather than a slider position, since it is a specific number you either want
  or do not and dragging onto it exactly is fiddly. A NaN colour compares
  False, so sources with no WISE match drop out while that toggle is on --
  stated in a caption under it.
- **Buckets**: one button each, stable order, name only. A popover
  adds a new one — normalize (strip, collapse whitespace), case-insensitive
  duplicate check with a "did you mean *AGN*?" nudge, append to the registry,
  usable on the next click.
- **Navigation**: `←`/`→`, *jump to next unlabeled*, progress bar over the
  *filtered* subset.

Ordering stays Δmag-descending within the filtered subset, preserving the
existing triage habit.

### Main panel

- Page header: `page_id`, position in filtered set, RA/Dec, your current bucket.
- The rasterized PDF (as today: PyMuPDF, first page, 150 dpi, `st.cache_data`).
- **Your comment**: `st.text_area` keyed by `page_id`, draft held in session
  state. Any committing action — clicking a bucket *or* navigating away — writes
  the current text along with it, so an arrow key can't discard a half-typed
  comment. You can also comment without bucketing.
- **Others' comments**: read-only list of `(member, timestamp, bucket, comment)`
  for the same page.

### Progress page

The bucket distribution per member, per-member counts, and a **disagreement
list** -- pages where two members chose different buckets. One `groupby` away,
and for a paper where classification is a judgment call, that list is arguably
the interesting output rather than a by-product.

The disagreement list is **the same table as the Table view**, narrowed to the
contested pages: its pooled bucket and comment columns already show the split
and the reasoning, so it needed no bespoke layout. The assembly and column
configuration live in `lib/pages_table.py` so the two views cannot drift apart.

There are no summary metrics at the top -- the counts they showed are all
readable from the sections below.

---

## 6. Build phases

> **Status (2026-09-14):** Phases 1-3 are built and verified locally.
> Phases 0, 4 and 5 remain. Run it with `scripts/classify_app`.

**Phase 0 — setup (manual, ~20 min).** Create the Sheet with three tabs; create a
GCP service account + JSON key; share the Sheet with the service-account email as
Editor; share the Drive analysis-pages folder with it as Viewer; create the
private GitHub repo. Exact click-path to be written out.

**Phase 1 — `scripts/build_classify_manifest.py`. [DONE]** Glob PDFs → read dmag
sidecars → load enriched ECSVs → join → select and derive columns → upload new
PDFs to Drive (reusing the helpers in `scripts/sync_followup_drive.py`, minus
`_set_anyone_reader`) → write `manifest.parquet`. Idempotent, with a
reconciliation report (PDFs without a table row, and vice versa).

**Phase 2 — app skeleton, local-only. [DONE]** Split the current single file into the
layout in §1 with `st.navigation`. Storage behind an interface so it runs with
**no Drive, no Sheets, no deploy** — local PDF dir + local CSV. Port the
existing rasterization, which already works well. `classify_pages <dir>` keeps
working offline.

**Phase 3 — buckets, identity, comments. [DONE except Sheets]** The registry, member selection, the
append-log store, comments, and the progress page.

**Phase 4 — Drive-backed pages + deploy.** `pages_store` fetches by file ID via
the service account; secrets configured; push; invite viewers; verify live.

**Phase 5 — PDF viewer component (optional, later).** Today's flat 150 dpi PNG
gives no pan or zoom, which is a real limitation for judging whether a source is
a transient or an artifact in a PanSTARRS stamp. An inline **CCv2** component
(`st.components.v2.component`, HTML+CSS+JS strings, no build step — so it works
on Community Cloud) wrapping PDF.js gives client-side pan/zoom with no server
round-trip. Self-contained swap: `pages_store` already yields PDF bytes.

> CCv2 only. `components.declare_component()`, `components.v1.html()`,
> `Streamlit.setComponentValue()`, `window.Streamlit` and
> `streamlit-component-lib` are all v1 and non-functional — and are what nearly
> every blog post and Stack Overflow answer will show.

---

## 7. What was actually built

```
scripts/build_classify_manifest.py     manifest builder (Phase 1)
scripts/classify_app                   launcher: classify_app [workspace_dir]
Source_Analysis/classify_app/
  streamlit_app.py                     entry: identity gate, then st.navigation
  app_pages/classify.py                the flipbook
  app_pages/progress.py                counts, bucket chart, disagreements
  lib/workspace.py                     workspace resolution + config.json
  lib/stores.py                        labels / buckets / members (local CSV)
  lib/identity.py                      member gate, query-param persistence
  lib/manifest.py                      manifest load + crossmatch summary chips
  lib/filters.py                        sidebar filters -> boolean mask
  lib/pages_store.py                   page_id -> PNG (the Drive seam)
  requirements.txt                     no astropy: everything is precomputed
  .streamlit/config.toml               theme
```

Built against `Data/kde_analysis_pages_9_9_2026_gaia_propagation` (570 PDFs)
and `Data/filter_results_7_30_2026_gaia_propagation/combined`. The manifest is
570 rows x 53 columns, 119 KB.

### The join came out clean

**Zero unmatched pages**, in both catalogs:

| catalog | pages | exact stem match | sky match | unmatched |
|---|---|---|---|---|
| in_both | 232 | 228 | 4 | 0 |
| in_ztf | 338 | 336 | 2 | 0 |

So the worry in §2 about the 232 `in_both` pages being stale was unfounded --
every one of them joins to a live row in the current `0_enriched.ecsv`. No need
to regenerate `in_both` before using the app.

### Filterable coverage in this sample

| | matched |
|---|---|
| SDSS | 47 / 570 |
| DESI | 45 / 570 |
| TNS | 12 / 570 |
| AllWISE | 195 / 570 |
| AGN-DB | 145 / 570 |
| any `spec_z` | 158 / 570 (agn_db 83, desi 44, sdss 23, tns 8) |
| `wise_w1_w2` | 195 / 570 |

### Verified end to end in the browser

Member gate and query-param persistence; bucket creation with case-insensitive
duplicate rejection (`Artifact` / `artifact` / `  ARTIFACT  ` all refused);
bucket assignment; comment saved in the same row as
the bucket; back-navigation restoring both; re-classification appending rather
than overwriting, with `latest_labels` resolving last-wins; every filter
including the `Unlabeled` / `Labeled by me` / `Disagreements` states and the
`W1-W2 > 0.8` preset; `Next unbucketed` skipping already-bucketed pages;
cross-member comments; and the progress page's counts, bucket chart and
disagreement list.

### Frontend revision (2026-09-14)

Phase 5 arrived early, by request. The page viewer is now an inline **CCv2**
component (`lib/components.py`) rather than `st.image`:

- An **always-visible** Expand button, and clicking the image, open a
  full-viewport modal overlay. CCv2 renders into a shadow root in the host
  document rather than an iframe, which is what lets `position: fixed` cover
  the whole screen.
- Zoom is cursor-anchored wheel zoom (1x-14x) with drag-to-pan, `+`/`-`/`0`
  keys, a Fit button, and three ways out: Close, Esc, or clicking the backdrop.
  A pan that happens to end on the backdrop does not count as that click.
- Pages now render at **200 dpi** (~4200 px wide, ~800 KB) instead of 150, for
  real detail when zoomed.
- A bordered frame around the image so it no longer blends into the page.
- The crossmatch summary chips and the photometry/full-properties expander are
  gone; that information is already printed on the analysis page itself.
- **Copy link** is a second small CCv2 component doing a one-click
  `navigator.clipboard.writeText` with inline "Copied" feedback, replacing
  Streamlit's header anchor icon (which looked like a copy button but only
  jumped to an anchor). The current page is mirrored into `?page=`, so the URL
  is always a working deep link -- verified by loading a bare `?page=` link and
  landing on that exact page.
- Prev / page-number / Next / Next unbucketed / status moved out of the sidebar
  into a bar that sticks to the bottom of the viewport.

> **Payload note:** the PNG reaches the component as a base64 data URI, ~1.1 MB
> per page, re-sent on every rerun that re-mounts it. Fine locally; worth
> revisiting for Phase 4 if page flips feel slow over the network.

### Three more bugs found and fixed in that revision

- **Esc did not close the overlay.** Streamlit binds Escape itself and was
  consuming the event before a bubble-phase listener on `document` ever saw it.
  The listener now runs on the **capture** phase and calls `stopPropagation`
  for keys it handles. (It originally let digits pass through to the bucket
  shortcuts; those bindings have since been removed.)
- **The sticky bottom bar did not stick.** `position: sticky` was set on the
  keyed container, but its immediate wrapper is exactly the bar's height, so
  there was no room to slide inside its containing block. The sticky now goes
  on that wrapper, selected with `:has(> .st-key-nav_bar)`, which can slide the
  full height of the vertical block.
- **Edits to `lib/` modules appeared not to take effect** (a DPI change and a
  component rewrite both needed a server restart). The cause was not module
  caching: `watchdog` was not installed and `runOnSave` was off, so nothing
  ever fired Streamlit's file watcher, and reloading the browser only reruns
  the script rather than re-importing changed local modules. Fixed by
  installing `watchdog` and setting `runOnSave = true` -- see "Local
  development" below. Components included, no restart needed.

### Two bugs found and fixed during testing

- **Filter reset left the sidebar lying.** Deleting a widget's key from
  `st.session_state` does not clear the browser-side state: `Label state` kept
  showing `Labeled by me` while Python had already reverted to `All` and the
  page count said 570. Widget keys are now suffixed with a generation counter
  that Reset bumps, so a reset builds genuinely new widgets.
- **The bucket chart dropped its axis labels.** Height was scaled by bucket
  count, but the bar count is buckets x members, so Vega silently omitted
  labels to fit. Height now scales with bars, and `stack=False` is explicit.

---

### Layout tuning

All CSS overrides live in one place, `lib/styles.py`, injected once from
`streamlit_app.py`. They use Streamlit's `data-testid` hooks plus the
`st-key-<key>` class derived from a container's `key`.

- **Main panel gutters** cut from ~5rem to 1.5rem with `max-width: none`, so
  the analysis page gets nearly the full window width.
- **Iterator bar** held at the bottom of the panel. It uses `position: sticky`
  rather than `fixed` deliberately: sticky inherits the panel's width, so it
  survives the sidebar being collapsed or drag-resized, and it introduces no
  new containing block -- a `transform`/`contain` on any ancestor would make
  the viewer overlay's own `position: fixed` resolve against that ancestor
  instead of the viewport, breaking the fullscreen modal.
- **Sidebar starts at the top.** Streamlit stacks a 32px logo placeholder, a
  28px collapse button and a 16px margin above the nav, putting "Classify"
  52px down. The unused logo spacer is hidden and the collapse button floated
  into the top-right corner, bringing the first nav link to 8px. Nav links get
  right padding so their text never runs under that button. Vertical gaps in
  the sidebar are tightened to 0.4rem and the 96px dead space at its foot cut
  to 1.5rem.

Two things that had to be got right, both found by measuring rather than
guessing:

- **The toolbar is a 60px opaque `absolute` header at z-index 999990.** A first
  pass set the panel's `padding-top` to 2rem, which slid the page title
  underneath it. It is now 4.5rem -- still less than the 6rem default, but
  clear of the header.
- **A permanently pinned bar hides the comment box.** A sticky element moves
  *up* from its natural position, so it floats over what precedes it. With the
  bar pinned at every scroll depth, the comment box below it could never be
  scrolled into view. `TAIL_SPACE` (4rem of panel `padding-bottom`) gives the
  bar somewhere to settle: pinned for all but the last stretch of scroll, where
  it comes to rest and the comment box is fully visible.

> **Not a code issue:** the sidebar width is user state -- Streamlit stores it
> from the drag handle on the sidebar's right edge and it survives reloads. If
> it looks too wide, drag the edge back; no CSS should pin it, or the handle
> stops working.

### Sidebar and chrome revision (2026-09-14)

- **Sidebar order** is now: Classify/Progress nav, **Catalog**, search box, then
  buckets and the remaining filters. This needed `lib/filters.py` split into
  `scope_controls` (catalog + search) and `detail_filters` (presets, label
  state, catalog matches, numeric cuts), because the sidebar's reading order
  and its computation order disagree: catalog and search render *above* the
  bucket buttons, but the buttons need the current page, which depends on every
  filter. `classify.py` reserves three containers up front and fills them in
  computation order.
- **Catalog is single-select and required**, with an **All** option:
  `All | in_both | in_ztf`. One catalog at a time is the default (`in_both`),
  the way the original `classify_pages` script worked, so the page numbers in
  the bottom bar refer to a real sequence -- `of 232`, `of 338`, or `of 570`
  for All.

  Picking All also changes the **ordering**. The manifest is stored sorted by
  catalog *then* dmag, so All would otherwise present two separate blocks and
  you would work through every `in_both` page before seeing an `in_ztf` one.
  `filters.triage_order` sorts the filtered frame by mean |dmag| descending,
  which is a no-op for a single catalog and interleaves them for All. Verified
  against the manifest: the first five rows under All are 4.87 `in_both`,
  4.01 `in_ztf`, 3.86 `in_ztf`, 3.84 `in_both`, 3.83 `in_ztf`.
- **The workspace directory left the sidebar.** It is now a `dir` badge with a
  folder icon beside the filename, carrying the full directory name as a
  tooltip, and it also appears inside the account menu.
- **The member name left the sidebar too.** `identity.account_menu` puts an
  account icon in the top-right corner; its popover shows who you are, the
  workspace, and **Log out**. Logging out clears the member from session state
  and from the URL while keeping `?page=`, so signing back in returns you to
  the same page.

The page title and the account icon share one row, vertically centred, via
`lib/chrome.page_header`. That forced the account control out of
`streamlit_app.py`: two elements can only share a row if the same code emits
them, so each page now calls `page_header` as its first output and the entry
point renders no chrome of its own.

Getting them onto a common centre line took two fixes, both found by measuring
the boxes rather than eyeballing:

- **`st.columns(vertical_alignment='center')` does not do it.** It centres each
  column's block independently, which left the title and the icon 8px apart. A
  horizontal `st.container` does: its `vertical_alignment` maps straight onto
  flexbox `align-items`. The heading takes `width='stretch'` to claim the
  leftover space and push the icon to the right edge, so no spacer is needed.
- **A heading's box is taller than its text**, because of its bottom margin, so
  it sits high inside its own box. `styles.py` collapses that margin inside
  `.st-key-page_header`.

Streamlit's toolbar (Deploy, and the options menu) is now hidden outright with
`toolbarMode = "minimal"` in `.streamlit/config.toml`. On a deployed app the
Deploy button would have gone anyway -- it is a developer option, shown only on
localhost or to the owner on Community Cloud -- but the options menu would have
remained for viewers, so this removes both. It applies locally too, so "Clear
cache" and "Rerun" are gone from development; runOnSave and a restart cover
both.

That also lifted the constraint that had kept the header row 72px down. With
the toolbar hidden, `stHeader` is an empty, transparent, `pointer-events: none`
strip that gains no content even during a rerun -- verified with a
MutationObserver across a rerun, and by hit-testing the strip, which returns
the content underneath. So the panel's `padding-top` dropped from 4.5rem to
1.5rem and the title now sits at the very top (27px on Classify, 24px on
Progress), still centre-aligned with the account icon.

The directory badge beside the filename shows the directory name itself rather
than the word "dir", with the full path as its tooltip.

### Keyboard bindings removed (2026-09-14)

Bucket buttons originally rendered as `Follow-up (1)`, where `(1)` was the
assigned digit **keybinding** spelled into the label — and Streamlit draws its
own badge for `shortcut` as well, so the key appeared twice. It read as a count
of pages filed in the bucket, and since it was a key binding it never changed:
a number apparently stuck at 1 no matter how much you labelled.

All keyboard bindings were removed at the user's request: the digit keys on the
bucket buttons and the `Left`/`Right` keys on Prev/Next. Bucket buttons show
the bucket name and nothing else, and `add_bucket` no longer assigns a
shortcut. (The arrows came back later, on their own -- see "Arrow keys for Prev
and Next". The digit keys did not: the objection was to labels carrying a
number that looked like a count, which is a bucket-button problem, not a
keyboard one.)

The zoom overlay keeps its own `Esc` / `+` / `-` / `0`, which are modal
controls rather than app-wide bindings — closing a fullscreen view with Esc is
the behaviour people expect.

The counts that do exist were never broken, and were re-verified after this
change: the bottom bar reads `3 / 232 bucketed · 229 left`, and the progress
page `Bucketed by you 3` / `Total edits 5` against a five-row log. Both derive
from `labels.csv`, whose reads are cached on `(path, mtime, size)`, so an
append invalidates them immediately.

### Arrow keys for Prev and Next

Left and right arrows step through the filtered set, exactly as the buttons do.
Streamlit has no keyboard API, so this is a third CCv2 component, `key_nav`. It
renders a hidden span, listens on the **document** in the capture phase, and
reports a press as a `setTriggerValue` -- an event that arrives as one rerun
and is gone again, rather than state that would have to be cleared.

Its key is constant across pages (`key='key_nav'`), so the listener is
installed once and survives paging instead of being torn down and rebuilt on
every rerun. The payload carries a counter alongside the direction, so a second
press the same way is a new event rather than a repeat of the value already
sent. `styles.py` collapses `.st-key-key_nav`, which would otherwise sit below
the bottom bar as an empty row with the layout's gap above it.

Four guards, all verified by dispatching events at the live page:

| Situation | Behaviour |
|---|---|
| focus in a text field, number box or `contenteditable` | ignored |
| Cmd / Ctrl / Alt / Shift held | ignored (Cmd+Arrow is browser Back) |
| the zoom overlay is open | ignored |
| focus inside a dataframe | ignored (arrows move the cell cursor) |
| already on the first or last page | ignored, with no wasted rerun |

The field check walks `event.composedPath()`, not `event.target`: the app's own
widgets sit in shadow roots, and a flattened target would make every field look
like a plain div.

The overlay guard needed a channel between two components. The overlay lives in
the viewer's shadow root where no selector from `key_nav` can reach it, so the
viewer now mirrors its open state onto `document.documentElement.dataset.
pageViewerOpen` and `key_nav` reads that. `stopPropagation` would not have
worked: both listeners are on `document`, and stopping propagation there does
not stop other listeners on the same node.

Navigating by key runs `flush_comment()` first, so an arrow can never discard
text typed into the comment box -- the behaviour the function was written for
back when the arrows existed the first time.

**Testing note:** the browser-automation `key` action delivers nothing to the
page -- a document-level capture logger recorded zero keydown events from it --
so every check here was done with dispatched `KeyboardEvent`s. That exercises
the whole path (listener, guards, trigger, rerun, Python) but not `isTrusted`,
which nothing in this path inspects.

### Class filter and the AGN-DB list columns

The class control started as a dropdown over `sdss_class` and `desi_spectype`
only. It is now a **pills** row, matching the "Require a match in" control, and
covers four catalogs: SDSS, DESI, TNS and AGN-DB.

AGN-DB was missing because its class column is **not a scalar**.
`agn_db_best_class_all` records one class per contributing sub-catalog and
stores them as a JSON list, so a raw cell reads
`["unknown", "generic_agn", "generic_agn"]`. Treated as a category that yields
19 distinct meaningless strings; unpacked, the vocabulary is four values --
`generic_agn` (134 rows), `unknown` (43), `type1` (6), `star` (2). `_as_list`
unpacks them, and `_class_sets` reduces every class column, list-valued or not,
to a set per row so one membership test serves both.

Two traps here, both caught by checking counts against the manifest:

- **`pd.NA` stringifies to `"<NA>"`.** The first cut guarded only `float` NaN,
  so unmatched rows produced a bogus `AGN-DB: <NA>` option covering 425 rows.
- **Late binding in the match loop.** The per-selection lambda has to pin
  `value=value`, or every selection matches only the last one chosen.

Matching is **any-of, at both levels**. Within a row, a selected class matches
if it appears anywhere in that object's list -- 37 of the 145 AGN-DB matches
carry more than one distinct class, and such an object is returned by a filter
on *each* of them. Across selections, the pills are OR'd.

Verified live: `AGN-DB: generic_agn` takes `in_both` from 232 to 132 and
`AGN-DB: unknown` to 43, both matching the manifest exactly, and
`in_both/130p3744_22p6017.pdf` (classed `["unknown", "generic_agn", ...]`) is
returned under both. Adding `SDSS: QSO` on top of `generic_agn` leaves the
count at 132, which is correct rather than a stuck filter -- all 39 of those
QSOs are also `generic_agn`, so the union is unchanged.

One thing the set reduction discards: repetition. A cell reading
`["generic_agn", "generic_agn", "generic_agn"]` means three sub-catalogs agreed,
and collapsing to `{generic_agn}` loses that consensus count. It does not
affect filtering, but the raw string is still in the manifest if that degree of
agreement ever matters.

The **redshift** slider and its keep-missing toggle were removed at the user's
request; `spec_z` remains in the manifest for analysis.

### Links out to the survey archives

A row at the foot of the page (`lib/external.py`) opens the source elsewhere:

| Link | Target |
|---|---|
| ZTF images | `irsa.ipac.caltech.edu/applications/ztf/?api=searchPos&ra=&dec=&execute=true` |
| PanSTARRS images | `ps1images.stsci.edu/cgi-bin/ps1cutouts` -- stack **and warp** cutouts, colour + grizy |
| TNS | a 5" cone search on the position, always -- never by designation |

Every URL shape was checked against the live service rather than written from
memory, using real positions from the manifest:

- ZTF uses the ZTF application's **documented URL API**, not the IBE endpoint.
  IBE returns exposure *metadata* -- 1335 rows of field/ccdid/filtercode with
  no picture in sight, useless for eyeballing a source. `api=searchPos` lands
  on the Results tab with an image preview centred on the target, a crosshair
  on the position, and separate tabs for science, reference and deep-reference
  products. Verified at two positions: 1851 exposures at dec +25 and 1335 at
  dec -19, the latter matching the IBE count for the same position exactly.
  The app takes 20-40 s to populate its preview.

  Two things that did **not** work, checked before settling on this:
  IRSA's Finder Chart has no ZTF survey (only 2MASS, SDSS, DSS, WISE, IRIS,
  SEIP); and Aladin Lite's standalone page ignores the `survey=` URL parameter,
  rendering an empty canvas even though the ZTF DR7 HiPS exists at CDS and does
  cover these positions -- confirmed separately through hips2fits.
- PS1 returns actual cutout markup, not a "no images" page. `filetypes`
  repeats like `filter` does, so `filetypes=stack&filetypes=warp` asks for
  both: the service echoes the form with **both boxes ticked** and the page
  renders the stacks on the top row and the single-epoch warps, each labelled
  with its MJD, beneath. At the test position that is 6 stack images and 106
  warps, all loading. The warps are what make the individual epochs visible;
  they also make the page long and slow, which the button's tooltip says.
- TNS has **no login wall**, and the query is **always positional**. Linking
  by `tns_designation` would assert an identification that came from our own
  crossmatch and that the link itself cannot show you; a cone search puts
  whatever TNS actually holds near the position in front of you, including
  nothing. Verified both ways: the search on a matched source returns
  `2020jhs` / `SN IIn`, and on an unmatched one reports "No results".

The row sits last in the flow, below the comment box, so the pinned bottom bar
rides over it until you scroll to the end -- the same behaviour the comment box
already has. Verified that at full scroll the links clear the bar and hit-test
as clickable.

### The PS1 catalog lookup, tried and dropped

There is no PS1 catalog link or panel in the app. Both attempts are recorded
here so neither gets rebuilt by accident.

**A link to the mean catalog was wrong.** The pipeline works entirely off PS1
**stack** measurements -- `{b}PSFMag`, `{b}KronMag`, their errors and
`{b}infoFlag` / `{b}infoFlag2`, out of StackObjectThin for g, r, i
(`Extracting/PS1_Local.py`, `Extracting/Catalogs.py`). No mean-object column
appears anywhere in the repo.

**No archive page serves the stack view over a URL.** Three were checked:

- `catalogs.mast.stsci.edu/panstarrs/search-results.html` is the obvious one
  and *can* query `dr2/stack`, but it takes its target from **localStorage**,
  not the query string -- its JS reads only `objID` and `MatchID` off the URL.
  Loaded with our ra/dec it ignored them, re-ran the browser's previous search
  and rendered a full table headed "RA: 227.16487 Dec: -16.62564" with no hint
  anything was wrong. A link that confidently shows the wrong position is worse
  than no link.
- `archive.stsci.edu/panstarrs/search.php` *is* URL-drivable and was the link
  for a day, but only ever returns MeanObjectView: `catalog=stack` and
  `release=dr2` are both accepted and both ignored, byte-identical responses
  either way.
- VizieR II/389 is PS1 DR2 but the mean table, and its HTML interface now sits
  behind an Anubis proof-of-work wall that stalls the first click.

**An in-app table was overkill.** That left the v0.1 JSON/CSV API -- positional,
~0.3 s, `columns=[...]` bracketed list, `sort_by=distance` -- so the rows were
fetched and rendered behind a switch next to the outbound links. It worked, and
it was removed at the user's request as more machinery than the job wanted. If
it is ever wanted back: `distance` comes back in **degrees**; an empty cone is
HTTP 200 with an empty body, not an error; unrecognised column names are
dropped **silently**, so a typo shows up as a missing column rather than a
failure.

**One finding from it worth keeping.** `st.dataframe` prints a missing number
as the literal string `None` -- with a printf `format`, with a nullable
`Float64` dtype, and with plain `NaN` alike. `step=0.001` leaves the hole blank
but truncates instead of rounding, displaying 22.3677 as 22.367. The only thing
that both rounds and leaves a hole empty is pre-formatting to text, which is
what the table view already does for its W1-W2 column:
`'' if pd.isna(v) else f'{v:.3f}'`. The cost is string sorting.

### Per-bucket colours

Each bucket carries its own colour everywhere it appears: the sidebar button,
the badge beside the page title, other members' verdicts, and the disagreement
list on the progress page.

Colour is **chosen, with a derived default**. A `⋯` button beside each bucket
opens a picker of the six palette colours, each rendered in its own colour; the
choice saves immediately. A bucket with no stored colour falls back to one
derived from `sort_order`, which is fixed at creation -- so a new bucket looks
right before anyone picks anything, and registries written before the column
existed keep working.

`constants.BUCKET_PALETTE` holds six pairs of (Streamlit colour name, CSS hex);
the name drives `st.badge`, the hex drives the sidebar. Grey is deliberately
not in the palette: it means *unbucketed*. Beyond six buckets the default
cycles, and the picker lets you break a tie by hand.

`stores.color_name_for` returns grey for a bucket it does not recognise, so a
label can safely outlive its registry row.

**The registry is append-only**, like the label log: recolouring appends a
fresh copy of the row and `read_buckets` takes the last row per `bucket_id`.
Two people recolouring at once therefore cannot clobber each other, and the
history is kept.

Three implementation notes, each found by breaking something:

- **Buttons take no colour argument**, so the sidebar is the one place the
  palette needs CSS. Each button is reached through the `st-key-` class
  Streamlit derives from its key -- hence `bucket_<id>` rather than
  `bucket::<id>`, since a `::` would need escaping in the selector.
- **Adding a column silently corrupted the store.** `buckets.csv` still had
  the pre-`color` header on disk, so appending a nine-field row to an
  eight-column file left it unparseable. `_ensure_header` now migrates a store
  once, in place and atomically, the first time a wider row needs to go in --
  filling missing columns as empty and carrying any unknown ones along.
- **`current` is the page_id.** A page module is one flat scope, so naming the
  picker's local `current` sent the viewer off to load a page called "green".
  It is `current_color` now.
- **`type='primary'` had to go.** It paints every selected button the same
  blue, overriding the very colours this is for. The bucket a page is filed
  under is instead shown with a wash of its own colour (8-digit hex for the
  alpha), a matching border, bold text and a check icon -- so selection and
  identity are carried by the same colour. Verified on two pages: with
  `Follow-up` filed the blue button is washed and ticked while `Meh` stays
  plain green-striped, and on a `Meh` page the states swap.

### Table view

A third page, `app_pages/table.py`: one row per page with **open**, file,
catalog, mean |dmag|, RA, Dec, **W1-W2**, classification, bucket and comments.
Columns sort natively by clicking a header.

`W1-W2` is shown to **3 decimals**, matching the precision AllWISE quotes its
magnitudes to -- 2 would have invented ties between genuinely different colours,
the same trap the mean |dmag| column fell into.

Only 195 of 570 pages have a WISE match, and the rest are **empty**. Getting
that blank required making the column **text** rather than a number, because a
`NumberColumn` renders a missing value as a greyed "None" no matter what:
float64/NaN, nullable `Float64`, and a Styler with `na_rep=''` all render the
same, since the column *type* decides, not the data. (`column_config`
formatting also takes precedence over a Styler's, so that route is closed from
both ends.)

> **The cost, measured rather than assumed:** a text column sorts
> lexicographically. The 185 positive colours therefore sort correctly, and
> negatives correctly sort before positives -- but the **10 negative values
> sort in reverse among themselves**. Confirmed in the app: ascending gives
> -0.042, -0.046, -0.079, -0.098, -0.126, -0.142 rather than starting at
> -0.371. Reverting to a `NumberColumn` fixes the order and brings "None" back;
> there is no third option that gives both.

- **Classification is collapsed into one column** by `classes.class_summary`,
  e.g. `SDSS QSO · DESI QSO · AGN-DB generic_agn/unknown`. 159 of 570 pages
  carry something.
- **Bucket and comments are pooled across members**: buckets as a `/`-joined
  set, comments as `member: text` joined by `|`. So the table answers "what has
  anyone said about this" rather than only showing your own view.
- **Search** covers everything displayed -- filename, catalog, classification,
  bucket, comment, rounded coordinates and W1-W2 -- not just the filename.
  (`QSO` narrows 232 in_both pages to 61; `1.191` finds the 3 pages with that
  colour; `-0.` finds the 10 negative ones.) A missing colour contributes an
  empty string rather than "nan", which would otherwise have matched 375 rows.
- **Filters are the same controls as the Classify sidebar**, rendered at the
  top of the page. They share widget keys deliberately, so a cut made in one
  view is still in force in the other.
- The class machinery moved out of `filters.py` into **`lib/classes.py`** now
  that two pages need it.

Three details worth recording:

- **`st.dataframe` is a canvas.** There are no DOM anchors, so a `LinkColumn`
  cell is drawn and handled by the grid's own JS. It works, but it cannot be
  verified by looking for an `<a>`.
- **`LinkColumn` opens a new browser tab**, which is a *new Streamlit session* --
  so the link has to carry `member` or you land on the identity gate instead of
  the page you clicked. `chrome.page_url` takes it as an optional argument for
  exactly this: present for getting around inside the app, absent for links
  meant to be handed to someone else.
- The **`open` column sits first**. Behind the wide comments column it was
  permanently off-screen.

### Label state and the W1-W2 switch

**Label state** is five-way: All, Unlabeled, Labeled by me, Unlabeled by me,
Disagreements.

"Unlabeled" and "Unlabeled by me" are genuinely different and the distinction
needed a second helper. The original "Unlabeled" was implemented against
`_bucketed_by(latest, member)` -- so it already meant *not labelled by me*, and
simply adding the new option would have produced two identical entries.
`_bucketed_by_anyone` now backs "Unlabeled", which means work **nobody** has
touched; "Unlabeled by me" includes pages a collaborator has filed and you
have not.

With one labeller the two coincide, so the divergence was proved by adding a
second member's verdict on a page the first had not filed: Unlabeled fell
562 -> 561 while Unlabeled by me stayed at 562. The test row was then removed.

**W1-W2** is a three-way switch -- `Any`, `≥ 0.8`, `< 0.8` -- rather than a
single toggle. The two cuts are complementary, so every *measured* colour falls
in exactly one: verified in the app at 128 and 67 against manifest counts of
128 and 67, summing to the 195 pages that have a colour. The other 375 have no
WISE match and a NaN compares False either way, so they are excluded by **both**
cuts rather than shuffled into the `<` side -- which the caption says.

> **An empty result set used to crash the Table page.** Selecting Disagreements
> with none on record gave `ValueError: Per-column arrays must each be
> 1-dimensional`: `DataFrame.apply(axis=1)` on an empty frame returns a
> *DataFrame*, not a Series, so `class_summary(...).to_numpy()` was 2-D.
> `class_summary` now returns early for an empty frame, and the table shows a
> "no pages match" notice instead of an empty grid.

### "reset" inside the Filters header

`st.expander` takes a **string** label, so a widget cannot be a child of its
header. Both therefore live in one keyed container, `filters_row`, and
`styles.py` lifts the button onto the header bar with `position: absolute`.
Taking it out of the flow also collapses the row back to the expander's own
height, so nothing is pushed down. The expander's chevron is on the left, which
leaves the right side free.

Two details that had to be right:

- **The reset button's key is deliberately *not* generation-scoped.** Every
  other filter widget's key carries the generation counter that reset bumps,
  but the CSS selector has to keep matching afterwards -- and a button holds no
  state to reset anyway.
- **Default button height is 40px against a 38px header**, so it hung below the
  bar. Trimmed to 30px it sits centred inside it: measured 581-611 within
  578-616 in the sidebar, 265-295 within 262-300 on the Table page.

The expander is created empty, the button next, and the expander filled only
afterwards -- so a click is handled before any filter widget is instantiated
and the rerun costs nothing. Verified working: 232 -> 224 under "Unlabeled by
me", back to 232 on reset.

### The sign-in picker

One clickable tile per active member, then **New person**. A tile signs you in
on a single click -- there is no second "start" step -- and the roster is read
from `members.csv`, so it grows as people add themselves.

`stores.add_member` derives the `member_id` by slugging the display name and
refuses a collision on *either* the name or the derived id, case-insensitively
and whitespace-normalised. That matters more than it looks: two people sharing
an id would silently share their verdicts, since every label row is keyed by
`member`. Blank and punctuation-only names are refused too. Verified: `Sam
Jones` -> `sam_jones`, then `sam jones` and `Sam  Jones` both rejected.

`read_members` now takes the **last row per member_id**, matching the buckets
registry and the label log, so the roster stays append-only and editable
without rewrites.

New people are added through a form inside a popover, so Enter submits and the
field clears. On success the app signs the new person straight in.

The tiles are centred and oversized (2026-09-14): title and caption use
`text_alignment='center'`, the row is a horizontal container keyed
`who_tiles` with `horizontal_alignment='center'`, and CSS bumps the tile
padding plus the leading Material icon from 16px to 36px. The icon is reached
with `button span:has(+ [data-testid="stMarkdownContainer"])` -- the icon is the
bare span that precedes the label's markdown container, so there is no class to
hook.

### The blank white band above a collapsed sidebar

Collapsing the sidebar dropped an empty 60px white band across the top of the
page, hiding the page title and the account icon behind it.

`stHeader` behaves differently in the two states, which is what made this easy
to miss. With the sidebar **open** it is already invisible -- `background:
rgba(0,0,0,0)`, `pointer-events: none`, and empty, since the toolbar is hidden
via `toolbarMode = "minimal"`. Collapsing the sidebar flips all three at once:
Streamlit moves the expand arrow into the strip, and the strip turns opaque
white and starts taking clicks. Measured in both states:

| | sidebar open | sidebar collapsed |
|---|---|---|
| height | 60px | 60px |
| background | `rgba(0,0,0,0)` | `rgb(255,255,255)` |
| pointer-events | `none` | `auto` |
| `stExpandSidebarButton` | absent | present |

The fix is to keep it see-through in both states. Nothing below it moves,
because the strip is absolutely positioned and never occupied layout space --
it was only ever painting over the content. `pointer-events` has to be handed
back to the arrow itself, or the sidebar can no longer be reopened (verified
with a real mouse click, not a scripted one, since a scripted `.click()` would
have passed either way).

That leaves the arrow floating over the top-left corner, exactly where the page
title starts, so the title row takes a 2.5rem left indent while the arrow is on
screen -- keyed off the arrow's own existence, `body:has([data-testid=
"stExpandSidebarButton"])`, so the indent appears and disappears with it. The
sign-in gate renders no sidebar at all, so nothing there changes.

A side benefit: the opaque strip sat at `z-index: 999990` and took clicks, so
it was covering the top of the page viewer's zoom overlay whenever the sidebar
was collapsed. That is gone too.

### `st.html` silently drops a whole stylesheet

Worth knowing before touching `lib/styles.py`: **no angle brackets anywhere in
that stylesheet, not even inside a CSS comment.** A literal tag name written
in a comment (mine said "span", in brackets) made `st.html`'s sanitiser discard
the *entire* `style` block rather than the offending comment. There is no error
and no console warning -- the CSS is simply absent from the DOM, so every
override reverts at once and the page looks like a fresh Streamlit app.

The tell: `stMainBlockContainer` padding reads back as Streamlit's defaults
(80px top / 96px sides) instead of our 1.5rem. Cheapest check, in the browser
console:

    getComputedStyle(document.querySelector('[data-testid="stMainBlockContainer"]')).paddingTop

A warning comment now sits at the top of `lib/styles.py`.

### The "duplicate" mean |dmag| values -- and what they exposed

Reported as duplicate values in the table. They are **not** duplicates:

- All **570** `mean_abs_dmag` values are distinct, as are all per-band values
  (422 g, 544 r, 488 i non-null, all distinct).
- Every one matches its sidecar entry looked up by `(catalog, filename)` --
  **0 mismatches across 570**, so the manifest join is correctly aligned.
- `mean_abs_dmag` equals the nanmean of the per-band `|dmag|` for all 570 rows,
  to within 1e-6.

The display was at fault: `format='%.2f'` collapsed runs like 1.446435 ...
1.450279 onto "1.45". The column now shows **4 decimals**.

**The clustering underneath is real, though, and it says something about the
`in_both` pages.** Counting 0.01-wide bins across the 570: 56 rows at 0.36,
53 at 0.37, 30 at 1.45. Those are not scattered coincidences -- the 0.36 group
is sidecar ranks **12,924 to 12,979, a perfectly contiguous run of 56**, and
**143 of the 232** `in_both` pages fall in ranks 12,000-13,000 of a 14,012-row
sidecar.

So the `in_both` PDFs are not a dmag-ranked top-N or any coherent cut: they are
mostly one narrow slice from deep down the ranking, plus scattered others. That
is the fingerprint of a partial or interrupted generation run, which fits the
`in_both` loop being commented out at `store_src_plots.py:158` -- these 232
files are leftovers from an earlier pass.

`in_ztf` is healthy by contrast: 338 of 426 sidecar rows (79%), spread evenly
across the whole rank range, with no bin holding more than 15.

> **Consequence for the science, not the app:** any statistic computed over the
> `in_both` pages is a statistic about that leftover slice, not about the
> filter output. Regenerating `in_both` would fix it. The app itself is
> reporting the data faithfully.

### A real bug in the page/sidecar naming

Prompted by the mean |dmag| question. The dmag arithmetic itself is sound:
`store_src_plots.mean_abs_dmag` and the per-band computation inside
`write_dmag_sidecar` are two separate implementations of the same flag/limit
substitution, and they agree to 1e-6 on all 570 pages. Magnitudes are not being
read back as `0.0` either -- per-band dmag spans -7.84 to +6.70 with nothing in
the 13-24 range a zero-magnitude artefact would produce.

The bug is in the **filenames**. `coord_utils.coord_stem` rounds to 4 decimal
places -- a grid of about 0.36" -- so two sources closer than that get the same
stem. In the current page directory:

| | |
|---|---|
| Colliding filenames | **128** (124 pairs + 4 triples) covering 260 sidecar rows |
| Separations | min 0.000", median 0.093", max 0.312" -- **all inside the 0.36" grid** |
| Genuinely distinct nearby sources | 118 pairs (0.01-0.36" apart) |
| Same position, different photometry | 6 pairs -- duplicate catalog rows, a separate upstream issue |
| Their `mean_abs_dmag` disagreement | median 0.040, max 0.851; 17 of 124 pairs differ by >0.1 |
| Pages in the current sample affected | **6 of 570** (4 `in_both`, 2 `in_ztf`) |

Three consequences, in increasing order of seriousness:

1. `save_src_plot` writes both sources to the same path, so **one analysis page
   silently overwrites the other** and only one of the two is ever reviewable.
2. The sidecar carries two rows under one filename with different dmag.
3. Anything joining on filename has to pick one. The manifest builder keeps the
   last, and that choice is independent of which PDF survived -- so for those 6
   pages **the dmag shown may belong to the source whose page was overwritten**.

`load_dmag_sidecars` now logs the collision count and the dmag spread instead
of calling `drop_duplicates` silently.

> **Fixing it properly is upstream**, in `coord_stem`: more decimal places (5
> d.p. is ~0.036") or a disambiguating suffix when a stem is already taken.
> That changes every page filename, so it wants doing at the same time as the
> `in_both` regeneration rather than on its own.

### "Matched but not classified"

Being in a catalog and being *typed* by it are different questions, and TNS is
where they come apart: 12 of the 570 pages have a TNS match and **4 of those
carry no type**. Those four used to drop out of the classification column
entirely, reading as though TNS had never heard of the source.

`class_sets` now fills `UNCLASSIFIED` ("unknown") for any row a catalog matched
but did not classify, driven by the `<prefix>_matched` flags in
`MATCHED_COLUMN`. The fill goes into the **class set**, not the display string,
so the summary column and the Class filter pills stay in step for free: the
column reads `TNS unknown`, a `TNS: unknown` pill appears, and selecting it
returns exactly those 4 pages -- verified in the app.

It is written generally but only TNS triggers it today: SDSS (47), DESI (45) and
AGN-DB (145) each classify every source they match. One page shows why the fill
belongs in the set rather than replacing the whole summary --
`SDSS QSO · TNS unknown · AGN-DB generic_agn/type1`, where TNS has the source
but has not typed it while two other catalogs have.

### Local development

`.streamlit/config.toml` sets `runOnSave = true`, and `watchdog` is installed,
so saving any file under `Source_Analysis/classify_app/` reruns the script and
pushes the change to every open browser tab -- no restart, no manual refresh,
and session state (who you are, which page you are on) survives. Verified for
page files, `lib/` modules, and the CCv2 components alike.

`folderWatchBlacklist` excludes `Data/`, so the watcher ignores the hundreds of
analysis-page PDFs that never change.

Two things still need a restart, both by nature rather than by bug: changing
`.streamlit/config.toml` itself, and changing the `--workspace` argument.

## 8. Open questions / risks

- **Regenerate `in_both`.** The 232 pages join cleanly, but they are not a
  coherent sample: 143 of them come from one contiguous slice of the dmag
  ranking (see the mean |dmag| investigation below). Fine for exercising the
  app, wrong for drawing conclusions from.
- **`Data/` is gitignored**, so the workspace -- `manifest.parquet` included --
  is local-only today. Phase 4 has to copy the manifest into the app repo (or
  build it there), rather than relying on the path in `config.json`.
- ~~**Keyboard shortcuts don't fire while a text box has focus.**~~ Moot:
  there are no keyboard bindings left outside the zoom overlay.
- **Bucket renames and merges** aren't in the MVP. Renaming is a `buckets` tab
  edit; merging needs a small migration over the `labels` log. Add only if needed.
- **Community Cloud apps sleep** after ~12 h idle; the next load takes ~30 s.
  Harmless, but people will notice and should be told.
- **The single private-app slot** is consumed by this app. A second private app
  means making this one public or deleting it.
- **Sheets API quotas** are 300 writes/min/project — far above one click per few
  seconds per person, but worth knowing if the team ever grows a lot.
