/** Mirrors the FastAPI response models in `api/models.py`.
 *
 * Kept hand-written rather than generated: the surface is small, and the
 * shapes the charts care about (the light-curve and spectrum payloads) are
 * served as raw gzipped JSON and never appear in the OpenAPI schema at all.
 */

export interface SourceSummary {
  source_uuid: string
  coord_slug: string | null
  ra: number | null
  dec: number | null
  n_epochs: number
  surveys: string[]
  n_spectra: number
  spec_surveys: string[]
  /** This author's label, when the request named one. */
  label: string | null
  n_annotations: number
  /** Sub-catalogue in the `catalog=` filter's terms; null outside both. */
  catalog: 'in_both' | 'in_ztf' | null
}

export interface SourcePage {
  total: number
  offset: number
  limit: number
  /** How many of `total` the requesting author has filed. */
  n_labelled: number
  items: SourceSummary[]
}

/** One entry in the shared label vocabulary (`labels` in annotations.sqlite). */
export interface Label {
  name: string
  created_by: string | null
  created_at: string
  /** '#rrggbb', or null for the default derived from the name. */
  color: string | null
  /** Sources anyone has filed under it. */
  n_sources: number
}

/** Follow-up observing state, in order. null is the fourth ("nothing"). */
export const FOLLOWUP_STATUSES = ['to queue', 'queued', 'observed'] as const
export type FollowupStatus = typeof FOLLOWUP_STATUSES[number]

export interface Annotation {
  source_uuid: string
  author: string
  label: string | null
  confidence: string | null
  note: string | null
  tags: string[]
  status: FollowupStatus | null
  created_at: string | null
  updated_at: string | null
}

export interface SourceDetail {
  source_uuid: string
  coord_slug: string | null
  ra: number | null
  dec: number | null
  n_epochs: number
  surveys: string[]
  n_spectra: number
  spec_surveys: string[]
  /** The ~300 catalog columns not promoted onto the source itself. */
  catalog: Record<string, unknown>
  /** Cross-match classifications: display prefix -> classes. Derived by the
   *  API from `catalog`, so the table and this page cannot disagree. */
  classes: Record<string, string[]>
  /** Same keys as `classes`, for the catalogs reporting a redshift. A catalog
   *  can be in `classes` and absent here (SIMBAD stars, most TNS rows), so a
   *  missing entry means "no redshift", not "no match". */
  redshifts: Record<string, number>
  annotations: Annotation[]
}



export interface CatalogGroup {
  key: string
  label: string
  n_columns: number
  columns: string[]
}

export interface CatalogSchema {
  groups: CatalogGroup[]
  n_columns: number
}

// ── the chart payloads ────────────────────────────────────────────────────────
//
// These are served as pre-gzipped JSON straight from the database, so they are
// not in the OpenAPI schema. Columnar per (survey, band): arrays are parallel
// and `mjd` is sorted ascending.

export interface BandSeries {
  mjd: number[]
  mag: number[]
  magerr: number[]
  snr?: number[]
  /** ZTF only. The `catflags & 6141` cut is the consumer's job. */
  catflags?: number[]
  /** ZTF only: the DR objectid, which encodes the field/ccd/quadrant. */
  catalog_id?: (string | null)[]
  clrcoeff?: number[]
  /** PanSTARRS only. */
  infoFlag?: number[]
  infoFlag2?: number[]
  /** SDSS only. */
  psf_mag?: number[]
  psf_magerr?: number[]
  /** custom and ztforce. */
  is_upper_limit?: boolean[]
  telescope?: (string | null)[]
  instrument?: (string | null)[]
}

export interface LightCurve {
  source_uuid: string
  n_epochs: number
  /** survey -> band -> series */
  surveys: Record<string, Record<string, BandSeries>>
}

export interface Spectrum {
  survey: string
  spec_id: string
  spec_class: string | null
  redshift: number | null
  n_pix: number
  flux: (number | null)[]
  model: (number | null)[] | null
  /** Present only when the grid was irregular; otherwise rebuild from
   *  `wave_start` + i * `wave_step`, applying 10** when scale is 'log10'. */
  wavelength?: (number | null)[]
  wave_scale?: 'linear' | 'log10' | null
  wave_start?: number
  wave_step?: number
}

export interface Spectra {
  source_uuid: string
  n_spectra: number
  spectra: Spectrum[]
}

export interface SourceQuery {
  offset?: number
  limit?: number
  /** A catalog column, or one of the short aliases 'uuid' | 'ra' | 'dec' |
   *  'epochs'. Not a union of the four any more: the server orders by any
   *  column it has, including derived ones like `mean_abs_dmag`, and
   *  validates the name against the catalog schema. */
  sort?: string
  desc?: boolean
  /** Sub-catalogue: 'in_both' or 'in_ztf'; omit for all. */
  catalog?: string
  author?: string
  /** Substring of `coord_slug`, or a uuid prefix. Matched server-side: the
   *  client holds only a window of the list, so filtering what it has would
   *  silently miss every match outside it. */
  q?: string
  /** A filter document, JSON-encoded. Sent in the URL rather than a body so
   *  that a filtered view stays a link you can paste to someone. */
  filter?: string
  /** Verdict cuts, resolved server-side against `annotations.sqlite`. These
   *  were sent but never declared: an object literal assigned through a
   *  generic type argument is not excess-property-checked, so the one object
   *  both pages serialise into URLs had no type cover at all. */
  bucket_in?: string[]
  bucket_out?: string[]
  status_in?: string[]
  status_out?: string[]
}

/** Ready-to-use cutout URLs, survey -> band -> URL.
 *
 * The PanSTARRS stack filenames are resolved at build time, so these arrive
 * with the page rather than costing a ~0.74 s lookup per source at view
 * time. The browser fetches the JPEGs straight from STScI.
 */
/** The exposure behind one light-curve point (`/epoch/{survey}/{band}`). */
export interface EpochImage {
  survey: string
  band: string
  mjd: number
  image_id: string
  /** Exposure start, UTC. */
  obs_mjd: number
  details: Record<string, number | null>
  /** Rendered cutout, on the same grid as the reference stamp. */
  png: string
  pixel_arcsec: number
  /** FITS crop around the source. */
  fits_url: string
  /** ZTF: the whole science image; PS1: a JPEG view at STScI. */
  full_url: string
}

export interface Cutouts {
  source_uuid: string
  images: Record<string, Record<string, string>>
  /** survey -> arcsec per pixel, for sizing the scale-bar overlay. */
  pixel_arcsec: Record<string, number>
}

// ── table page ────────────────────────────────────────────────────────────────
//
// `/api/sources` carries none of the catalog columns the table shows, and the
// detail endpoint carries all ~300 of them one source at a time. `/api/table`
// sits between the two: every source reduced to the fields the table displays
// plus the crossmatch flags its filters cut on, so filtering, sorting and
// search run locally the way the Streamlit page runs them as pandas masks.

export interface TableAnnotation {
  author: string
  /** The bucket. Null or empty means filed by nobody. */
  label: string | null
  note: string | null
  status: FollowupStatus | null
}

export interface TableRow {
  source_uuid: string
  coord_slug: string | null
  ra: number | null
  dec: number | null
  catalog_tier: string | null
  /** Mean |ZTF - PanSTARRS| over g/r/i, recomputed by the API. */
  mean_abs_dmag: number | null
  /** Mean |Δmag σ| over g/r/i against each field's envelope (capped at 8). */
  mean_abs_dmag_sigma: number | null
  wise_w1w2: number | null
  /** Crossmatch prefix (`sdss`, `wise`, ...) -> did that catalog match. */
  matched: Record<string, boolean>
  /** Display prefix (`SDSS`, `AGN-DB`, ...) -> classes, sorted and unique. */
  classes: Record<string, string[]>
  /** Every author's verdict, pooled rather than filtered to one member. */
  annotations: TableAnnotation[]
  /** Whether ztforce has a stored result for this source. */
  ztforce: boolean
}

export interface TablePayload {
  total: number
  offset: number
  limit: number
  /** Rows arrive without verdicts; `getTableAnnotations` carries those. */
  rows: Omit<TableRow, 'annotations' | 'ztforce'>[]
}

/** Which window of the catalog the table should show.
 *
 *  `filter` is the same document the classify page builds, so narrowing the
 *  table to something worth reading is the same gesture in both places. */
export interface TableQuery {
  offset?: number
  limit?: number
  sort?: string
  desc?: boolean
  catalog?: string
  q?: string
  filter?: string
  /** Cut server-side so the window is drawn from the matching sources, not
   *  filtered after the fact. 'Labeled by me' / 'Unlabeled by me' need
   *  `author`. */
  label_state?: string
  author?: string
  /** Buckets anyone filed the source under; several are an OR. */
  bucket_in?: string[]
  /** Buckets that exclude the source if anyone filed it under them. */
  bucket_out?: string[]
  /** Follow-up state someone set; 'none' is nobody set one. */
  status_in?: string[]
  status_out?: string[]
}

export interface TableAnnotations {
  annotations: Record<string, { author: string; label: string | null;
                                note: string | null
                                status: FollowupStatus | null }[]>
  /** Sources with a stored ztforce result. */
  ztforce_done: string[]
}

// ── progress page ─────────────────────────────────────────────────────────────
//
// One response carrying the label x author cross-tab, the per-author table and
// the contested list. `/api/stats` has the first two only as separate
// one-dimensional counts, which cannot be crossed back together.

export interface MemberProgress {
  author: string
  /** Distinct sources this author has filed. */
  n_sources: number
  n_labels: number
  n_comments: number
}

export interface Verdict {
  author: string
  label: string | null
  note: string | null
}

export interface Disagreement {
  source_uuid: string
  coord_slug: string | null
  ra: number | null
  dec: number | null
  /** The distinct labels the source was filed under, sorted. */
  labels: string[]
  verdicts: Verdict[]
}

export interface Progress {
  n_sources: number
  n_labelled: number
  authors: string[]
  /** label -> author -> count. Only labelled annotations are counted. */
  by_label: Record<string, Record<string, number>>
  by_member: MemberProgress[]
  /** Total contested sources, which can exceed `disagreements.length`. */
  n_disagreements: number
  disagreements: Disagreement[]
}

export type ZtforceStatus = 'none' | 'queued' | 'running' | 'done' | 'error'

/** A source's ztforce run, from the background queue. `stacks` holds the stored
 *  result, if any -- including an older one while a re-run is pending, flagged
 *  `stale`. */
export interface ZtforceResult {
  source_uuid: string
  status: ZtforceStatus
  /** How it was queued: the "Run ztforce" label, or "run now" on the page. */
  origin: 'label' | 'manual' | null
  requested_at: string | null
  started_at: string | null
  finished_at: string | null
  /** Stacking windows offered, in days; 0 is unbinned. */
  windows: number[]
  /** String(window) -> band -> series, stacked by ztforce. Stacks below S/N 3
   *  (and, unbinned, single-epoch non-detections) are `is_upper_limit`. */
  stacks: Record<string, Record<string, BandSeries>> | null
  stale: boolean
  /** Good (unflagged) epochs across bands. */
  n_epochs: number | null
  /** e.g. epochs IRSA could not serve. */
  note: string | null
  error: string | null
  /** 0-based place in the queue, while queued. */
  queue_position: number | null
  eta_s: number | null
  /** Seconds since the run started, while running. */
  elapsed_s: number | null
}

export interface ZtforceQueue {
  /** The annotation label that queues a source. */
  label: string
  paused: boolean
  concurrency: number
  n_queued: number
  n_running: number
  n_done: number
  n_error: number
  running: { source_uuid: string; elapsed_s: number | null }[]
  /** Rough seconds to drain the queue. */
  eta_s: number | null
}


// ── filters ───────────────────────────────────────────────────────────────────
//
// A filter is a document, not a string of SQL: it survives a URL, it can be
// saved and named, and the server can refuse it before it runs. Groups nest
// arbitrarily; the builder UI only produces one level, but a hand-written or
// saved filter may be deeper and will still round-trip.

export type FilterOp =
  | '=' | '!=' | '<' | '<=' | '>' | '>='
  | 'between' | 'in' | 'not_in'
  | 'contains' | 'startswith' | 'endswith' | 'has_element'
  | 'is_null' | 'is_not_null' | 'is_true' | 'is_false'

/** What to do with sources where the column is missing.
 *
 *  Not a detail: SIMBAD matches 26% of the catalog, so `otype != 'Star'`
 *  means two very different things depending on this, and the difference is
 *  145,000 sources. */
export type MissingPolicy = 'exclude' | 'include'

export interface FilterLeaf {
  column: string
  op: FilterOp
  value?: unknown
  missing?: MissingPolicy
}

export interface FilterExpr {
  expr: string
  missing?: MissingPolicy
}

export interface FilterGroup {
  op: 'and' | 'or' | 'not'
  clauses: FilterNode[]
}

export type FilterNode = FilterLeaf | FilterExpr | FilterGroup

export const isGroup = (n: FilterNode): n is FilterGroup =>
  typeof n === 'object' && n !== null && 'clauses' in n
export const isExpr = (n: FilterNode): n is FilterExpr =>
  typeof n === 'object' && n !== null && 'expr' in n

export interface FilterColumn {
  name: string
  type: 'number' | 'text' | 'bool'
  n_non_null: number
  /** 0..1. Shown in the picker because a cut on a 0.6%-filled column is a
   *  cut on 0.6% of the catalog, and nothing else on screen says so. */
  fill: number
  filterable: boolean
}

export interface FilterSchema {
  operators: FilterOp[]
  missing_policies: MissingPolicy[]
  groups: { key: string; label: string; columns: FilterColumn[] }[]
  n_columns: number
}

export interface ColumnStats {
  column: string
  type: 'number' | 'text' | 'bool'
  n_non_null: number
  fill: number
  min?: number
  max?: number
  quantiles?: Record<string, number>
  n_distinct?: number
  values?: { value: string | number | boolean; count: number }[]
}

export interface FilterPreview {
  total: number
  /** How many the rest of the sidebar leaves, before this filter. */
  scope_total: number
  clauses: { clause: FilterNode; matches?: number; error?: string }[]
  sql: string
}
