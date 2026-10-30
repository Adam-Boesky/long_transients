/** What the Table page shows and how it is cut down.
 *
 * Ports `classify_app/lib/pages_table.py` (row assembly, searchable text) and
 * the mask half of `classify_app/lib/filters.py`. Kept out of the page for
 * the same reason the Streamlit app keeps it out of `app_pages/table.py`:
 * the Progress view narrows the identical table to the disagreed sources, so
 * the assembly should exist once.
 */
import { FOLLOWUP_STATUSES } from '../api/types'
import type { FollowupStatus, TableRow } from '../api/types'

export const CLASS_PREFIXES = ['SDSS', 'DESI', 'TNS', 'SIMBAD', 'AGN-DB']

export const LABEL_STATES = ['All', 'Unlabeled', 'Labeled', 'Labeled by me',
                             'Unlabeled by me', 'Disagreements'] as const
export type LabelState = (typeof LABEL_STATES)[number]

export interface DisplayRow {
  uuid: string
  file: string
  catalog: string
  sigma: number | null
  dmag: number | null
  ra: number | null
  dec: number | null
  w1w2: number | null
  classification: string
  bucket: string
  /** Follow-up state, pooled across members; '' when nobody has set one. */
  status: string
  /** 'yes' when ztforce has a stored result, else '' so it sorts last. */
  ztforce: string
  comments: string
  /** The applied coordinate list's value columns, by name. */
  listValues: Record<string, number | null>
  /** Everything above, lowercased, for the free-text filter. */
  search: string
}

/** A column key, or `list:<column>` for a value column of the applied
 *  coordinate list, which the server orders by under the same name. */
export type SortKey = 'file' | 'catalog' | 'sigma' | 'dmag' | 'ra' | 'dec' | 'w1w2'
  | 'classification' | 'bucket' | 'status' | 'ztforce' | 'comments' | `list:${string}`

export const LIST_SORT_PREFIX = 'list:'

export function listColumnOf(key: string): string | null {
  return key.startsWith(LIST_SORT_PREFIX) ? key.slice(LIST_SORT_PREFIX.length) : null
}

/** Which catalog column a sort key is, for the columns the server can order.
 *
 * Sorting in the browser can only rank the window that was fetched, which for
 * a *ranking* is the wrong answer rather than a slow one: the largest
 * |dmag| among the first 5,000 sources by uuid is 7.73, and in the catalog it
 * is 8.95. These five are real columns (`mean_abs_dmag` is derived in the
 * query engine), so the server ranks all 197,933 and sends back the page you
 * asked for.
 *
 * The three that are missing -- classification, bucket, comments -- are built
 * from the annotations database and from Python-side string assembly, so they
 * have no column to order by and still sort within the window. The caption
 * says which you are getting rather than letting the two look alike.
 */
export const SERVER_SORT: Partial<Record<SortKey, string>> = {
  file: 'coord_slug',
  catalog: 'catalog_tier',
  sigma: 'mean_abs_dmag_sigma',
  dmag: 'mean_abs_dmag',
  ra: 'ra',
  dec: 'dec',
  // `wise_w1w2_mean`, matching what the column shows: sorting by the
  // AllWISE coadd while displaying the binned light-curve colour put the
  // rows in an order the numbers beside them did not explain.
  w1w2: 'wise_w1w2_mean',
}

/** The server's name for a sort key, or undefined for the columns it cannot
 *  order by. A list value column is sent as it is. */
export function serverSortKey(key: SortKey): string | undefined {
  return listColumnOf(key) !== null ? key : SERVER_SORT[key]
}

const filed = (a: { label: string | null }) => (a.label ?? '').trim()

/** One readable classification string, across all four catalogs.
 *
 * e.g. `SDSS QSO · DESI QSO · AGN-DB generic_agn/unknown`. Empty when nothing
 * classified the source.
 */
function classification(row: TableRow): string {
  return CLASS_PREFIXES
    .map((p) => {
      const values = row.classes[p] ?? []
      return values.length ? `${p} ${values.join('/')}` : ''
    })
    .filter(Boolean)
    .join(' · ')
}

export function buildRow(row: TableRow): DisplayRow {
  // Verdicts pooled across members: the table answers "what has anyone said
  // about this source", which is the useful question when several people are
  // working through the same list.
  const buckets = [...new Set(row.annotations.map(filed).filter(Boolean))].sort()
  const comments = row.annotations
    .filter((a) => (a.note ?? '').trim())
    .map((a) => `${a.author}: ${a.note}`)
    .join('  |  ')
  // Ordered by FOLLOWUP_STATUSES, not alphabetically, so a sort reads as
  // progress rather than 'observed' before 'queued'.
  const statuses = [...new Set(row.annotations
      .map((a) => a.status).filter((s): s is FollowupStatus => !!s))]
    .sort((a, b) => FOLLOWUP_STATUSES.indexOf(a) - FOLLOWUP_STATUSES.indexOf(b))

  const out: DisplayRow = {
    uuid: row.source_uuid,
    file: row.coord_slug ?? row.source_uuid,
    catalog: row.catalog_tier ?? '',
    sigma: row.mean_abs_dmag_sigma,
    dmag: row.mean_abs_dmag,
    ra: row.ra,
    dec: row.dec,
    w1w2: row.wise_w1w2,
    classification: classification(row),
    bucket: buckets.join(' / '),
    status: statuses.join(' / '),
    ztforce: row.ztforce ? 'yes' : '',
    comments,
    listValues: row.list_values ?? {},
    search: '',
  }
  out.search = [
    out.file, out.catalog, out.classification, out.bucket, out.status,
    out.comments,
    out.ra?.toFixed(4) ?? '', out.dec?.toFixed(4) ?? '', w1w2Text(out.w1w2),
  ].join(' ').toLowerCase()
  return out
}

/** W1-W2 as text. 3 decimals matches the precision AllWISE quotes its
 *  magnitudes to; coarser invents ties between genuinely different colours.
 *  Empty, not a placeholder number, where there is no WISE match. */
export function w1w2Text(value: number | null): string {
  return value === null ? '' : value.toFixed(3)
}

/** Biggest mean |Δmag σ| first, sources with none last.
 *
 * With one catalog selected this is already the stored order. It matters for
 * "All": without it the tiers stay as separate blocks and you would work
 * through every tier-1 source before seeing an in_both one, rather than
 * seeing the largest changes across both.
 */
export function triageOrder(rows: DisplayRow[]): DisplayRow[] {
  return sortRows(rows, 'sigma', true)
}

export function sortRows(rows: DisplayRow[], key: SortKey,
                         desc: boolean): DisplayRow[] {
  const column = listColumnOf(key)
  const value = (r: DisplayRow) => (column !== null ? r.listValues[column] ?? null : r[key as Exclude<SortKey, `list:${string}`>])
  return [...rows].sort((x, y) => {
    const a = value(x), b = value(y)
    // Missing values sort last whichever way the column is pointing, so a
    // blank cell never takes the top of the table.
    const aEmpty = a === null || a === ''
    const bEmpty = b === null || b === ''
    if (aEmpty || bEmpty) return aEmpty && bEmpty ? 0 : aEmpty ? 1 : -1
    const d = typeof a === 'number' && typeof b === 'number'
      ? a - b : String(a).localeCompare(String(b))
    return desc ? -d : d
  })
}
