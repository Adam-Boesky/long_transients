/** What the Table page shows and how it is cut down.
 *
 * Ports `classify_app/lib/pages_table.py` (row assembly, searchable text) and
 * the mask half of `classify_app/lib/filters.py`. Kept out of the page for
 * the same reason the Streamlit app keeps it out of `app_pages/table.py`:
 * the Progress view narrows the identical table to the disagreed sources, so
 * the assembly should exist once.
 */
import type { TableRow } from '../api/types'

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
  /** Everyone's notes as one line, for the search box and the tooltip. */
  comments: string
  /** The same notes, one entry each, for the cell to shorten. */
  commentList: { author: string; note: string }[]
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

/** The server's name for each sort key.
 *
 * Every column is ordered by the server over the whole sample. Sorting in
 * the browser could only rank the page that was fetched, which for a
 * *ranking* is the wrong answer rather than a slow one: the largest |dmag|
 * among the first 5,000 sources by uuid is 7.73, and in the catalog it is
 * 8.95. The catalog columns are real (or derived) columns of the query
 * engine; the `ann:` ones are rendered from the annotations database per
 * request and handed to the engine as values to order by (see
 * `api.annotation_sorts`), so a bucket sort ranks all 197,933 too.
 */
export const SERVER_SORT: Record<Exclude<SortKey, `list:${string}`>, string> = {
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
  classification: 'classification',
  // By the cell's text; the status in progress order (to queue < queued <
  // observed); comments by how many; ztforce by whether it has run.
  bucket: 'ann:bucket',
  status: 'ann:status',
  comments: 'ann:comments',
  ztforce: 'ann:ztforce',
}

/** The server's name for a sort key. A list value column is sent as it is. */
export function serverSortKey(key: SortKey): string {
  return listColumnOf(key) !== null ? key : SERVER_SORT[key as keyof typeof SERVER_SORT]
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
  const commentList = row.annotations
    .filter((a) => (a.note ?? '').trim())
    .map((a) => ({ author: a.author, note: (a.note ?? '').trim() }))
  const comments = commentList.map((c) => `${c.author}: ${c.note}`).join('  |  ')

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
    // One per object now, so there is nothing to pool.
    status: row.followup?.status ?? '',
    ztforce: row.ztforce ? 'yes' : '',
    comments,
    commentList,
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
