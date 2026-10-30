/** Every source in one sortable table: properties, verdicts and comments.
 *
 * Laid out like `app_pages/table.py`, top to bottom:
 *
 *   header    title
 *   card      catalog · label state · the Filters expander (with reset)
 *   search    one box over everything the table shows
 *   caption   how many rows survived, and the hint about sorting
 *   table     open · file · catalog · mean |dmag| · ra · dec · W1-W2 ·
 *             classification · bucket · comments
 *
 * The whole sample arrives in one `/api/table` request and every cut runs
 * locally, mirroring the Streamlit page's pandas masks over the manifest.
 * That is what keeps a slider drag interactive; at hundreds of rows the
 * filter costs less than the round trip would.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { isAbort } from '../api/client'
import { getTable, getTableAnnotations } from '../api/client'
import { FilterBuilder } from '../components/FilterBuilder'
import { Sidebar } from '../components/Sidebar'
import { useFilters } from '../lib/filterState'
import { useStableValue } from '../lib/useStableValue'
import type { TableAnnotations, TableRow } from '../api/types'
import { PageHeader } from '../components/chrome'
import {
  buildRow, LIST_SORT_PREFIX, listColumnOf, serverSortKey, sortRows,
  triageOrder, w1w2Text,
} from '../lib/tableModel'
import type { DisplayRow, SortKey } from '../lib/tableModel'

interface Column {
  key: SortKey
  label: string
  width?: number
  /** Numbers read as columns of digits, so they are right-aligned. */
  numeric?: boolean
  cell: (r: DisplayRow) => string
}

// 4 decimals on dmag, not 2: the sample clusters tightly enough that two
// decimals collapsed runs of distinct sources onto the same displayed value
// and read as duplicate rows.
/** The scroll viewport, and one row's height inside it.
 *
 * Both are fixed because the body is virtualised: only the rows you can see
 * exist in the DOM, and the ones above and below are two spacer cells sized
 * to what they stand in for. Rendering all 5,000 built 50,000 cells on every
 * sort and froze the tab for tens of seconds -- the query behind it takes
 * 35 ms, so the browser was the entire cost.
 *
 * A fixed height is what makes the arithmetic exact; the cells clip rather
 * than wrap, which they already did.
 */
const VIEWPORT_HEIGHT = 620
const ROW_HEIGHT = 25
/** The sticky header, which the viewport has to leave room for when the
 *  list is short enough that the box shrinks to fit. */
const HEADER_HEIGHT = 30
/** Rows kept rendered beyond each edge, so a flick does not show a gap. */
const OVERSCAN = 8

const COLUMNS: Column[] = [
  { key: 'file', label: 'File', width: 196, cell: (r) => r.file },
  { key: 'catalog', label: 'Catalog', width: 124, cell: (r) => r.catalog },
  { key: 'sigma', label: 'Mean |σ|', width: 84, numeric: true,
    cell: (r) => r.sigma?.toFixed(2) ?? '' },
  { key: 'dmag', label: 'Mean |dmag|', width: 96, numeric: true,
    cell: (r) => r.dmag?.toFixed(4) ?? '' },
  { key: 'ra', label: 'RA', width: 92, numeric: true,
    cell: (r) => r.ra?.toFixed(5) ?? '' },
  { key: 'dec', label: 'Dec', width: 92, numeric: true,
    cell: (r) => r.dec?.toFixed(5) ?? '' },
  { key: 'w1w2', label: 'W1-W2', width: 76, numeric: true,
    cell: (r) => w1w2Text(r.w1w2) },
  { key: 'classification', label: 'Classification', width: 200,
    cell: (r) => r.classification },
  { key: 'bucket', label: 'Bucket', width: 104, cell: (r) => r.bucket },
  { key: 'status', label: 'Follow-up Status', width: 120, cell: (r) => r.status },
  { key: 'ztforce', label: 'ztforce', width: 64, cell: (r) => r.ztforce },
  { key: 'comments', label: 'Comments', width: 360, cell: (r) => r.comments },
]
/** A value column of the applied coordinate list, shown beside W1-W2. */
function listColumn(name: string): Column {
  return { key: `${LIST_SORT_PREFIX}${name}`, label: name, width: 90, numeric: true,
           cell: (r) => { const v = r.listValues[name]; return v == null ? '' : v.toFixed(3) } }
}
const LIST_COLUMNS_AFTER = COLUMNS.findIndex((c) => c.key === 'w1w2') + 1
const PAGE_SIZES = [50, 100, 250, 500, 1000, 5000]
const PAGE_SIZE_KEY = 'table-page-size'

function loadPageSize(): number {
  try {
    const n = Number(localStorage.getItem(PAGE_SIZE_KEY))
    return PAGE_SIZES.includes(n) ? n : 100
  } catch { return 100 }
}

const DEFAULT_WIDTHS = Object.fromEntries(
  COLUMNS.map((c) => [c.key, c.width ?? 120])) as Record<string, number>
const MIN_COL_WIDTH = 40
const WIDTHS_KEY = 'table-column-widths'

function loadWidths(): Record<string, number> {
  try {
    return { ...DEFAULT_WIDTHS,
             ...JSON.parse(localStorage.getItem(WIDTHS_KEY) ?? '{}') }
  } catch { return DEFAULT_WIDTHS }
}

export function TablePage({ member }: { member: string }) {
  // What `/api/table` returns: the catalog half, before the verdicts are
  // joined on. `TableRow` is the joined shape.
  const [rows, setRows] =
    useState<Omit<TableRow, 'annotations' | 'ztforce'>[] | null>(null)
  // Every author's verdicts, for every source -- not just this page's.
  const [ann, setAnn] = useState<TableAnnotations | null>(null)
  // What the database holds, which is not always what arrived: the endpoint
  // caps a request, and the identity tier is far larger than any one cut.
  const [total, setTotal] = useState(0)
  const [err, setErr] = useState<string | null>(null)
  // Null until the rows land: the catalog list and the dmag span that seed
  // the defaults are properties of the data, not constants.
  const [query, setQuery] = useState('')
  // Null means the triage order -- biggest change first -- which is what the
  // table opens on and what a header click replaces.
  // The default is `dmag` descending: the table's triage order, and now the
  // server's, so the largest change in the *catalog* is on the first screen
  // rather than the largest among the first 5,000 sources by uuid.
  const scroller = useRef<HTMLDivElement>(null)
  // Only what this page itself reads. Every other cut the sidebar offers
  // reaches it through `query`, and the sidebar sets it from the provider
  // directly rather than through twenty props forwarded from here.
  const { labelState, columnFilter, setColumnFilter,
          query: sharedQuery, coordList } = useFilters()
  // The applied list's value columns sit beside the catalog's, and sort
  // across the whole list like them; they go when the list does.
  const listColumns = coordList?.columns ?? []
  const columns = useMemo(() => [
    ...COLUMNS.slice(0, LIST_COLUMNS_AFTER), ...listColumns.map(listColumn),
    ...COLUMNS.slice(LIST_COLUMNS_AFTER),
  ], [listColumns])
  const [scrollTop, setScrollTop] = useState(0)
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>(
    { key: 'sigma', desc: true })
  useEffect(() => {
    const column = listColumnOf(sort.key)
    if (column !== null && !listColumns.includes(column)) setSort({ key: 'sigma', desc: true })
  }, [sort.key, listColumns])
  const [widths, setWidths] = useState(loadWidths)
  useEffect(() => {
    try { localStorage.setItem(WIDTHS_KEY, JSON.stringify(widths)) } catch { /* private window */ }
  }, [widths])
  // Set while a resize drag is in progress, so its mouseup is not a sort click.
  const resizing = useRef(false)
  const startResize = (key: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    resizing.current = true
    const x0 = e.clientX
    const w0 = widths[key]
    const move = (ev: MouseEvent) => setWidths((w) => (
      { ...w, [key]: Math.max(MIN_COL_WIDTH, (w0 ?? MIN_COL_WIDTH) + ev.clientX - x0) }))
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      setTimeout(() => { resizing.current = false })
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }
  // Comments takes any spare width, but never less than its own.
  const widthOf = (c: Column) => widths[c.key] ?? c.width ?? MIN_COL_WIDTH
  const minTableWidth = 54 + columns.reduce((n, c) => n + widthOf(c), 0)
  const serverKey = serverSortKey(sort.key)
  // The catalog toggle goes to the server too. It used to cut the window
  // after it arrived, so picking tier 0 showed 3,826 of 186,970 tier-0
  // sources -- the tier-0 members of a window chosen across every tier.
  // Sent as a filter, the window is the top 5,000 *of that tier*.
  // Everything the server can answer, compared by value -- the object is
  // rebuilt every render, so depending on it would refetch whenever anything
  // else on the page re-renders.
  const scope = useStableValue(sharedQuery)
  const [pageSize, setPageSize] = useState(loadPageSize)
  useEffect(() => {
    try { localStorage.setItem(PAGE_SIZE_KEY, String(pageSize)) } catch { /* private window */ }
  }, [pageSize])
  // The page number belongs to one sample and ordering; any change to those
  // starts back at page 1 without a second fetch for the stale page.
  const pageKey = JSON.stringify([serverKey, sort.desc, scope, labelState,
                                  member, pageSize])
  const [pageState, setPageState] = useState({ key: pageKey, n: 0 })
  const page = pageState.key === pageKey ? pageState.n : 0
  const setPage = (n: number) => setPageState({ key: pageKey, n })
  const nPages = Math.max(1, Math.ceil(total / pageSize))
  const activeLabel = columns.find((c) => c.key === sort.key)?.label ?? sort.key

  // Two fetches on separate schedules, joined below. The catalog half is a
  // window into a static release and depends on the sort, the page and the
  // cuts; the verdicts are the whole annotation set and depend on none of
  // them. Joined in one `Promise.all`, flipping a sort column refetched every
  // annotation in the database to redraw the same comments.
  useEffect(() => {
    const ac = new AbortController()
    getTable({
      ...scope,
      offset: page * pageSize, limit: pageSize,
      // An annotation column sorts locally, over the default |σ| window.
      // Sending no sort here fell back to the server's uuid order instead,
      // a different 5,000 rows that could hold none of the comments.
      ...(serverKey ? { sort: serverKey, desc: sort.desc }
                    : { sort: 'mean_abs_dmag_sigma', desc: true }),
      ...(labelState !== 'All' ? { label_state: labelState, author: member } : {}),
    }, ac.signal)
      .then((p) => { setRows(p.rows); setTotal(p.total) })
      .catch((e) => { if (!isAbort(e)) setErr(String(e)) })
    return () => ac.abort()
    // Refetch when the server-side ordering changes: a different order is a
    // different 5,000 rows, which is the whole point.
  }, [serverKey, sort.desc, scope, labelState, member, page, pageSize])

  // Once per visit. Verdicts are filed on the Classify page, which is a
  // different route, so coming back here remounts and re-asks.
  useEffect(() => {
    const ac = new AbortController()
    getTableAnnotations(ac.signal)
      .then(setAnn)
      .catch((e) => { if (!isAbort(e)) setErr(String(e)) })
    return () => ac.abort()
  }, [])

  // Rows keep the shape the rest of the page expects, so only this join
  // knows about the split.
  const joined = useMemo(() => {
    if (!rows || !ann) return null
    const done = new Set(ann.ztforce_done)
    return rows.map((r) => ({
      ...r, annotations: ann.annotations[r.source_uuid] ?? [],
      ztforce: done.has(r.source_uuid),
    }))
  }, [rows, ann])

  // No local cut left: every filter the page offers is applied server-side,
  // so the rows that arrive are already the sample. The free-text box below
  // is the one exception and runs over `table`.

  // Rows arrive in the order the server chose, so only re-sort locally for
  // the columns it cannot order by. Re-applying `triageOrder` here would
  // throw away a server sort and silently re-rank the window by dmag.
  const table = useMemo(() => {
    const built = (joined ?? []).map(buildRow)
    return serverKey ? built : triageOrder(built)
  }, [joined, serverKey])

  const searched = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return needle ? table.filter((r) => r.search.includes(needle)) : table
  }, [table, query])

  const shown = useMemo(
    () => (serverKey ? searched : sortRows(searched, sort.key, sort.desc)),
    [searched, sort, serverKey])

  // Only the rows on screen are rendered. Everything above and below is a
  // spacer of the right height, so the scrollbar still describes the whole
  // list and nothing else has to know the body is virtualised.
  const firstVisible = Math.max(
    0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const lastVisible = Math.min(
    shown.length,
    Math.ceil((scrollTop + VIEWPORT_HEIGHT) / ROW_HEIGHT) + OVERSCAN)
  const visible = useMemo(
    () => shown.slice(firstVisible, lastVisible),
    [shown, firstVisible, lastVisible])

  // A new ordering or a new sample puts you at the top of the list, not at
  // whatever pixel offset you happened to be at in the previous one. `rows`
  // belongs here too: without it a sidebar filter swapped the list underneath
  // a stale offset, landing you mid-table or on the spacer rows.
  useEffect(() => { setScrollTop(0); scroller.current?.scrollTo({ top: 0 }) },
            [sort, query, rows])

  if (err) return <pre style={{ color: 'crimson' }}>{err}</pre>

  // The panel renders whether or not the catalog has arrived. Returning a
  // bare "Loading…" in its place unmounted the sidebar on every tab switch,
  // which read as the filters flashing away and coming back.
  const sidebar = (
    <Sidebar mode="table"
        columnFilters={<FilterBuilder value={columnFilter}
                                      onChange={setColumnFilter}
                                      scope={sharedQuery} />} />
  )

  if (!joined) {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '236px 1fr',
                    gap: 18 }}>
        {sidebar}
        <div style={{ minWidth: 0 }}>
          <PageHeader title="Table" large />
          <div style={{ opacity: 0.6 }}>Loading the catalog…</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '236px 1fr', gap: 18 }}>
      {sidebar}

    <div style={{ minWidth: 0 }}>
      <PageHeader title="Table" large />

      <input
        value={query} onChange={(e) => setQuery(e.target.value)}
        placeholder="search this page: filename, classification, bucket, comment or coordinates"
        style={{ width: '100%', padding: '6px 9px', font: 'inherit',
                 borderRadius: 5, border: '1px solid var(--border-strong)',
                 background: 'var(--bg)', color: 'inherit',
                 marginBottom: 8 }} />

      {shown.length === 0 ? (
        <div style={{ opacity: 0.6 }}>
          No sources match the current filters and search.
        </div>
      ) : (
        <>
          <div style={{ color: 'var(--text-dim)', marginBottom: 6 }}>
            {(page * pageSize + 1).toLocaleString()}–
            {(page * pageSize + joined.length).toLocaleString()} of{' '}
            {total.toLocaleString()} sources
            {shown.length !== table.length
              && ` · ${shown.length.toLocaleString()} on this page match the search`}
            {serverKey
              ? ` · sorted by ${activeLabel} across all ${total.toLocaleString()}`
              : ` · sorted by ${activeLabel} within this page only`}
          </div>

          <div ref={scroller}
               onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
               style={{ height: Math.min(
                          VIEWPORT_HEIGHT,
                          shown.length * ROW_HEIGHT + HEADER_HEIGHT),
                        overflow: 'auto',
                        border: '1px solid var(--border)', borderRadius: 7 }}>
            <table style={{ borderCollapse: 'collapse', width: '100%',
                            minWidth: minTableWidth, tableLayout: 'fixed' }}>
              <colgroup>
                <col style={{ width: 54 }} />
                {columns.map((c) => (
                  <col key={c.key} style={c.key === 'comments' ? undefined
                                          : { width: widthOf(c) }} />
                ))}
              </colgroup>
              <thead>
                <tr>
                  <th style={head(false)} />
                  {columns.map((c) => {
                    const active = sort.key === c.key
                    const whole = serverSortKey(c.key) !== undefined
                    return (
                      <th key={c.key} style={head(c.numeric)}
                          onClick={() => {
                            if (resizing.current) return
                            setSort(active ? { key: c.key, desc: !sort.desc }
                                           : { key: c.key, desc: true })
                          }}
                          title={whole
                            ? `Sort all ${total.toLocaleString()} sources by ${c.label}`
                            : `Sort the loaded rows by ${c.label} -- this column is `
                              + 'built from verdicts, so it cannot be ordered across '
                              + 'the whole catalog'}>
                        {c.label}
                        <span style={{ opacity: active ? 0.8 : 0.25 }}>
                          {active ? (sort.desc ? ' ↓' : ' ↑') : ' ↕'}
                        </span>
                        <span className="col-resize" title="Drag to resize"
                              onMouseDown={(e) => startResize(c.key, e)}
                              onDoubleClick={(e) => {
                                e.stopPropagation()
                                setWidths((w) => ({ ...w, [c.key]: DEFAULT_WIDTHS[c.key] ?? c.width ?? MIN_COL_WIDTH }))
                              }} />
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {/* The rows scrolled past, as one cell of their combined
                    height, so the scrollbar still measures the whole list. */}
                {firstVisible > 0 && (
                  <tr style={{ height: firstVisible * ROW_HEIGHT }} aria-hidden>
                    <td colSpan={columns.length + 1} />
                  </tr>
                )}
                {visible.map((r) => (
                  <tr key={r.uuid} style={{ height: ROW_HEIGHT }}>
                    <td style={cell(false)}>
                      {/* First column, so the jump-to-source link never
                          scrolls out of sight behind the wide comments one. */}
                      <Link to={`/classify?page=${encodeURIComponent(r.uuid)}`}>
                        open
                      </Link>
                    </td>
                    {columns.map((c) => {
                      const text = c.cell(r)
                      return (
                        <td key={c.key} style={cell(c.numeric)} title={text}>
                          {text}
                        </td>
                      )
                    })}
                  </tr>
                ))}
                {lastVisible < shown.length && (
                  <tr style={{ height: (shown.length - lastVisible) * ROW_HEIGHT }}
                      aria-hidden>
                    <td colSpan={columns.length + 1} />
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {total > 0 && (
        <Pager page={page} nPages={nPages} onPage={setPage}
               pageSize={pageSize} onPageSize={setPageSize} />
      )}
    </div>
    </div>
  )
}

function Pager({ page, nPages, onPage, pageSize, onPageSize }: {
  page: number; nPages: number; onPage: (n: number) => void
  pageSize: number; onPageSize: (n: number) => void
}) {
  const btn = (label: string, to: number, title: string) => (
    <button onClick={() => onPage(to)} disabled={to < 0 || to >= nPages || to === page}
            title={title}
            style={{ padding: '3px 9px', font: 'inherit', borderRadius: 5,
                     border: '1px solid var(--border-strong)',
                     background: 'transparent', color: 'inherit',
                     cursor: 'pointer' }}>
      {label}
    </button>
  )
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                  justifyContent: 'center', margin: '10px 0' }}>
      {btn('«', 0, 'First page')}
      {btn('‹', page - 1, 'Previous page')}
      <span style={{ padding: '0 6px' }}>
        Page{' '}
        <input type="number" min={1} max={nPages} value={page + 1}
               onChange={(e) => {
                 const n = Math.round(Number(e.target.value)) - 1
                 if (n >= 0 && n < nPages) onPage(n)
               }}
               style={{ width: 64, padding: '2px 4px', font: 'inherit',
                        borderRadius: 4, border: '1px solid var(--border-strong)',
                        background: 'var(--bg)', color: 'inherit' }} />
        {' '}of {nPages.toLocaleString()}
      </span>
      {btn('›', page + 1, 'Next page')}
      {btn('»', nPages - 1, 'Last page')}
      <label style={{ marginLeft: 14, color: 'var(--text-dim)' }}>
        Rows per page{' '}
        <select value={pageSize} onChange={(e) => onPageSize(Number(e.target.value))}
                style={{ font: 'inherit', padding: '2px 4px', borderRadius: 4,
                         border: '1px solid var(--border-strong)',
                         background: 'var(--bg)', color: 'inherit' }}>
          {PAGE_SIZES.map((n) => <option key={n} value={n}>{n.toLocaleString()}</option>)}
        </select>
      </label>
    </div>
  )
}

function head(numeric?: boolean) {
  return {
    position: 'sticky' as const, top: 0, zIndex: 1,
    background: 'var(--bg-soft)', cursor: 'pointer', userSelect: 'none' as const,
    textAlign: (numeric ? 'right' : 'left') as 'right' | 'left',
    padding: '6px 8px', borderBottom: '1px solid var(--border-strong)',
    font: '600 12px system-ui', whiteSpace: 'nowrap' as const,
    overflow: 'hidden', textOverflow: 'ellipsis' as const,
  }
}

function cell(numeric?: boolean) {
  return {
    padding: '3px 8px', borderBottom: '1px solid var(--border)',
    textAlign: (numeric ? 'right' : 'left') as 'right' | 'left',
    whiteSpace: 'nowrap' as const, overflow: 'hidden',
    textOverflow: 'ellipsis' as const,
  }
}
