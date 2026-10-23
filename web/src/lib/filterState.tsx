/** The sidebar's filters, shared by every page that shows the sidebar.
 *
 * Held above the routes rather than inside each page so that narrowing the
 * sample on Classify and switching to Table shows *that* sample. Two copies
 * of this state would make the same panel mean different things depending on
 * which tab you were last on, which is worse than not sharing it at all.
 *
 * Only the cuts both pages understand live here. Which label you are filing,
 * and the comment you are filing it with, stay in `ClassifyPage`: they belong
 * to the source in front of you, not to the selection.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState }
  from 'react'
import type { ReactNode } from 'react'
import { addLabel, ApiError, deleteLabel, getFilterSchema, getLabels } from '../api/client'
import type { FilterNode, Label, SourceQuery } from '../api/types'
import type { LabelState } from './tableModel'


// `LABEL_STATES` stays in `tableModel`, next to the code that applies it.
// Re-exported here so a page reaching for the shared filters does not have to
// know which of the two files each piece came from.
export { LABEL_STATES } from './tableModel'
export type { LabelState } from './tableModel'

/** Which of the pipeline's sub-catalogues to work through: the separate
 *  enriched tables the filter stage emits, tagged `catalog_tier`. Declared
 *  here rather than beside the control that sets it, because the control
 *  imports this module -- the two used to import each other. */
export type CatalogFilter = 'all' | 'in_both' | 'in_ztf'
const CATALOG_FILTERS: readonly string[] = ['all', 'in_both', 'in_ztf']

/** The seven groups of `source_db.unified_class`, in its wedge order. */
export const UNIFIED_CLASSES = ['AGN', 'Galaxy', 'Transient', 'Star',
                                'Multiple Classes', 'Unclassified',
                                'Unmatched'] as const

/** Crossmatched catalogs, by display name -> their `<prefix>_matched` column.
 *  Only those present in the current release's columns are offered. */
export const MATCH_CATALOGS: [string, string][] = [
  ['SDSS', 'sdss_matched'], ['DESI', 'desi_matched'], ['SIMBAD', 'simbad_matched'],
  ['TNS', 'tns_matched'], ['AGN-DB', 'agn_db_matched'], ['WISE', 'wise_matched'],
  ['Gaia', 'gaia_matched'], ['eROSITA', 'erosita_matched'],
]

/** Stern et al. (2012) put the AGN boundary at W1 - W2 = 0.8 (Vega). */
export const W1W2_CUT = 0.8

/** How long the search box waits before narrowing the sample. Long enough to
 *  swallow a typed word, short enough not to feel like lag. */
const SEARCH_DEBOUNCE_MS = 300
export const W1W2_MODES = ['Any', `≥ ${W1W2_CUT}`, `< ${W1W2_CUT}`] as const
export type W1W2Mode = (typeof W1W2_MODES)[number]

export interface FilterState {
  search: string
  setSearch: (v: string) => void
  catalog: CatalogFilter
  setCatalog: (v: CatalogFilter) => void
  /** Per-bucket and per-status include/exclude; a value absent from the map
   *  is neutral. `min epochs` and `only unlabelled` used to live here and are
   *  now expressible as a column filter on `n_epochs` and as excluding every
   *  bucket. */
  bucketFilter: Record<string, 'in' | 'out'>
  setBucketFilter: (v: Record<string, 'in' | 'out'>) => void
  statusFilter: Record<string, 'in' | 'out'>
  setStatusFilter: (v: Record<string, 'in' | 'out'>) => void
  labelState: LabelState
  setLabelState: (v: LabelState) => void
  columnFilter: FilterNode | null
  setColumnFilter: (v: FilterNode | null) => void
  /** Unified class include/exclude, over `UNIFIED_CLASSES`. */
  classFilter: Record<string, 'in' | 'out'>
  setClassFilter: (v: Record<string, 'in' | 'out'>) => void
  /** Has / lacks a match, per crossmatched catalog (display name). */
  matchFilter: Record<string, 'in' | 'out'>
  setMatchFilter: (v: Record<string, 'in' | 'out'>) => void
  /** The `MATCH_CATALOGS` names this release has a column for. */
  matchCatalogs: string[]
  /** Stern+12 colour cut, and whether sources with no WISE colour come too. */
  w1w2: W1W2Mode
  setW1w2: (v: W1W2Mode) => void
  w1w2KeepMissing: boolean
  setW1w2KeepMissing: (v: boolean) => void
  /** Reset every control in the Filters card. The column-filter builder has
   *  its own reset, so it is deliberately left alone. */
  clearFilters: () => void
  /** The shared label vocabulary. Held here rather than per page so that
   *  switching tabs neither refetches it nor blinks an empty checklist, and
   *  so a bucket added from either page shows up on both. */
  labels: string[]
  labelColors: Record<string, string>
  /** Sources filed under each label, as of the last load. */
  labelCounts: Record<string, number>
  setLabels: (next: string[]) => void
  setLabelColors: (f: (c: Record<string, string>) => Record<string, string>) => void
  takeLabels: (ls: Label[]) => void
  reloadLabels: () => Promise<void>
  addBucket: (name: string) => Promise<void>
  /** Rejects with a message to show if the label could not be deleted. */
  deleteBucket: (name: string) => Promise<void>
  /** The parts the server understands, as one object both pages send. */
  query: SourceQuery
}

const Context = createContext<FilterState | null>(null)

export function FilterProvider({ member, children }:
                               { member: string; children: ReactNode }) {
  const [search, setSearch] = useState('')
  // What `query` actually carries. Split from `search` so the input stays
  // responsive while typing without each keystroke re-running the sample:
  // one character used to drop the candidate window and refetch /api/sources,
  // /api/table and /api/table/annotations, so a 12-character slug cost 36
  // requests whose first 33 answers were already stale on arrival.
  const [debouncedSearch, setDebouncedSearch] = useState('')
  useEffect(() => {
    // Immediate when cleared: emptying the box is "show me everything back",
    // and waiting on that reads as the control being broken.
    if (!search) { setDebouncedSearch(''); return }
    const t = setTimeout(() => setDebouncedSearch(search), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [search])
  // One sub-catalogue at a time by default, for the reason the Streamlit app
  // does it: the tiers are different populations and a mixed list is hard to
  // reason about.
  // Seeded from the address bar too: a copied link must open in the tab it was
  // copied from, or an in_both source is "not in the current selection".
  const [catalog, setCatalog] = useState<CatalogFilter>(() => {
    const raw = new URLSearchParams(window.location.search).get('catalog')
    return raw && CATALOG_FILTERS.includes(raw) ? raw as CatalogFilter : 'in_ztf'
  })
  const [bucketFilter, setBucketFilter] =
    useState<Record<string, 'in' | 'out'>>({})
  const [statusFilter, setStatusFilter] =
    useState<Record<string, 'in' | 'out'>>({})
  const [labelState, setLabelState] = useState<LabelState>('All')
  // Seeded from the address bar so a filtered view stays a link.
  const [columnFilter, setColumnFilter] = useState<FilterNode | null>(() => {
    const raw = new URLSearchParams(window.location.search).get('filter')
    if (!raw) return null
    try {
      return JSON.parse(raw) as FilterNode
    } catch {
      return null          // a mangled link filters nothing rather than erroring
    }
  })

  const [classFilter, setClassFilter] =
    useState<Record<string, 'in' | 'out'>>({})
  const [matchFilter, setMatchFilter] =
    useState<Record<string, 'in' | 'out'>>({})
  // Which match columns exist: a release built before a catalog was added has
  // none, and a clause naming a missing column would fail the whole query.
  const [matchCatalogs, setMatchCatalogs] = useState<string[]>([])
  useEffect(() => {
    const ac = new AbortController()
    getFilterSchema(ac.signal).then((schema) => {
      const have = new Set(schema.groups.flatMap((g) => g.columns.map((c) => c.name)))
      setMatchCatalogs(MATCH_CATALOGS.filter(([, col]) => have.has(col)).map(([name]) => name))
    }).catch(() => {})
    return () => ac.abort()
  }, [])
  const [w1w2, setW1w2] = useState<W1W2Mode>('Any')
  // Off by default, matching the cut's own logic: a source with no colour has
  // not passed or failed Stern+12, so it is not part of either answer.
  const [w1w2KeepMissing, setW1w2KeepMissing] = useState(false)
  const [labels, setLabels] = useState<string[]>([])
  const [labelColors, setLabelColors] = useState<Record<string, string>>({})
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({})
  const takeLabels = useCallback((ls: Label[]) => {
    setLabels(ls.map((l) => l.name))
    setLabelCounts(Object.fromEntries(ls.map((l) => [l.name, l.n_sources])))
    setLabelColors(Object.fromEntries(
      ls.filter((l) => l.color).map((l) => [l.name, l.color as string])))
  }, [])
  const reloadLabels = useCallback(
    () => getLabels().then(takeLabels).then(() => undefined), [takeLabels])
  useEffect(() => { reloadLabels().catch(() => {}) }, [reloadLabels])
  const addBucket = useCallback(async (name: string) => {
    try { await addLabel(name, member) }
    catch (e) {
      throw new Error(e instanceof ApiError && e.status === 409
        ? `"${name}" already exists` : String(e))
    }
    await reloadLabels()
  }, [member, reloadLabels])
  const deleteBucket = useCallback(async (name: string) => {
    try { await deleteLabel(name) }
    catch (e) {
      await reloadLabels()
      throw new Error(e instanceof ApiError && e.status === 409
        ? `"${name}" still has sources` : String(e))
    }
    setBucketFilter(({ [name]: _, ...rest }) => rest)
    await reloadLabels()
  }, [reloadLabels])

  const clearFilters = useCallback(() => {
    setBucketFilter({}); setStatusFilter({}); setClassFilter({}); setMatchFilter({})
    setW1w2('Any'); setW1w2KeepMissing(false)
  }, [])

  const pick = (m: Record<string, 'in' | 'out'>, want: 'in' | 'out') => {
    const out = Object.keys(m).filter((k) => m[k] === want).sort()
    return out.length ? out : undefined
  }
  // The class and colour controls are ordinary column filters; they are built
  // here rather than in the sidebar so both pages send one document and the
  // builder's own clauses AND with them.
  const filterDoc = useMemo<FilterNode | null>(() => {
    const clauses: FilterNode[] = []
    const inc = Object.keys(classFilter).filter((k) => classFilter[k] === 'in').sort()
    const out = Object.keys(classFilter).filter((k) => classFilter[k] === 'out').sort()
    if (inc.length) clauses.push({ column: 'unified_class', op: 'in', value: inc })
    if (out.length) clauses.push({ column: 'unified_class', op: 'not_in', value: out })
    // Each catalog is its own condition, ANDed: "in SDSS" and "in eROSITA" means both.
    for (const [name, column] of MATCH_CATALOGS) {
      const want = matchFilter[name]
      if (want) clauses.push({ column, op: want === 'in' ? 'is_true' : 'is_false' })
    }
    if (w1w2 !== 'Any') {
      clauses.push({
        // `wise_w1w2_mean`, not `wise_w1w2`: the first is the visit-binned
        // light-curve colour the histogram plots, the second the AllWISE
        // coadd `w1mpro - w2mpro`. They disagree by a median 0.10 mag and
        // land on opposite sides of 0.8 for 21,776 sources, so filtering on
        // the coadd while showing the mean reads as the panel contradicting
        // the filter.
        column: 'wise_w1w2_mean', op: w1w2 === W1W2_MODES[1] ? '>=' : '<',
        value: W1W2_CUT,
        // The whole point of the checkbox: `exclude` drops the 80,608 sources
        // with no colour, `include` keeps them alongside the ones that passed.
        missing: w1w2KeepMissing ? 'include' : 'exclude',
      })
    }
    if (columnFilter) clauses.push(columnFilter)
    if (!clauses.length) return null
    return clauses.length === 1 ? clauses[0] : { op: 'and', clauses }
  }, [classFilter, matchFilter, w1w2, w1w2KeepMissing, columnFilter])

  const query = useMemo<SourceQuery>(() => {
    // Assigned to a typed local before being returned, deliberately: a literal
    // returned straight out of `useMemo<SourceQuery>` is not
    // excess-property-checked, so a misspelled field compiled fine and was
    // silently dropped from the URL. Through a `const q: SourceQuery` it errors.
    const q: SourceQuery = {
      catalog: catalog === 'all' ? undefined : catalog,
      q: debouncedSearch.trim() || undefined,
      filter: filterDoc ? JSON.stringify(filterDoc) : undefined,
      bucket_in: pick(bucketFilter, 'in'),
      bucket_out: pick(bucketFilter, 'out'),
      status_in: pick(statusFilter, 'in'),
      status_out: pick(statusFilter, 'out'),
    }
    return q
  }, [catalog, debouncedSearch, filterDoc, bucketFilter, statusFilter])

  const value = useMemo<FilterState>(() => ({
    search, setSearch, catalog, setCatalog,
    bucketFilter, setBucketFilter, statusFilter, setStatusFilter,
    labelState, setLabelState, columnFilter, setColumnFilter, query,
    classFilter, setClassFilter, matchFilter, setMatchFilter, matchCatalogs,
    w1w2, setW1w2, w1w2KeepMissing, setW1w2KeepMissing, clearFilters,
    labels, labelColors, labelCounts, setLabels, setLabelColors, takeLabels,
    reloadLabels, addBucket, deleteBucket,
  }), [search, catalog, bucketFilter, statusFilter, labelState, columnFilter,
       classFilter, matchFilter, matchCatalogs, w1w2, w1w2KeepMissing, clearFilters,
       query, labels, labelColors, labelCounts, takeLabels, reloadLabels,
       addBucket, deleteBucket])

  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useFilters(): FilterState {
  const value = useContext(Context)
  if (!value) throw new Error('useFilters outside a FilterProvider')
  return value
}
