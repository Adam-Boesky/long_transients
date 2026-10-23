/** Build a filter over any of the catalog's 318 columns.
 *
 * The page used to offer four filters, all of them on the handful of columns
 * that happened to be promoted out of the JSON blob. This offers all of them,
 * because the engine behind it reads the columnar catalog rather than
 * inflating 314 columns of JSON per row.
 *
 * Three things the UI has to do, or it hands you a cut you cannot trust:
 *
 *   fill rate    SIMBAD matches 26% of sources, TNS 0.6%. A cut on either is
 *                a cut on that subset, and the picker says so before you
 *                pick, not after you wonder why you have 183 sources.
 *   real bounds  A slider that does not know its own range, or a text box
 *                for a column with 3 distinct values, is a guessing game.
 *                Both come from `/filters/stats`, fetched when you pick.
 *   missing      `otype != 'Star'` matches 51,471 sources or 197,106
 *                depending on what you meant by it. So it asks.
 *
 * The builder writes one level of AND/OR. The document format nests
 * arbitrarily and the server will run whatever it is given, so a filter
 * pasted into the URL or loaded from somewhere deeper still round-trips --
 * it just renders here as a clause this UI cannot edit, rather than being
 * silently flattened.
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { getColumnStats, getFilterSchema, isAbort, previewFilter } from '../api/client'
import { isExpr, isGroup } from '../api/types'
import type {
  ColumnStats, FilterColumn, FilterGroup, FilterLeaf, FilterNode, FilterOp,
  FilterSchema, SourceQuery,
} from '../api/types'
import { Card, Label, field } from './ui'
import { useStableValue } from '../lib/useStableValue'

/** Operators offered per column type. Ordered by how often you want them,
 *  not alphabetically -- `between` is the one you reach for on a magnitude. */
const OPS_BY_TYPE: Record<FilterColumn['type'], FilterOp[]> = {
  number: ['between', '>', '>=', '<', '<=', '=', '!=', 'is_null', 'is_not_null'],
  text: ['=', '!=', 'contains', 'in', 'not_in', 'startswith', 'endswith',
         'has_element', 'is_null', 'is_not_null'],
  bool: ['is_true', 'is_false', 'is_null', 'is_not_null'],
}

const NO_VALUE: FilterOp[] = ['is_null', 'is_not_null', 'is_true', 'is_false']

/** Columns whose values are a comma-joined list rather than one value, so
 *  `has_element` is the right default rather than `contains`. */
const LIST_COLUMNS = new Set(['surveys', 'spec_surveys'])

export interface FilterBuilderProps {
  value: FilterNode | null
  onChange: (next: FilterNode | null) => void
  /** The rest of the sidebar's filters, so every count here is scoped the
   *  same way the page header is. Without it the panel happily reports
   *  111,121 matches beside a header reading 1,270 -- both true, and no way
   *  for the reader to tell which question was answered. */
  scope?: SourceQuery
}

/** A leaf with sensible defaults for a freshly picked column. */
function leafFor(col: FilterColumn): FilterLeaf {
  if (col.type === 'bool') return { column: col.name, op: 'is_true' }
  if (col.type === 'number') return { column: col.name, op: '>', value: 0 }
  return {
    column: col.name,
    op: LIST_COLUMNS.has(col.name) ? 'has_element' : '=',
    value: '',
  }
}

export function FilterBuilder({ value, onChange, scope }: FilterBuilderProps) {
  const [schema, setSchema] = useState<FilterSchema | null>(null)
  const [total, setTotal] = useState<number | null>(null)
  const [scopeTotal, setScopeTotal] = useState<number | null>(null)
  const [counts, setCounts] = useState<(number | null)[]>([])
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const ac = new AbortController()
    getFilterSchema(ac.signal).then(setSchema)
      .catch((e) => { if (!isAbort(e)) setErr(String(e)) })
    return () => ac.abort()
  }, [])

  // The document this builder edits, always as a top-level group so adding a
  // second clause does not change the shape of what the parent holds.
  const group = useMemo<FilterGroup & { op: 'and' | 'or' }>(() => {
    if (!value) return { op: 'and', clauses: [] }
    if (isGroup(value) && value.op !== 'not') {
      return value as FilterGroup & { op: 'and' | 'or' }
    }
    return { op: 'and', clauses: [value] }
  }, [value])

  const emit = useCallback((clauses: FilterNode[], op: 'and' | 'or') => {
    onChange(clauses.length === 0 ? null : { op, clauses })
  }, [onChange])

  // Live counts, debounced: typing "0.85" into a threshold should not fire
  // four requests, and each one is only a few milliseconds server-side but a
  // round trip here.
  // Compared by value: the parent rebuilds `scope` every render, so
  // depending on the object itself would refetch on every keystroke.
  const stableScope = useStableValue(scope ?? {})
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => {
    window.clearTimeout(timer.current)
    if (group.clauses.length === 0) { setTotal(null); setCounts([]); setErr(null); return }
    const ac = new AbortController()
    timer.current = window.setTimeout(() => {
      previewFilter(group, stableScope, ac.signal)
        .then((p) => {
          setTotal(p.total)
          setScopeTotal(p.scope_total ?? null)
          setCounts(p.clauses.map((c) => c.matches ?? null))
          setErr(null)
        })
        .catch((e) => { if (!isAbort(e)) { setErr(String(e)); setTotal(null) } })
    }, 250)
    return () => { window.clearTimeout(timer.current); ac.abort() }
  }, [group, stableScope])

  const update = (i: number, next: FilterNode) => {
    const clauses = group.clauses.slice()
    clauses[i] = next
    emit(clauses, group.op)
  }
  const remove = (i: number) =>
    emit(group.clauses.filter((_, j) => j !== i), group.op)

  if (err && !schema) return <div style={{ color: 'crimson', fontSize: 12 }}>{err}</div>
  if (!schema) return <div style={{ opacity: 0.6, fontSize: 12 }}>loading columns…</div>

  const n = group.clauses.length

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', marginBottom: 6 }}>
        <Label>Column filters</Label>
        {n > 0 && (
          <button onClick={() => onChange(null)}
                  style={{ marginLeft: 'auto', border: 0, background: 'none', font: 'inherit',
                           fontSize: 11, opacity: 0.7, cursor: 'pointer', color: 'inherit' }}>
            clear
          </button>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {n > 1 && (
            <div style={{ display: 'flex', gap: 6, alignItems: 'center',
                          fontSize: 12, opacity: 0.8 }}>
              <span>match</span>
              {(['and', 'or'] as const).map((op) => (
                <button key={op} onClick={() => emit(group.clauses, op)}
                        style={{ ...chip(group.op === op), textTransform: 'none' }}>
                  {op === 'and' ? 'all' : 'any'}
                </button>
              ))}
              <span>of</span>
            </div>
          )}

          {group.clauses.map((clause, i) => (
            <ClauseRow key={i} schema={schema} clause={clause}
                       matches={counts[i] ?? null}
                       onChange={(next) => update(i, next)}
                       onRemove={() => remove(i)} />
          ))}

          <AddClause schema={schema}
                     onAdd={(c) => emit([...group.clauses, c], group.op)} />

          {err && <div style={{ color: 'crimson', fontSize: 12 }}>{err}</div>}
          {n > 0 && typeof total === 'number' && !err && (
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              <b>{total.toLocaleString()}</b> source{total === 1 ? '' : 's'} match
              {typeof scopeTotal === 'number' && scopeTotal !== total
                && ` of ${scopeTotal.toLocaleString()} in the current scope`}
            </div>
          )}
      </div>
    </div>
  )
}



/** An input that holds a draft and only commits on blur or Enter.
 *
 * Committing on every keystroke is not a performance problem here -- the
 * count behind it is a few milliseconds -- it is a correctness one. Each
 * commit rewrites the filter, which renumbers the list, moves the position,
 * rewrites the URL and loads a different source. Typing "0.85" into a
 * threshold would walk you through the filters `0`, `0.`, `0.8` and `0.85`,
 * landing on three sources nobody asked to see. Same reasoning as the jump
 * box in `NavBar`, and the same three ways out: blur, Enter, or Escape to
 * put back what was there.
 */
function DraftInput({ value, onCommit, ...rest }: {
  value: string
  onCommit: (v: string) => void
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>) {
  const [draft, setDraft] = useState(value)
  // Follow changes made elsewhere -- a cleared filter, a swapped column --
  // so the box never sits on a value the document no longer holds.
  useEffect(() => setDraft(value), [value])
  const commit = () => { if (draft !== value) onCommit(draft) }
  return (
    <input {...rest} value={draft}
           onChange={(e) => setDraft(e.target.value)}
           onBlur={commit}
           onKeyDown={(e) => {
             if (e.key === 'Enter') { e.preventDefault(); commit() }
             else if (e.key === 'Escape') setDraft(value)
           }} />
  )
}

/** One editable predicate. */
function ClauseRow({ schema, clause, matches, onChange, onRemove }: {
  schema: FilterSchema
  clause: FilterNode
  matches: number | null
  onChange: (next: FilterNode) => void
  onRemove: () => void
}) {
  const columns = useMemo(
    () => schema.groups.flatMap((g) => g.columns), [schema])

  // A nested group, or anything else this one-level builder did not write.
  // Shown read-only rather than flattened: a filter that arrived by URL or
  // from a saved set is still a real filter, and silently rewriting it would
  // change what the person is looking at.
  if (isGroup(clause)) {
    return (
      <Card pad={8}>
        <Row onRemove={onRemove} matches={matches}>
          <code style={{ fontSize: 11, opacity: 0.8 }}>
            nested {clause.op.toUpperCase()} of {clause.clauses.length}
          </code>
        </Row>
      </Card>
    )
  }

  if (isExpr(clause)) {
    return (
      <Card pad={8}>
        <Row onRemove={onRemove} matches={matches}>
          <DraftInput value={clause.expr} spellCheck={false}
                      onCommit={(v) => onChange({ ...clause, expr: v })}
                      placeholder="ZTF_gPSFMag - PSTARR_gPSFMag > 1.5"
                      style={{ ...field, fontFamily: 'ui-monospace, monospace',
                               fontSize: 12 }} />
        </Row>
        <MissingToggle node={clause} onChange={onChange} />
      </Card>
    )
  }

  const col = columns.find((c) => c.name === clause.column)
  const ops = col ? OPS_BY_TYPE[col.type] : (['=', '!='] as FilterOp[])

  return (
    <Card pad={8}>
      <Row onRemove={onRemove} matches={matches}>
        <ColumnPicker schema={schema} value={clause.column}
                      onChange={(name) => {
                        const next = columns.find((c) => c.name === name)
                        if (next) onChange(leafFor(next))
                      }} />
      </Row>
      <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
        <select value={clause.op} style={{ ...field, width: 'auto' }}
                onChange={(e) => {
                  const op = e.target.value as FilterOp
                  onChange(NO_VALUE.includes(op)
                    ? { column: clause.column, op, missing: clause.missing }
                    : { ...clause, op, value: clause.value ?? '' })
                }}>
          {ops.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
        {!NO_VALUE.includes(clause.op) && col && (
          <ValueInput column={col} clause={clause} onChange={onChange} />
        )}
      </div>
      {!NO_VALUE.includes(clause.op) && (
        <MissingToggle node={clause} onChange={onChange} column={col} />
      )}
    </Card>
  )
}


function Row({ children, matches, onRemove }: {
  children: React.ReactNode; matches: number | null; onRemove: () => void
}) {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
      <div style={{ flex: 1, minWidth: 0 }}>{children}</div>
      {typeof matches === 'number' && (
        <span title="sources this clause keeps, within the current scope"
              style={{ fontSize: 11, opacity: 0.6, whiteSpace: 'nowrap' }}>
          {matches.toLocaleString()}
        </span>
      )}
      <button onClick={onRemove} title="remove"
              style={{ border: 'none', background: 'transparent', cursor: 'pointer',
                       color: 'inherit', opacity: 0.6, fontSize: 15 }}>×</button>
    </div>
  )
}


/** Search across all 318 columns: a combobox with the list below the box.
 *
 * The WAI-ARIA "editable combobox with list autocomplete" pattern, as most
 * search bars behave: the best match is highlighted as you type, so Enter takes
 * it; arrows move the highlight, Escape closes, a click picks. Typing alone
 * never commits, or walking towards `ra_association_...` would pick `ra` on the
 * way past and reload the page under you.
 *
 * The list is portalled and fixed-positioned under the input, because the
 * sidebar it sits in scrolls and would clip an absolutely positioned one.
 */
const MAX_OPTIONS = 60

function ColumnPicker({ schema, value, onChange, autoFocus, onCancel }: {
  schema: FilterSchema; value: string; onChange: (name: string) => void
  autoFocus?: boolean
  /** Called when it is left without picking anything. */
  onCancel?: () => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLUListElement | null>(null)
  const id = useId()

  // Only matches once something is typed; the whole catalogue before that.
  const options = useMemo(() => {
    const ranked = rankColumns(schema, draft)
    return (draft.trim() ? ranked.filter((o) => o.r < 4) : ranked).slice(0, MAX_OPTIONS)
  }, [schema, draft])

  const known = (name: string) =>
    schema.groups.flatMap((g) => g.columns).some((c) => c.name === name)

  const pick = (name: string) => {
    setDraft(name); setOpen(false)
    if (name !== value) onChange(name)
  }
  const cancel = () => { setDraft(value); setOpen(false); onCancel?.() }

  // Keep the list under the input while the page or sidebar scrolls.
  useEffect(() => {
    if (!open) return
    const place = () => setRect(inputRef.current?.getBoundingClientRect() ?? null)
    place()
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [open])

  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: 'nearest' })
  }, [active])

  return (
    <>
      <input ref={inputRef} value={draft} spellCheck={false} autoComplete="off" autoFocus={autoFocus}
             role="combobox" aria-expanded={open} aria-controls={id}
             aria-activedescendant={open && options[active] ? `${id}-${active}` : undefined}
             placeholder="column"
             onFocus={(e) => { e.target.select(); setActive(0); setOpen(true) }}
             onChange={(e) => { setDraft(e.target.value); setActive(0); setOpen(true) }}
             onBlur={() => { if (known(draft)) pick(draft); else cancel() }}
             onKeyDown={(e) => {
               if (e.key === 'ArrowDown') {
                 e.preventDefault(); setOpen(true)
                 setActive((i) => Math.min(i + 1, options.length - 1))
               } else if (e.key === 'ArrowUp') {
                 e.preventDefault(); setActive((i) => Math.max(i - 1, 0))
               } else if (e.key === 'Enter') {
                 e.preventDefault()
                 const o = open ? options[active] : undefined
                 if (o) pick(o.col.name)
                 else if (known(draft)) pick(draft)
               } else if (e.key === 'Escape') {
                 cancel()
               }
             }}
             style={{ ...field, fontFamily: 'ui-monospace, monospace', fontSize: 12 }} />
      {open && rect && options.length > 0 && createPortal(
        <ul ref={listRef} id={id} role="listbox"
            // Keeps focus in the input so a click is not lost to its blur.
            onMouseDown={(e) => e.preventDefault()}
            style={{ position: 'fixed', top: rect.bottom + 2, left: rect.left,
                     width: Math.max(rect.width, 320), maxHeight: 300, overflowY: 'auto',
                     margin: 0, padding: 4, listStyle: 'none', zIndex: 1000,
                     background: 'var(--bg)', color: 'var(--fg, inherit)',
                     border: '1px solid var(--border-strong)', borderRadius: 6,
                     boxShadow: '0 6px 18px rgba(0,0,0,0.15)', font: '12px system-ui' }}>
          {options.map((o, i) => (
            <li key={o.col.name} id={`${id}-${i}`} role="option" aria-selected={i === active}
                onMouseEnter={() => setActive(i)}
                onClick={() => pick(o.col.name)}
                style={{ display: 'flex', gap: 10, alignItems: 'baseline',
                         padding: '4px 8px', borderRadius: 4, cursor: 'pointer',
                         background: i === active ? 'rgba(37, 99, 235, 0.14)'
                                                  : 'transparent' }}>
              <span style={{ fontFamily: 'ui-monospace, monospace' }}>{o.col.name}</span>
              <span style={{ marginLeft: 'auto', opacity: 0.55, whiteSpace: 'nowrap' }}>
                {o.group} · {Math.round(o.col.fill * 100)}%
              </span>
            </li>
          ))}
        </ul>,
        document.body,
      )}
    </>
  )
}


/** Columns ordered best match first, match-sorter style: exact, prefix,
 *  a `_`-separated word starting with the query, then anywhere. Ties go to the
 *  shorter name, then the better-filled column. Non-matches keep schema order. */
function rankColumns(schema: FilterSchema, query: string) {
  const q = query.trim().toLowerCase()
  const rank = (name: string) => {
    const n = name.toLowerCase()
    if (!q) return 0
    if (n === q) return 0
    if (n.startsWith(q)) return 1
    if (n.split(/[_.]/).some((w) => w.startsWith(q))) return 2
    if (n.includes(q)) return 3
    return 4
  }
  const all = schema.groups.flatMap((g, gi) => g.columns
    .filter((c) => c.filterable)
    .map((col, ci) => ({ col, group: g.label, order: gi * 10000 + ci, r: rank(col.name) })))
  return all.sort((a, b) => a.r - b.r
    || (a.r < 4 && q ? a.col.name.length - b.col.name.length || b.col.fill - a.col.fill : 0)
    || a.order - b.order)
}


/** The value editor, shaped by what the column actually holds. */
function ValueInput({ column, clause, onChange }: {
  column: FilterColumn
  clause: FilterLeaf
  onChange: (next: FilterLeaf) => void
}) {
  const [stats, setStats] = useState<ColumnStats | null>(null)
  useEffect(() => {
    setStats(null)
    const ac = new AbortController()
    getColumnStats(column.name, ac.signal).then(setStats)
      .catch((e) => { if (!isAbort(e)) setStats(null) })
    return () => ac.abort()
  }, [column.name])

  const listId = `vals-${column.name}`

  if (clause.op === 'between') {
    const pair = Array.isArray(clause.value) ? clause.value as unknown[] : ['', '']
    return (
      <>
        {[0, 1].map((i) => (
          <DraftInput key={i} type="number" value={String(pair[i] ?? '')}
                      placeholder={stats ? String(
                        i === 0 ? round(stats.quantiles?.p01 ?? stats.min)
                                : round(stats.quantiles?.p99 ?? stats.max)) : ''}
                      onCommit={(v) => {
                        const next = pair.slice()
                        next[i] = v === '' ? '' : Number(v)
                        onChange({ ...clause, value: next })
                      }}
                      style={{ ...field, width: 0, flex: 1 }} />
        ))}
      </>
    )
  }

  if (clause.op === 'in' || clause.op === 'not_in') {
    const list = Array.isArray(clause.value) ? clause.value as unknown[] : []
    return (
      <DraftInput value={list.join(', ')} placeholder="a, b, c"
                  onCommit={(v) => onChange({
                    ...clause,
                    value: v.split(',').map((x) => x.trim()).filter(Boolean),
                  })}
                  style={{ ...field, flex: 1 }} />
    )
  }

  return (
    <>
      <DraftInput list={stats?.values ? listId : undefined}
                  type={column.type === 'number' ? 'number' : 'text'}
                  value={clause.value === undefined || clause.value === null
                           ? '' : String(clause.value)}
                  placeholder={numericHint(stats)}
                  onCommit={(v) => onChange({
                    ...clause,
                    value: column.type === 'number' && v !== '' ? Number(v) : v,
                  })}
                  style={{ ...field, flex: 1 }} />
      {stats?.values && (
        <datalist id={listId}>
          {stats.values.map((v) => (
            <option key={String(v.value)} value={String(v.value)}
                    label={`${v.count.toLocaleString()}`} />
          ))}
        </datalist>
      )}
    </>
  )
}


/** Whether sources missing this column survive the cut.
 *
 * Rendered as a sentence rather than a checkbox labelled "include nulls",
 * because the consequence is the thing people get wrong, not the mechanism.
 * Only shown when it can actually change the answer -- a fully populated
 * column has no missing sources to argue about.
 */
function MissingToggle({ node, onChange, column }: {
  node: FilterLeaf | { expr: string; missing?: 'exclude' | 'include' }
  onChange: (next: never) => void
  column?: FilterColumn
}) {
  if (column && column.fill >= 1) return null
  const include = node.missing === 'include'
  const absent = column ? Math.round((1 - column.fill) * 100) : null
  return (
    <label style={{ display: 'block', marginTop: 6, fontSize: 11, opacity: 0.75 }}>
      <input type="checkbox" checked={include}
             onChange={(e) => onChange({
               ...node, missing: e.target.checked ? 'include' : 'exclude',
             } as never)} />
      {' '}keep sources with no value
      {absent !== null && absent > 0 && ` (${absent}% of the catalog)`}
    </label>
  )
}


function AddClause({ schema, onAdd }: {
  schema: FilterSchema; onAdd: (clause: FilterNode) => void
}) {
  const [picking, setPicking] = useState(false)
  if (!picking) {
    return (
      <div style={{ display: 'flex', gap: 6 }}>
        <button onClick={() => setPicking(true)}
                style={{ ...field, cursor: 'pointer', fontSize: 12, flex: 1 }}>
          + Filter
        </button>
        <button onClick={() => onAdd({ expr: '' })}
                title="an arithmetic comparison across columns"
                style={{ ...field, cursor: 'pointer', fontSize: 12, flex: 1 }}>
          + Expression
        </button>
      </div>
    )
  }
  return (
    <ColumnPicker schema={schema} value="" autoFocus
                  onCancel={() => setPicking(false)}
                  onChange={(name) => {
                    const col = schema.groups.flatMap((g) => g.columns)
                      .find((c) => c.name === name)
                    if (col) { onAdd(leafFor(col)); setPicking(false) }
                  }} />
  )
}


const round = (v: number | undefined) =>
  v === undefined ? '' : Math.round(v * 1000) / 1000

function numericHint(stats: ColumnStats | null): string {
  if (!stats || stats.min === undefined) return ''
  return `${round(stats.min)} … ${round(stats.max)}`
}

function chip(active: boolean): React.CSSProperties {
  return {
    font: 'inherit', fontSize: 12, padding: '2px 8px', borderRadius: 999,
    border: '1px solid var(--border-strong)', cursor: 'pointer', color: 'inherit',
    background: active ? 'var(--accent-bg)' : 'transparent',
    fontWeight: active ? 600 : 400,
  }
}
