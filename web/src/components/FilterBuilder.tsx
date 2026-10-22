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
import { getColumnStats, getFilterSchema, isAbort, previewFilter } from '../api/client'
import { isExpr, isGroup } from '../api/types'
import type {
  ColumnStats, FilterColumn, FilterGroup, FilterLeaf, FilterNode, FilterOp,
  FilterSchema, SourceQuery,
} from '../api/types'
import { Card, field } from './ui'
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
  const [open, setOpen] = useState(false)
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
      <button onClick={() => setOpen((v) => !v)}
              style={{ ...field, textAlign: 'left', cursor: 'pointer',
                       display: 'flex', justifyContent: 'space-between' }}>
        <span>{open ? '▾' : '▸'} Column filters</span>
        {n > 0 && (
          <span style={{ fontSize: 11, padding: '0 6px', borderRadius: 999,
                         border: '1px solid var(--border-strong)' }}>{n}</span>
        )}
      </button>

      {open && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
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
          {typeof total === 'number' && !err && (
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              <b>{total.toLocaleString()}</b> source{total === 1 ? '' : 's'} match
              {typeof scopeTotal === 'number' && scopeTotal !== total
                && ` of ${scopeTotal.toLocaleString()} in the current scope`}
            </div>
          )}
          {n > 0 && (
            <button onClick={() => onChange(null)}
                    style={{ ...field, cursor: 'pointer', width: 'auto',
                             alignSelf: 'flex-start', fontSize: 12 }}>
              clear all
            </button>
          )}
        </div>
      )}
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


/** Search across all 318 columns, grouped, with the fill rate on each.
 *
 * A `<datalist>` rather than a custom dropdown: it is searchable for free,
 * keyboard-navigable for free, and 318 options is exactly the size where a
 * hand-rolled list starts needing virtualisation to stay smooth.
 */
function ColumnPicker({ schema, value, onChange }: {
  schema: FilterSchema; value: string; onChange: (name: string) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  // `useId`, not a random ref: several pickers are on screen at once and each
  // needs its own <datalist>, but a ref initialiser re-evaluates on every
  // render even though only the first value is kept.
  const id = useId()

  const known = (name: string) =>
    schema.groups.flatMap((g) => g.columns).find((c) => c.name === name)

  const commit = (name: string) => {
    if (name === value) return
    if (known(name)) onChange(name)
    else setDraft(value)          // not a column: put back what was there
  }

  return (
    <>
      <input list={id} value={draft} spellCheck={false}
             onChange={(e) => {
               setDraft(e.target.value)
               // Picking from the dropdown commits straight away; typing does
               // not, or walking the keyboard towards `ra_association_...`
               // would commit `ra` on the way past and reload the page under
               // you. Browsers report a list selection as a replacement
               // rather than an insertion, which is the difference.
               const how = (e.nativeEvent as InputEvent).inputType
               if (how === undefined || how === 'insertReplacementText') {
                 commit(e.target.value)
               }
             }}
             onBlur={(e) => commit(e.target.value)}
             onKeyDown={(e) => {
               if (e.key === 'Enter') {
                 e.preventDefault(); commit((e.target as HTMLInputElement).value)
               } else if (e.key === 'Escape') setDraft(value)
             }}
             placeholder="column"
             style={{ ...field, fontFamily: 'ui-monospace, monospace', fontSize: 12 }} />
      <datalist id={id}>
        {schema.groups.map((g) => g.columns.filter((c) => c.filterable).map((c) => (
          <option key={c.name} value={c.name}
                  label={`${g.label} · ${Math.round(c.fill * 100)}% filled`} />
        )))}
      </datalist>
    </>
  )
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
                style={{ ...field, cursor: 'pointer', fontSize: 12 }}>
          + column
        </button>
        <button onClick={() => onAdd({ expr: '' })}
                title="an arithmetic comparison across columns"
                style={{ ...field, cursor: 'pointer', fontSize: 12 }}>
          + expression
        </button>
      </div>
    )
  }
  return (
    <Card pad={8}>
      <ColumnPicker schema={schema} value=""
                    onChange={(name) => {
                      const col = schema.groups.flatMap((g) => g.columns)
                        .find((c) => c.name === name)
                      if (col) { onAdd(leafFor(col)); setPicking(false) }
                    }} />
    </Card>
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
