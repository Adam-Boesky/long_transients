/** The ~300 catalog columns, grouped so a human can read them.
 *
 * Grouping comes from `/api/catalog/schema`, which is stable across rebuilds
 * and fetched once. The detail response stays flat, so nothing repeats the
 * group names per source.
 *
 * Empty columns are hidden by default. Roughly a third of the 301 are null
 * for any given source -- a catalog that did not match contributes ~20 nulls
 * -- and showing them buries the ones that matter.
 */
import { useMemo, useState } from 'react'
import type { CatalogSchema } from '../api/types'
import { formatValue, isEmpty } from '../lib/format'

/** Open on load. The rest start collapsed: ZTF is large and mostly noise. */
const INITIALLY_OPEN = ['tns', 'simbad', 'gaia', 'sdss', 'desi']

export interface CatalogPanelProps {
  schema: CatalogSchema | null
  catalog: Record<string, unknown> | null
}

export function CatalogPanel({ schema, catalog }: CatalogPanelProps) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(INITIALLY_OPEN))
  const [showEmpty, setShowEmpty] = useState(false)
  const [query, setQuery] = useState('')

  const groups = useMemo(() => {
    if (!schema || !catalog) return []
    const needle = query.trim().toLowerCase()
    return schema.groups.map((g) => {
      const rows = g.columns
        .filter((c) => (showEmpty ? true : !isEmpty(catalog[c])))
        .filter((c) => !needle
          || c.toLowerCase().includes(needle)
          || formatValue(catalog[c]).toLowerCase().includes(needle))
        .map((c) => [c, catalog[c]] as const)
      return { ...g, rows, nFilled: g.columns.filter((c) => !isEmpty(catalog[c])).length }
    }).filter((g) => g.rows.length > 0 || !needle)
  }, [schema, catalog, showEmpty, query])

  if (!schema || !catalog) {
    return <div style={{ opacity: 0.6, font: '13px system-ui' }}>Loading catalog…</div>
  }

  const toggle = (key: string) => setOpen((prev) => {
    const next = new Set(prev)
    if (next.has(key)) next.delete(key); else next.add(key)
    return next
  })

  return (
    <div style={{ font: '13px system-ui' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
        <input
          value={query} onChange={(e) => setQuery(e.target.value)}
          placeholder="filter columns or values"
          style={{ flex: 1, padding: '5px 8px', font: 'inherit',
                   border: '1px solid var(--border-strong)', borderRadius: 5,
                   background: 'var(--bg)', color: 'inherit' }}
        />
        <label style={{ whiteSpace: 'nowrap', opacity: 0.8 }}>
          <input type="checkbox" checked={showEmpty}
                 onChange={(e) => setShowEmpty(e.target.checked)} /> show empty
        </label>
      </div>

      {groups.map((g) => {
        const isOpen = open.has(g.key) || query.trim().length > 0
        return (
          <section key={g.key} style={{ marginBottom: 6 }}>
            <button
              onClick={() => toggle(g.key)}
              style={{ width: '100%', textAlign: 'left', cursor: 'pointer',
                       font: '600 13px system-ui', padding: '6px 8px',
                       border: '1px solid var(--border)', borderRadius: 5,
                       background: 'var(--bg-soft)', color: 'inherit' }}
            >
              <span style={{ opacity: 0.6, marginRight: 6 }}>{isOpen ? '▾' : '▸'}</span>
              {g.label}
              <span style={{ opacity: 0.55, fontWeight: 400, marginLeft: 8 }}>
                {g.nFilled} of {g.n_columns}
              </span>
            </button>
            {isOpen && (
              <table style={{ width: '100%', borderCollapse: 'collapse',
                              margin: '2px 0 0', tableLayout: 'fixed' }}>
                <tbody>
                  {g.rows.map(([col, value]) => (
                    <tr key={col}>
                      <td style={{ padding: '2px 8px', opacity: 0.7, width: '48%',
                                   wordBreak: 'break-word',
                                   borderBottom: '1px solid var(--border)' }}>
                        {col}
                      </td>
                      <td style={{ padding: '2px 8px', fontVariantNumeric: 'tabular-nums',
                                   wordBreak: 'break-word',
                                   opacity: isEmpty(value) ? 0.4 : 1,
                                   borderBottom: '1px solid var(--border)' }}>
                        {formatValue(value)}
                      </td>
                    </tr>
                  ))}
                  {g.rows.length === 0 && (
                    <tr><td colSpan={2} style={{ padding: '4px 8px', opacity: 0.5 }}>
                      nothing populated
                    </td></tr>
                  )}
                </tbody>
              </table>
            )}
          </section>
        )
      })}
    </div>
  )
}
