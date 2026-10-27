/** Pick a saved coordinate list, or upload a new one, to narrow both pages to its sources.
 *
 * Lists are shared: anyone can apply, rename or delete any of them. A new file
 * is matched as a preview first, so nothing is saved until "Save & show", and a
 * delete asks first and only hides the list.
 *
 * Matching happens server-side; the queries carry only the list's id, so the
 * view stays a link however long the list is.
 */
import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  createCoordList, deleteCoordList, listCoordLists, previewCoordList, renameCoordList,
} from '../api/client'
import type { CoordListPreview, CoordListSummary } from '../api/types'
import type { CatalogFilter } from '../lib/filterState'
import { CardTitle, Label, TONE, field } from './ui'

const DEFAULT_RADIUS_ARCSEC = 2

const button = {
  padding: '4px 10px', borderRadius: 5, font: 'inherit', cursor: 'pointer',
  border: '1px solid var(--border-strong)', background: 'transparent', color: 'inherit',
} as const
const small = { ...button, padding: '2px 8px', fontSize: 12 } as const

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`

export function CoordListDialog({ current, catalog, onApply, onClose }: {
  /** The list applied now, if any. */
  current: CoordListSummary | null
  /** The catalog tab in force, to say how many matches it hides. */
  catalog: CatalogFilter
  onApply: (v: CoordListSummary | null) => void
  onClose: () => void
}) {
  const [lists, setLists] = useState<CoordListSummary[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const reload = useCallback(() => {
    listCoordLists().then(setLists).catch((e) => setErr(String(e)))
  }, [])
  useEffect(reload, [reload])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const apply = (l: CoordListSummary | null) => { onApply(l); onClose() }

  return createPortal(
    <div onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
         style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.35)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div role="dialog" aria-modal="true" aria-label="Filter by coordinates"
           style={{ width: 'min(600px, calc(100vw - 32px))', maxHeight: 'calc(100vh - 64px)',
                    overflowY: 'auto', boxSizing: 'border-box', padding: 16,
                    background: 'var(--bg)', color: 'var(--fg, inherit)', borderRadius: 8,
                    border: '1px solid var(--border-strong)',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.25)', font: '13px system-ui' }}>
        <CardTitle>Filter by coordinates</CardTitle>

        <Label>Saved lists</Label>
        {lists === null && !err && <p style={{ opacity: 0.6, margin: '0 0 12px' }}>Loading…</p>}
        {lists?.length === 0 && (
          <p style={{ opacity: 0.6, margin: '0 0 12px' }}>No saved lists yet.</p>
        )}
        {lists && lists.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14,
                        maxHeight: 260, overflowY: 'auto' }}>
            {lists.map((l) => (
              <SavedList key={l.id} list={l} applied={l.id === current?.id}
                         catalog={catalog} onShow={() => apply(l)}
                         onChanged={(next) => {
                           reload()
                           if (l.id === current?.id) onApply(next)
                         }}
                         onError={setErr} />
            ))}
          </div>
        )}

        <NewList catalog={catalog} onSaved={apply} onError={setErr} />

        {err && <p style={{ color: TONE.red, margin: '10px 0 0' }}>{err}</p>}

        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          {current && (
            <button onClick={() => apply(null)} style={button}>Stop filtering by coordinates</button>
          )}
          <button onClick={onClose} style={{ ...button, marginLeft: 'auto' }}>Close</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

/** One saved list: show, rename or delete it. */
function SavedList({ list, applied, catalog, onShow, onChanged, onError }: {
  list: CoordListSummary
  applied: boolean
  catalog: CatalogFilter
  onShow: () => void
  /** After a rename (the new summary) or a delete (null). */
  onChanged: (next: CoordListSummary | null) => void
  onError: (msg: string | null) => void
}) {
  const [mode, setMode] = useState<'view' | 'rename' | 'delete'>('view')
  const [name, setName] = useState(list.name)
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<CoordListSummary | null>) => {
    setBusy(true); onError(null)
    try { onChanged(await action()); setMode('view') }
    catch (e) { onError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }
  const hidden = catalog === 'all' ? 0 : list.n_sources - (list.by_catalog[catalog] ?? 0)

  return (
    <div style={{ padding: '6px 8px', borderRadius: 6,
                  border: `1px solid ${applied ? `${TONE.blue}88` : 'var(--border)'}`,
                  background: applied ? `${TONE.blue}14` : 'var(--bg-soft)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {mode === 'rename' ? (
          <input value={name} autoFocus onChange={(e) => setName(e.target.value)}
                 onKeyDown={(e) => {
                   if (e.key === 'Enter') run(() => renameCoordList(list.id, name))
                 }}
                 style={{ ...field, padding: '2px 6px' }} />
        ) : (
          <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis',
                         whiteSpace: 'nowrap' }} title={list.name}>
            {list.name}
          </span>
        )}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 4, flexShrink: 0 }}>
          {mode === 'view' && <>
            <button onClick={onShow} disabled={applied} style={{ ...small, fontWeight: 600,
                                                                 opacity: applied ? 0.5 : 1 }}>
              {applied ? 'Showing' : 'Show'}
            </button>
            <button onClick={() => setMode('rename')} style={small}>Rename</button>
            <button onClick={() => setMode('delete')} style={small}>Delete</button>
          </>}
          {mode === 'rename' && <>
            <button disabled={busy || !name.trim()} style={small}
                    onClick={() => run(() => renameCoordList(list.id, name))}>Save</button>
            <button onClick={() => { setName(list.name); setMode('view') }} style={small}>Cancel</button>
          </>}
          {mode === 'delete' && <>
            <span style={{ color: TONE.red, alignSelf: 'center', fontSize: 12 }}>Delete for everyone?</span>
            <button disabled={busy} style={{ ...small, color: TONE.red, borderColor: `${TONE.red}88` }}
                    onClick={() => run(() => deleteCoordList(list.id).then(() => null))}>
              Delete
            </button>
            <button onClick={() => setMode('view')} style={small}>Keep</button>
          </>}
        </span>
      </div>
      <div style={{ opacity: 0.7, fontSize: 12, marginTop: 2 }}>
        {plural(list.n_sources, 'source')} from {plural(list.n_coords, 'position')}
        {' · '}{list.radius_arcsec}″{list.created ? ` · ${list.created.slice(0, 10)}` : ''}
        {(list.columns ?? []).length > 0 && ` · sort by ${list.columns.join(', ')}`}
        {hidden > 0 && <span style={{ color: TONE.orange }}> · {hidden} outside {catalog}</span>}
      </div>
    </div>
  )
}

/** Choose a file, preview what it matches, then save it under a name. */
function NewList({ catalog, onSaved, onError }: {
  catalog: CatalogFilter
  onSaved: (l: CoordListSummary) => void
  onError: (msg: string | null) => void
}) {
  const [text, setText] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [radius, setRadius] = useState(DEFAULT_RADIUS_ARCSEC)
  const [preview, setPreview] = useState<CoordListPreview | null>(null)
  const [busy, setBusy] = useState(false)

  const pick = async (file: File | undefined) => {
    setPreview(null); onError(null)
    if (!file) { setText(null); return }
    setText(await file.text())
    setName(file.name.replace(/\.(txt|csv|dat)$/i, ''))
  }
  const act = async (action: () => Promise<void>) => {
    setBusy(true); onError(null)
    try { await action() }
    catch (e) { onError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }
  // A preview belongs to the radius it was made with
  useEffect(() => setPreview(null), [radius])

  const hidden = preview && catalog !== 'all'
    ? preview.n_sources - (preview.by_catalog[catalog] ?? 0) : 0

  return (
    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
      <Label>New list</Label>
      <p style={{ margin: '0 0 8px', opacity: 0.75, lineHeight: 1.45 }}>
        A text file with one position per line: RA and Dec in decimal degrees
        (<code>150.1192 2.2058</code>) or sexagesimal (<code>10:00:28.6 +02:12:21</code>),
        optionally after a name. Commas or spaces separate them; blank lines and anything
        after <code>#</code> are ignored. Numbers after the position become sortable
        columns when a header line names them: <code>name ra dec stetson_J rho</code>.
      </p>
      <input type="file" accept=".txt,.csv,.dat,text/plain"
             onChange={(e) => act(() => pick(e.target.files?.[0]))}
             style={{ ...field, padding: 4 }} />

      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, marginTop: 10 }}>
        <div style={{ flex: 1 }}>
          <Label>Name</Label>
          <input value={name} onChange={(e) => setName(e.target.value)}
                 placeholder="e.g. Oct follow-up candidates" maxLength={120} style={field} />
        </div>
        <div style={{ width: 120 }}>
          <Label>Radius (″)</Label>
          <input type="number" min={0.1} max={60} step={0.5} value={radius}
                 onChange={(e) => setRadius(Number(e.target.value))} style={field} />
        </div>
        <button disabled={!text || busy || !(radius > 0)}
                onClick={() => act(async () => setPreview(await previewCoordList(text!, radius)))}
                style={{ ...button, opacity: !text || busy ? 0.5 : 1 }}>
          {busy && !preview ? 'Matching…' : 'Match'}
        </button>
      </div>

      {preview && (
        <div style={{ marginTop: 12, padding: 10, borderRadius: 6,
                      border: '1px solid var(--border)', background: 'var(--bg-soft)' }}>
          <div>
            {preview.n_matched_coords.toLocaleString()} of {plural(preview.n_coords, 'position')} matched
            within {preview.radius_arcsec}″ → <b>{plural(preview.n_sources, 'source')}</b>
          </div>
          {preview.n_sources > 0 && (
            <div style={{ opacity: 0.75, marginTop: 2 }}>
              {Object.entries(preview.by_catalog).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </div>
          )}
          {(preview.columns ?? []).length > 0 && (
            <div style={{ opacity: 0.75, marginTop: 2 }}>
              Sortable columns: {preview.columns.join(', ')}
            </div>
          )}
          {hidden > 0 && (
            <div style={{ color: TONE.orange, marginTop: 6 }}>
              {hidden} of these are outside the <b>{catalog}</b> tab; choose <b>everything</b> to see them all.
            </div>
          )}
          {preview.n_unmatched > 0 && (
            <details style={{ marginTop: 6 }}>
              <summary style={{ cursor: 'pointer' }}>{plural(preview.n_unmatched, 'unmatched position')}</summary>
              <pre style={{ margin: '4px 0 0', maxHeight: 140, overflowY: 'auto', fontSize: 11 }}>
                {preview.unmatched.join('\n')}
                {preview.n_unmatched > preview.unmatched.length && '\n…'}
              </pre>
            </details>
          )}
          {preview.n_unparsed > 0 && (
            <details style={{ marginTop: 6 }}>
              <summary style={{ cursor: 'pointer', color: TONE.orange }}>
                {plural(preview.n_unparsed, 'line')} not read as a position
              </summary>
              <pre style={{ margin: '4px 0 0', maxHeight: 140, overflowY: 'auto', fontSize: 11 }}>
                {preview.unparsed.join('\n')}
                {preview.n_unparsed > preview.unparsed.length && '\n…'}
              </pre>
            </details>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
            <span style={{ opacity: 0.65, fontSize: 12 }}>
              Everyone will see it, and anyone can rename or delete it.
            </span>
            <button disabled={busy || !name.trim()}
                    onClick={() => act(async () => onSaved(await createCoordList(text!, name, radius)))}
                    style={{ ...button, marginLeft: 'auto', fontWeight: 600,
                             opacity: busy || !name.trim() ? 0.5 : 1 }}>
              Save & show
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
