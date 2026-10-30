/** The left-hand panel: the scope and cuts both pages share, plus the
 * Classify page's filing controls.
 *
 * Split out of `ui.tsx`, which is now only presentational primitives. This
 * file reads `useFilters()` directly: every cut it draws is provider state,
 * and forwarding each one through a prop meant both pages repeated the same
 * fifteen-line call and any new filter had to be threaded through three
 * files to reach the control that sets it.
 */
import { useState } from 'react'
import type { ReactNode } from 'react'
import { FOLLOWUP_STATUSES } from '../api/types'
import { CoordListDialog } from './CoordListDialog'
import type { FollowupStatus } from '../api/types'
import {
  DUPLICATE_MODES, LABEL_STATES, UNIFIED_CLASSES, W1W2_MODES, useFilters,
} from '../lib/filterState'
import type { CatalogFilter, DuplicateMode, LabelState, W1W2Mode } from '../lib/filterState'
import {
  CardTitle, Label, SegmentedControl, TONE, TriChecklist, field,
} from './ui'

/** Which of the pipeline's sub-catalogues to work through.
 *
 * These are the separate enriched tables the filter stage emits, tagged in
 * the database as `catalog_tier`: `0_enriched` is the sources detected in
 * both surveys, `1_enriched` the ZTF-only ones, and `1_wide_association` the
 * ZTF-only detections with a PS1 source 1-3" away. As in the Streamlit app you
 * work through one at a time, because they are different populations and
 * mixing them makes the run hard to reason about.
 */

// A partition, matching `api.catalog_filters.CATALOG_TIERS`: the views are
// mutually exclusive, which is what a segmented control promises.
const CATALOGS: readonly CatalogFilter[] = ['in_ztf', 'in_ztf_wide', 'in_both', 'all']

const CATALOG_LABEL: Record<CatalogFilter, string> = {
  in_ztf: 'in_ztf', in_ztf_wide: 'in_ztf_wide', in_both: 'in_both', all: 'everything',
}

/** "nobody has set one", in both checklists. Matches
 *  `api.annotation_filters.NONE`. */
export const NONE_VALUE = 'none'

export interface SidebarProps {
  /** Which page this is.
   *
   *  Table reads verdicts back and never files one, so it gets the
   *  label-state cut; Classify files one and gets the `This source` card.
   *  This used to be inferred from whether `labelStates` was passed, which
   *  made a constant array double as a mode switch.
   */
  mode: 'classify' | 'table'

  // ── filing a verdict (Classify only) ────────────────────────────────────
  /** This author's label for the source in front of you. */
  currentLabel?: string
  onLabel?: (label: string) => void
  /** Who *else* filed this source under each label, by label. */
  othersByLabel?: Record<string, string[]>
  /** Every label, in the order it was dragged into. */
  onReorder?: (labels: string[]) => void
  onLabelColor?: (label: string, color: string) => void
  /** Deleting can fail (a label with sources), and the page owns the place
   *  that failure is shown, so it stays a prop. */
  onDeleteLabel?: (label: string) => void
  /** Follow-up observing state; null is "nothing". */
  currentStatus?: FollowupStatus | null
  onStatus?: (status: FollowupStatus | null) => void
  /** Rendered directly under the label buttons: the comment box, so the
   *  verdict and its reasoning are filed from the same place. */
  belowLabels?: ReactNode
  /** The column-filter builder, under the fixed filters. Passed in rather
   *  than imported so this file does not depend on the filter editor. */
  columnFilters?: ReactNode
}

/** Sidebar reading order matches the Streamlit app: scope, then buckets,
 *  then the rest of the filters. */
export function Sidebar(p: SidebarProps) {
  // Every cut below is provider state, read here rather than forwarded: the
  // two pages were passing the same fifteen props through to get it.
  const {
    search, setSearch, catalog, setCatalog,
    bucketFilter, setBucketFilter, statusFilter, setStatusFilter,
    classFilter, setClassFilter, matchFilter, setMatchFilter, matchCatalogs,
    labelState, setLabelState,
    w1w2, setW1w2, w1w2KeepMissing, setW1w2KeepMissing, clearFilters,
    duplicates, setDuplicates, hasGroups, coordList, coordListId, setCoordList,
    labels, labelColors, labelCounts, reloadLabels, addBucket,
  } = useFilters()
  const [coordDialog, setCoordDialog] = useState(false)
  // Classify-only handlers, defaulted so the dozen uses below need no guard.
  const othersByLabel = p.othersByLabel ?? {}
  const onLabel = p.onLabel ?? (() => {})
  const onReorder = p.onReorder ?? (() => {})
  const onLabelColor = p.onLabelColor ?? (() => {})
  const classify = p.mode === 'classify'
  // Drag state: which label is in hand, and the gap it would drop into
  // (0 = above the first, n = below the last).
  const [dragging, setDragging] = useState<string | null>(null)
  const [gap, setGap] = useState<number | null>(null)
  // The label whose colour popover is open (from its ⋯ button).
  const [colorMenu, setColorMenu] = useState<string | null>(null)
  const drop = () => {
    if (dragging !== null && gap !== null) {
      const from = labels.indexOf(dragging)
      const rest = labels.filter((l) => l !== dragging)
      // Removing the dragged label shifts every gap below it up by one.
      const at = gap > from ? gap - 1 : gap
      rest.splice(at, 0, dragging)
      if (rest.some((l, i) => l !== labels[i])) onReorder(rest)
    }
    setDragging(null); setGap(null)
  }

  // Ordered here rather than inline so `This source` can sit above the
  // filters: it is what you act on, and the filters are the standing scope.
  // `labelStates` is what tells the two pages apart -- Table reads verdicts
  // back and never files one, so it gets the label-state cut instead.
  // Identical on Classify and Table: both narrow the same sample, and the
  // provider holds the state so switching tabs keeps it. `min epochs` and
  // `only unlabelled` used to live here -- the first is a column filter on
  // `n_epochs`, the second is the `(unlabelled)` row. Boxed like the card
  // above it, so the two groups read as what they are: this one narrows the
  // sample, that one edits one source.
  // "Anything set" across the whole card, so one `clear` can mean all of it.
  const anySet = Object.keys(bucketFilter).length > 0
              || Object.keys(statusFilter).length > 0
              || Object.keys(classFilter).length > 0
              || Object.keys(matchFilter).length > 0
              || w1w2 !== W1W2_MODES[0]
              || duplicates !== DUPLICATE_MODES[0]
  const filtersSection = (
      <section style={{ border: '1px solid var(--border-strong)',
                        borderRadius: 7, padding: '10px 10px 12px',
                        background: 'var(--bg-soft)' }}>
        <div style={{ display: 'flex', alignItems: 'baseline' }}>
          <CardTitle>Filters</CardTitle>
          {anySet && (
            <button onClick={clearFilters}
                    style={{ marginLeft: 'auto', border: 0, background: 'none',
                             font: 'inherit', fontSize: 11, opacity: 0.7,
                             cursor: 'pointer', color: 'inherit',
                             paddingBottom: 10 }}>
              clear
            </button>
          )}
        </div>
        <TriChecklist title="Buckets"
                      options={[...labels, NONE_VALUE]}
                      value={bucketFilter}
                      // Same stripe the label list shows, chosen colour or
                      // generated default. `(no bucket)` is not a label, so
                      // it gets none.
                      tone={(v) => (v === NONE_VALUE ? undefined
                                    : labelColors[v] ?? labelTone(v))}
                      format={(v) => (v === NONE_VALUE ? '(no bucket)' : v)}
                      onChange={setBucketFilter} />
        <AddLabel onAdd={addBucket} what="bucket" />
        <TriChecklist title="Follow-up Status"
                      options={[...FOLLOWUP_STATUSES, NONE_VALUE]}
                      value={statusFilter}
                      format={(v) => (v === NONE_VALUE ? '(not set)' : v)}
                      onChange={setStatusFilter} />
        <TriChecklist title="Class" options={[...UNIFIED_CLASSES]}
                      value={classFilter}
                      onChange={setClassFilter} />
        {matchCatalogs.length > 0 && (
          <TriChecklist title="Matched" options={matchCatalogs}
                        value={matchFilter}
                        onChange={setMatchFilter} />
        )}
        {hasGroups && (
          <div style={{ marginTop: 10 }}>
            {/* The same object found again in an overlapping field. */}
            <Label>Duplicates</Label>
            <SegmentedControl options={[...DUPLICATE_MODES]} value={duplicates}
                              onChange={(v) => setDuplicates(v as DuplicateMode)} />
          </div>
        )}
        <div style={{ marginTop: 10 }}>
          <Label>W1 - W2</Label>
          <SegmentedControl options={[...W1W2_MODES]} value={w1w2}
                            onChange={(v) => setW1w2(v as W1W2Mode)} />
          {/* Only meaningful once a cut is chosen: with "Any" nothing is
              being excluded for want of a colour. */}
          {w1w2 !== W1W2_MODES[0] && (
            <label style={{ display: 'block', marginTop: 5, fontSize: 12,
                            opacity: 0.85 }}>
              <input type="checkbox" checked={w1w2KeepMissing}
                     onChange={(e) => setW1w2KeepMissing(e.target.checked)} />
              {' '}also keep sources with no WISE colour
            </label>
          )}
        </div>
        {p.columnFilters && (
          <div style={{ marginTop: 10 }}>{p.columnFilters}</div>
        )}
      </section>
  )

  // Everything here applies to the one source on screen, not to the window --
  // boxed and titled so it does not read as another filter, and placed above
  // the filters because it is what you act on. Classify only: the Table page
  // reads verdicts back rather than filing them.
  const sourceCard = !classify ? null : (
        <section style={{ border: '1px solid var(--border-strong)',
                          borderRadius: 7, padding: '10px 10px 12px',
                          background: 'var(--bg-soft)' }}>
          <CardTitle>This source</CardTitle>
          {p.onStatus && (
            <div style={{ marginBottom: 12 }}>
              <Label>Follow-up Status</Label>
              {/* No status is no button pressed; clicking the active one
                  clears it, the same toggle the bucket buttons use. */}
              <SegmentedControl
                options={FOLLOWUP_STATUSES}
                value={p.currentStatus ?? ''}
                onChange={(v) => p.onStatus?.(
                  v === p.currentStatus ? null : v as FollowupStatus)} />
            </div>
          )}
          {/* The drop target is the heading and the whole list, not each
              button: with per-button targets, letting go just above the first
              label (on the heading) or in the gap below the last one hit
              nothing, so there was no way to move a label to either end. */}
          <div
            onDragOver={(e) => {
              if (dragging === null) return
              e.preventDefault()
              const buttons = [...e.currentTarget.querySelectorAll<HTMLElement>(
                '[data-label-button]')]
              const below = buttons.findIndex((b) => {
                const r = b.getBoundingClientRect()
                return e.clientY < r.top + r.height / 2
              })
              setGap(below === -1 ? buttons.length : below)
            }}
            onDragLeave={(e) => {
              // Leaving for a child is not leaving the list.
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setGap(null)
            }}
            onDrop={(e) => { e.preventDefault(); drop() }}
          >
          <Label>Buckets</Label>
          {labels.map((l, i) => {
            const active = l === p.currentLabel
            const others = othersByLabel[l] ?? []
            const tone = labelColors[l] ?? labelTone(l)
            // A row rather than one button: the colour stripe is its own
            // control (a colour input cannot live inside a button), and the
            // rest of the row files the label.
            return (
              <div key={l} data-label-button
                   onMouseLeave={() => setColorMenu((m) => (m === l ? null : m))}
                   style={{
                     display: 'flex', marginBottom: 4, borderRadius: 5,
                     position: 'relative', border: '1px solid var(--border)',
                     background: active ? 'var(--accent-bg)' : 'transparent',
                     opacity: dragging === l ? 0.4 : 1,
                     // The drop line: an inset shadow on the edge the label
                     // would land against, so nothing reflows mid-drag.
                     boxShadow: gap === i ? 'inset 0 2px 0 var(--text)'
                              : gap === i + 1 && i === labels.length - 1
                                ? 'inset 0 -2px 0 var(--text)'
                                : undefined,
                   }}>
                <label title="Change colour"
                       style={{ position: 'relative', width: 7, flex: 'none',
                                background: tone, cursor: 'pointer',
                                borderRadius: '4px 0 0 4px' }}>
                  <input type="color" value={tone}
                         onChange={(e) => onLabelColor(l, e.target.value)}
                         style={{ position: 'absolute', inset: 0, opacity: 0,
                                  width: '100%', height: '100%', padding: 0,
                                  border: 0, cursor: 'pointer' }} />
                </label>
                <button onClick={() => onLabel(l)}
                        title={others.length ? `Also filed here by ${others.join(', ')}`
                                             : 'Drag to reorder'}
                        draggable
                        onDragStart={(e) => {
                          setDragging(l)
                          e.dataTransfer.effectAllowed = 'move'
                          // Firefox will not start a drag without some data.
                          e.dataTransfer.setData('text/plain', l)
                        }}
                        onDragEnd={() => { setDragging(null); setGap(null) }}
                        style={{
                          flex: 1, display: 'flex', alignItems: 'center',
                          textAlign: 'left', padding: '6px 9px', border: 'none',
                          background: 'transparent', cursor: 'pointer',
                          font: 'inherit', color: 'inherit',
                          fontWeight: active ? 600 : 400,
                        }}>
                  <span style={{ flex: 1 }}>{active ? '✓ ' : ''}{l}</span>
                  {others.length > 0 && (
                    <span style={{ fontSize: 11, fontWeight: 400, opacity: 0.7,
                                   padding: '0 6px', borderRadius: 999,
                                   border: '1px solid var(--border-strong)' }}>
                      {others.length}
                    </span>
                  )}
                </button>
                <button title="Change colour" aria-label="Change colour"
                        onClick={() => {
                          if (colorMenu !== l) reloadLabels?.()
                          setColorMenu((m) => (m === l ? null : l))
                        }}
                        style={{ flex: 'none', padding: '0 7px', border: 'none',
                                 background: 'transparent', cursor: 'pointer',
                                 font: 'inherit', color: 'inherit',
                                 opacity: colorMenu === l ? 1 : 0.5 }}>
                  ⋯
                </button>
                {colorMenu === l && dragging === null && (
                  // paddingLeft bridges the gap so the pointer can reach it.
                  <div style={{ position: 'absolute', left: '100%', top: -1,
                                paddingLeft: 6, zIndex: 20 }}>
                    <div style={{ display: 'flex', gap: 4, padding: 5,
                                  borderRadius: 6, background: 'var(--bg)',
                                  border: '1px solid var(--border-strong)',
                                  boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
                      {LABEL_SWATCHES.map((c) => (
                        <button key={c} title={c} aria-label={`Colour ${c}`}
                                onClick={() => onLabelColor(l, c)}
                                style={{ width: 18, height: 18, padding: 0,
                                         borderRadius: 4, cursor: 'pointer',
                                         background: c,
                                         border: c === tone ? '2px solid var(--text)'
                                                            : '1px solid var(--border)' }} />
                      ))}
                      {p.onDeleteLabel && (() => {
                        const n = labelCounts?.[l] ?? 0
                        return (
                          <button disabled={n > 0}
                                  title={n > 0 ? `${n} source${n === 1 ? '' : 's'} filed here; empty it to delete`
                                               : `Delete "${l}"`}
                                  aria-label={`Delete ${l}`}
                                  onClick={() => { setColorMenu(null); p.onDeleteLabel?.(l) }}
                                  style={{ width: 18, height: 18, padding: 0, marginLeft: 4,
                                           border: 'none', background: 'transparent',
                                           color: n > 0 ? 'inherit' : TONE.red,
                                           opacity: n > 0 ? 0.3 : 1,
                                           cursor: n > 0 ? 'not-allowed' : 'pointer',
                                           display: 'flex', alignItems: 'center',
                                           justifyContent: 'center' }}>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                                 stroke="currentColor" strokeWidth="2"
                                 strokeLinecap="round" strokeLinejoin="round">
                              <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" />
                            </svg>
                          </button>
                        )
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
          </div>
          <AddLabel onAdd={addBucket} />
          {p.belowLabels && <div style={{ marginTop: 12 }}>{p.belowLabels}</div>}
        </section>
  )

  // Boxed like the other two, and above the filters for the same reason the
  // source card is on Classify: it is the coarsest question on the page.
  const labelStateSection = classify ? null : (
        <section style={{ border: '1px solid var(--border-strong)',
                          borderRadius: 7, padding: '10px 10px 12px',
                          background: 'var(--bg-soft)' }}>
          <CardTitle>Bucket state</CardTitle>
          <SegmentedControl
            options={LABEL_STATES} value={labelState}
            // Display only: the API still takes the 'Labeled' spellings.
            format={(v) => v.replace('Unlabeled', 'Unbucketed').replace('Labeled', 'Bucketed')}
            onChange={(v) => setLabelState(v as LabelState)} />
        </section>
  )

  return (
    <aside style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <section>
        <input value={search} onChange={(e) => setSearch(e.target.value)}
               placeholder="uuid, file, or RA Dec" style={field} />
        {/* One option at a time, and which one is active is the most
            important thing about the current run -- so a segmented control
            rather than a dropdown that hides it. */}
        <div style={{ marginTop: 6 }}>
          <SegmentedControl options={CATALOGS} value={catalog}
                            onChange={(v) => setCatalog(v as CatalogFilter)}
                            format={(v) => CATALOG_LABEL[v as CatalogFilter]} />
        </div>
        {/* A list of positions narrows everything else, so it sits with the
            scope controls rather than among the filters. */}
        {coordListId ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6,
                        padding: '4px 8px', borderRadius: 5, fontSize: 12,
                        border: `1px solid ${TONE.blue}66`, background: `${TONE.blue}18` }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  title={coordList?.name}>
              Coordinates: {coordList ? `${coordList.name} (${coordList.n_sources.toLocaleString()})` : '…'}
            </span>
            <button onClick={() => setCoordDialog(true)}
                    style={{ marginLeft: 'auto', border: 0, background: 'none', font: 'inherit',
                             color: 'inherit', opacity: 0.75, cursor: 'pointer', padding: 0 }}>
              edit
            </button>
            <button onClick={() => setCoordList(null)} aria-label="Remove coordinate filter"
                    title="Remove coordinate filter"
                    style={{ border: 0, background: 'none', font: 'inherit', color: 'inherit',
                             opacity: 0.75, cursor: 'pointer', padding: 0 }}>
              ✕
            </button>
          </div>
        ) : (
          <button onClick={() => setCoordDialog(true)}
                  style={{ ...field, marginTop: 6, cursor: 'pointer', textAlign: 'left',
                           opacity: 0.85 }}>
            Filter by coordinates…
          </button>
        )}
        {coordDialog && (
          <CoordListDialog current={coordList} catalog={catalog} onApply={setCoordList}
                           onClose={() => setCoordDialog(false)} />
        )}
      </section>

      {sourceCard}
      {labelStateSection}
      {filtersSection}
    </aside>
  )
}

/** A label's default stripe colour, until someone picks one: from its name
 *  rather than its place in the list, so dragging a label does not repaint
 *  it or its neighbours. */
const LABEL_SWATCHES = [...Object.values(TONE), '#db2777', '#0d9488', '#0891b2']

function labelTone(name: string): string {
  const tones = Object.keys(TONE).filter((t) => t !== 'gray')
  let h = 0
  for (const c of name.toLowerCase()) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return TONE[tones[h % tones.length]]
}

/** "+ new bucket": a link until clicked, then a field. Enter adds, Escape or
 *  leaving an empty field puts the link back. */
function AddLabel({ onAdd, what = 'bucket' }:
                  { onAdd: (name: string) => Promise<void>; what?: string }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const close = () => { setOpen(false); setDraft(''); setError(null) }
  const submit = async () => {
    const name = draft.trim()
    if (!name) { close(); return }
    setBusy(true)
    try { await onAdd(name); close() }
    catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  if (!open) {
    return (
      <button onClick={() => setOpen(true)}
              style={{ font: 'inherit', fontSize: 12, cursor: 'pointer',
                       background: 'none', border: 'none', padding: '2px 0',
                       color: 'inherit', opacity: 0.6 }}>
        + new {what}
      </button>
    )
  }
  return (
    <div>
      <input autoFocus value={draft} maxLength={40} disabled={busy}
             placeholder={`${what} name, Enter to add`}
             onChange={(e) => { setDraft(e.target.value); setError(null) }}
             onKeyDown={(e) => {
               if (e.key === 'Enter' && !e.nativeEvent.isComposing) void submit()
               else if (e.key === 'Escape') close()
             }}
             onBlur={() => { if (!draft.trim()) close() }}
             style={field} />
      {error && <div style={{ color: 'crimson', fontSize: 12, marginTop: 3 }}>
        {error}
      </div>}
    </div>
  )
}

