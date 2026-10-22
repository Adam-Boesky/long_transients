/** Small shared pieces: badges, cards, the sidebar and the bottom nav bar.
 *
 * Deliberately plain elements rather than a component library. The Streamlit
 * app's look comes from a handful of primitives (`st.badge`, bordered
 * containers, a horizontal nav row), and matching that needs styling, not a
 * dependency.
 */
import { useEffect, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'

export const TONE: Record<string, string> = {
  blue: '#3b82f6', green: '#16a34a', violet: '#8b5cf6',
  orange: '#ea580c', red: '#dc2626', yellow: '#ca8a04', gray: '#888888',
}

export function Badge({ children, tone = 'gray' }:
                      { children: ReactNode; tone?: keyof typeof TONE | string }) {
  const hex = TONE[tone] ?? TONE.gray
  return (
    <span style={{ padding: '2px 8px', borderRadius: 999, fontSize: 12,
                   // 8-digit hex is an alpha channel: a light wash, so the
                   // label stays readable in either theme.
                   background: `${hex}22`, border: `1px solid ${hex}66`,
                   color: 'inherit', whiteSpace: 'nowrap' }}>
      {children}
    </span>
  )
}

export function Card({ children, pad = 12, style }:
                     { children: ReactNode; pad?: number; style?: CSSProperties }) {
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 7,
                  padding: pad, ...style }}>
      {children}
    </div>
  )
}

/** The same panel on every page, with one section that differs.
 *
 * Classify files verdicts, so its label section is a set of buttons that
 * assign one. Table reads them back, so the same slot holds the label-state
 * cut instead. Everything above -- scope, then filters -- is shared state and
 * means the same thing in both places, which is the point: narrowing the
 * sample on one page and switching to the other shows that sample.
 *
 * Which mode you get is decided by which props arrive, not by a flag: pass
 * `labelStates` and you get the filter, pass `labels` and you get the
 * buttons.
 */
export const field: CSSProperties = {
  width: '100%', padding: '5px 7px', font: 'inherit', borderRadius: 5,
  border: '1px solid var(--border-strong)', background: 'transparent',
  color: 'inherit', boxSizing: 'border-box',
}

/** Sidebar section headings are capitalised; control captions are not. */
/** A boxed group's title. Deliberately a step above `Label`, which titles
 *  the controls *inside* it: the two were 11px and 12px, so every group read
 *  as smaller than its own children. */
export function CardTitle({ children }: { children: ReactNode }) {
  return (
    <div style={{ font: '600 12px system-ui', letterSpacing: '0.08em',
                  textTransform: 'uppercase', opacity: 0.8,
                  marginBottom: 10 }}>
      {children}
    </div>
  )
}

export function Label({ children, caps = true }:
               { children: ReactNode; caps?: boolean }) {
  return (
    <div style={{ font: '600 11px system-ui', opacity: 0.55,
                  marginBottom: caps ? 6 : 5,
                  textTransform: caps ? 'uppercase' : 'none',
                  letterSpacing: caps ? 0.5 : 0 }}>
      {children}
    </div>
  )
}

/** What you can order the candidate queue by.
 *
 * A short list rather than all 318 columns: this sits in the navigation bar,
 * where the question is "which end of the sample do I want to start from",
 * not "which column am I interested in". Anything not here is still reachable
 * -- filter on it in the sidebar, and the queue narrows to what you meant.
 *
 * Each entry is the catalog column the server orders by, so adding one is a
 * line here and nothing else.
 */
export const SORT_OPTIONS: readonly { key: string; label: string;
                                      desc: boolean }[] = [
  // The triage order, and the default: the largest change between surveys
  // first, which is what the sample was built to find.
  { key: 'mean_abs_dmag', label: 'Mean |dmag|', desc: true },
  { key: 'wise_w1w2', label: 'W1-W2', desc: true },
  { key: 'n_epochs', label: 'Epochs', desc: true },
  { key: 'ra', label: 'RA', desc: false },
  { key: 'dec', label: 'Dec', desc: false },
  { key: 'coord_slug', label: 'Filename', desc: false },
  // The old default. Effectively arbitrary order, which is the point: it is
  // the only option here that does not bias which sources you see first.
  { key: 'uuid', label: 'Unsorted', desc: false },
]

export interface NavBarProps {
  position: number; total: number; labelled: number
  onPrev: () => void; onNext: () => void
  onJump: (index: number) => void
  onNextUnlabelled: () => void
  /** Which column the queue is ordered by, and which way. */
  sort: string; desc: boolean
  onSort: (key: string, desc: boolean) => void
}

export function NavBar(p: NavBarProps) {
  // The jump box holds a draft and only commits on release -- blur, Enter,
  // or letting go of the spinner. Committing on every keystroke meant that
  // typing "12" navigated to source 1 on the way, and each of those loads
  // pulled a light curve, spectra and six stamps nobody asked for.
  const [draft, setDraft] = useState(String(p.position + 1))

  // Follow navigation that happened elsewhere (Prev/Next, arrow keys) so the
  // box does not sit on a stale number.
  useEffect(() => { setDraft(String(p.position + 1)) }, [p.position])

  const commit = () => {
    const n = Number(draft)
    if (!draft.trim() || !Number.isFinite(n)) {
      setDraft(String(p.position + 1))
      return
    }
    // `p.total` is 0 while a filter change is in flight; clamping to it then
    // would rewrite the typed number as 1.
    const clamped = p.total > 0
      ? Math.max(1, Math.min(Math.round(n), p.total))
      : Math.max(1, Math.round(n))
    setDraft(String(clamped))
    if (clamped - 1 !== p.position) p.onJump(clamped - 1)
  }

  return (
    <Card pad={8} style={{ display: 'flex', gap: 10, alignItems: 'center',
                           flexWrap: 'wrap' }}>
      <button onClick={p.onPrev} disabled={p.position <= 0}
              style={btn(p.position <= 0)}>‹ Prev</button>
      <input
        type="number" min={1} max={p.total} value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        // The spinner arrows are a mouse gesture, so they commit on release
        // rather than on every increment.
        onMouseUp={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit()
          else if (e.key === 'Escape') setDraft(String(p.position + 1))
        }}
        style={{ ...field, width: 80 }} />
      <span style={{ opacity: 0.7 }}>of {p.total.toLocaleString()}</span>
      <button onClick={p.onNext} disabled={p.position >= p.total - 1}
              style={btn(p.position >= p.total - 1)}>Next ›</button>
      <button onClick={p.onNextUnlabelled} style={btn(false)}>Next unbucketed »</button>

      {/* Which end of the sample to start from. Changing it renumbers the
          queue, so it sits with the controls that move through it rather
          than down in the sidebar with the cuts that decide its contents. */}
      <label style={{ display: 'flex', gap: 5, alignItems: 'center',
                      opacity: 0.8 }}>
        <span>sort</span>
        <select value={p.sort} style={{ ...field, width: 'auto' }}
                onChange={(e) => {
                  const next = SORT_OPTIONS.find((o) => o.key === e.target.value)
                  // Each column carries the direction that makes it useful --
                  // brightest change first, lowest RA first -- so picking one
                  // does not also mean picking a direction.
                  if (next) p.onSort(next.key, next.desc)
                }}>
          {SORT_OPTIONS.map((o) => (
            <option key={o.key} value={o.key}>{o.label}</option>
          ))}
        </select>
        <button onClick={() => p.onSort(p.sort, !p.desc)}
                title={p.desc ? 'Largest first' : 'Smallest first'}
                disabled={p.sort === 'uuid'}
                style={{ ...btn(p.sort === 'uuid'), padding: '5px 8px' }}>
          {p.desc ? '↓' : '↑'}
        </button>
      </label>
      <span style={{ marginLeft: 'auto', opacity: 0.7 }}>
        {p.labelled} / {p.total} bucketed
      </span>
    </Card>
  )
}

function btn(disabled: boolean): CSSProperties {
  return {
    font: 'inherit', padding: '5px 10px', borderRadius: 5, color: 'inherit',
    border: '1px solid var(--border-strong)', background: 'transparent',
    cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.45 : 1,
  }
}


// ── form controls ─────────────────────────────────────────────────────────
//
// Stand-ins for the Streamlit widgets `lib/filters.py` uses -- there is no
// DOM equivalent of `st.segmented_control`, `st.pills` or a two-handle
// `st.slider`. Everything underneath is a plain element: `Expander` is a
// `<details>`, `RangeSlider` is a pair of `<input type="range">`.

function chip(active: boolean): CSSProperties {
  return {
    ...field, padding: '4px 11px', cursor: 'pointer',
    // Selected reads as a dark neutral fill rather than the link blue, which
    // made every active control on the page shout.
    background: active ? 'var(--text)' : 'var(--bg)',
    borderColor: active ? 'var(--text)' : 'var(--border-strong)',
    color: active ? 'var(--bg)' : 'inherit',
    fontWeight: active ? 600 : 400,
  }
}

/** Single-select, always exactly one option chosen. */
export function SegmentedControl({ label, options, value, onChange, format }: {
  label?: string
  options: readonly string[]
  value: string
  onChange: (v: string) => void
  format?: (v: string) => string
}) {
  return (
    <div>
      {label && <Label caps={false}>{label}</Label>}
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        {options.map((o) => (
          <button key={o} type="button" onClick={() => onChange(o)}
                  aria-pressed={o === value} style={chip(o === value)}>
            {format ? format(o) : o}
          </button>
        ))}
      </div>
    </div>
  )
}




/** Tri-state include/exclude, one row per value.
 *
 * Tri-state rather than two checkboxes: include and exclude are mutually
 * exclusive for the same bucket, and a pair of boxes would let you ask for
 * both. Clicking cycles neutral -> include -> exclude -> neutral.
 *
 * Any-author throughout, matching the server: one person filing a source is
 * enough for it to count.
 */
export function TriChecklist({ title, options, value,
                              tone = () => undefined,
                              format = (v) => v, onChange }: {
  title: string
  options: string[]
  value: Record<string, 'in' | 'out'>
  /** Stripe colour per value, or undefined for no stripe. A resolver rather
   *  than a map so buckets can fall back to `labelTone` exactly as the label
   *  list does, while follow-up stays stripe-free. */
  tone?: (value: string) => string | undefined
  format?: (value: string) => string
  onChange: (next: Record<string, 'in' | 'out'>) => void
}) {
  const cycle = (b: string) => {
    const next = { ...value }
    if (next[b] === 'in') next[b] = 'out'
    else if (next[b] === 'out') delete next[b]
    else next[b] = 'in'
    onChange(next)
  }
  return (
    <div style={{ marginTop: 8 }}>
      <Label>{title}</Label>
      {options.map((b) => {
        const state = value[b]
        const stripe = tone(b)
        return (
          <button key={b} onClick={() => cycle(b)}
                  title={state === 'in' ? 'included -- click to exclude'
                       : state === 'out' ? 'excluded -- click to clear'
                       : 'click to include'}
                  style={{ display: 'flex', alignItems: 'center', gap: 7,
                           width: '100%', marginBottom: 4, padding: '5px 8px',
                           border: '1px solid var(--border)', borderRadius: 5,
                           background: state ? 'var(--accent-bg)' : 'transparent',
                           font: 'inherit', color: 'inherit', cursor: 'pointer',
                           textAlign: 'left' }}>
            {stripe && <span style={{ width: 7, height: 14, flex: 'none',
                                      borderRadius: 2, background: stripe }} />}
            <span style={{ flex: 1,
                           textDecoration: state === 'out' ? 'line-through' : 'none',
                           opacity: state === 'out' ? 0.6 : 1 }}>{format(b)}</span>
            <span style={{ fontSize: 12, width: 12, textAlign: 'center',
                           opacity: state ? 1 : 0.25 }}>
              {state === 'in' ? '✓' : state === 'out' ? '✕' : '·'}
            </span>
          </button>
        )
      })}
    </div>
  )
}
