/** The eROSITA DR2 counterpart, if any, from the `erosita_*` catalog columns.
 *
 * Matched within a fixed 10" (`Source_Analysis.catalogs.erosita`). Only the
 * eROSITA position error is shown: ours is ~0.1", far below it. Outside the
 * eROSITA-DE half of the sky there is nothing to say beyond "not covered".
 */
import type { CSSProperties } from 'react'
import { MODEBAR_GUTTER } from '../charts/plotly'
import { isTrue } from '../lib/format'

export interface XRayPanelProps {
  catalog: Record<string, unknown> | null
  ra: number | null
  dec: number | null
}

export function XRayPanel({ catalog, ra, dec }: XRayPanelProps) {
  // A release built before the eROSITA columns existed says nothing either way.
  if (!catalog || catalog.erosita_matched === undefined) return null
  const matched = isTrue(catalog.erosita_matched)
  const covered = isTrue(catalog.erosita_covered)

  return (
    // Margins line the box up with the histogram's plot area above it.
    <div style={{ font: '16px system-ui', margin: `0 ${MODEBAR_GUTTER}px 0 52px`, padding: '12px 16px',
                  border: '1px solid var(--border-strong)',
                  borderRadius: 7, background: 'var(--bg-soft)' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 8 }}>
        <span style={{ font: '700 18px system-ui' }}>X-ray</span>
        <span style={{ fontSize: 13, opacity: 0.6 }}>eROSITA DR2 · 0.2–2.3 keV</span>
      </div>
      {!matched ? (
        <div style={{ opacity: 0.6 }}>
          {covered ? 'No eROSITA source within 10″.' : 'Not covered by eROSITA-DE.'}
        </div>
      ) : (
        <table style={{ borderCollapse: 'collapse', fontVariantNumeric: 'tabular-nums' }}>
          <tbody>
            <Row label="Flux" value={flux(num(catalog.erosita_flux), num(catalog.erosita_flux_err))} />
            <Row label="Offset" value={arcsec(num(catalog.erosita_sep_arcsec))} />
            <Row label="eROSITA" value={coords(num(catalog.erosita_ra), num(catalog.erosita_dec))} />
            <Row label="Pos. error" value={posErr(num(catalog.erosita_pos_err))} />
            <Row label="Ours" value={coords(ra, dec)} />
          </tbody>
        </table>
      )}
    </div>
  )
}

const LABEL: CSSProperties = { padding: '3px 18px 3px 0', opacity: 0.6, whiteSpace: 'nowrap' }

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <td style={LABEL}>{label}</td>
      <td style={{ padding: '3px 0', whiteSpace: 'pre', fontWeight: 500 }}>{value}</td>
    </tr>
  )
}

/** `7.1 ± 0.6 × 10⁻¹³ erg s⁻¹ cm⁻²`, both on the value's power of ten. */
function flux(f: number | null, err: number | null): string {
  if (f === null) return '—'
  const exp = Math.floor(Math.log10(Math.abs(f)))
  const scale = 10 ** exp
  const body = err !== null ? `${(f / scale).toFixed(1)} ± ${(err / scale).toFixed(1)}`
                            : (f / scale).toFixed(1)
  return `${body} × 10${superscript(exp)} erg s⁻¹ cm⁻²`
}

function arcsec(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(1)}″`
}

/** POS_ERR is 1-sigma per coordinate (Ramos-Ceja et al. 2026, Sect. 4.1). */
function posErr(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(1)}″ (1σ, RA and Dec)`
}

function coords(ra: number | null, dec: number | null): string {
  if (ra === null || dec === null) return '—'
  return `(${ra.toFixed(5)}, ${dec >= 0 ? '+' : ''}${dec.toFixed(5)})`
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

const SUPERSCRIPT: Record<string, string> = {
  '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹',
}
function superscript(n: number): string {
  return String(n).split('').map((c) => SUPERSCRIPT[c] ?? c).join('')
}
