/** Turning catalog cell values into something readable.
 *
 * The 301 columns are a mix of floats, ints, booleans, nulls and strings that
 * are really JSON lists -- AGN-DB records one class per contributing
 * sub-catalog and stores `["generic_agn", "generic_agn"]` in a single cell,
 * which `Source_Analysis.classify_app.lib.classes` also has to unpack.
 *
 * Large integers arrive as *strings* on purpose: values beyond 2^53 cannot
 * survive JSON parsing as numbers, so the serving tier emits them as text.
 * They must be displayed verbatim and never coerced back.
 */

/** SDSS and the extraction pipeline both use -999 style sentinels. */
export function isSentinel(v: unknown): boolean {
  return typeof v === 'number' && v <= -999
}

/** `["a", "b"]` stored in one cell -> `['a','b']`; anything else -> null. */
function asList(v: unknown): string[] | null {
  if (typeof v !== 'string') return null
  const t = v.trim()
  if (!t.startsWith('[') || !t.endsWith(']')) return null
  try {
    const parsed = JSON.parse(t)
    if (!Array.isArray(parsed)) return null
    return parsed.map((x) => String(x))
  } catch {
    return null
  }
}

export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  const list = asList(v)
  if (list) return list.join(', ')
  if (typeof v === 'string') return v === '' ? '—' : v
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '—'
    if (isSentinel(v)) return `${v} (sentinel)`
    if (Number.isInteger(v)) return v.toLocaleString()
    const abs = Math.abs(v)
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e7)) return v.toExponential(4)
    return v.toFixed(abs < 1 ? 5 : 4)
  }
  return String(v)
}

/** Columns whose value is missing or a sentinel, so the UI can hide them. */
export function isEmpty(v: unknown): boolean {
  return v === null || v === undefined || v === '' || isSentinel(v)
    || (typeof v === 'number' && !Number.isFinite(v))
}


/** A catalog flag, which reaches the page as a bool, number or string. */
export function isTrue(v: unknown): boolean {
  if (typeof v === 'string') return ['true', 't', '1', 'yes'].includes(v.trim().toLowerCase())
  return v === true || v === 1
}
