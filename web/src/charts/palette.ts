/** The plotting conventions the analysis pages already use.
 *
 * Taken from `Light_Curve.LC_COLOR_INFO` and `LC_MARKER_INFO`, so a chart
 * here reads the same way as the matplotlib figures: **colour is the band,
 * marker is the survey**. Two encodings rather than one, which is what lets a
 * single panel carry six surveys without becoming unreadable.
 */

/** Matplotlib's named colours, resolved to hex. */
const BAND_COLOR: Record<string, string> = {
  u: '#000000',   // black
  g: '#228B22',   // forestgreen
  r: '#B22222',   // firebrick
  i: '#9932CC',   // darkorchid
  z: '#F08080',   // lightcoral
  y: '#FFD700',   // gold
  b: '#4682B4',   // steelblue
}

/** Band names that are not a bare lowercase letter.
 *
 *  PTF reports Mould R as `R` (source_db/lightcurves/irsa.py, PTF_FID_TO_BAND),
 *  which is 3.6M epochs across 105k sources -- over half the catalogue. Without
 *  this it missed `BAND_COLOR` and fell through to the grey fallback, while
 *  `Light_Curve.LC_MAG_INFO` maps `R_mag` onto the `r` slot and draws it
 *  firebrick. Aliased rather than lowercased so the legend still says `ptf R`:
 *  Mould R is not SDSS r, it only shares the colour.
 */
const BAND_ALIAS: Record<string, string> = { R: 'r' }

/** WISE gets its own panel and its own two colours, as in `plot_wise_lc`. */
const WISE_BAND_COLOR: Record<string, string> = {
  w1: '#8B4513',  // saddlebrown
  w2: '#F4A460',  // sandybrown
  // Both are drawn: the chart loops over whatever bands the payload holds,
  // and WISE ships w1-w4 (2.45M and 2.47M epochs, ~123k sources each).
  // `plot_wise_lc` hard-codes w1/w2, so the figure and the page differ here.
  w3: '#6B8E23',  // olivedrab
  w4: '#708090',  // slategray
}

/** Matplotlib marker -> the nearest Plotly symbol. */
const SURVEY_SYMBOL: Record<string, string> = {
  ptf: 'cross',              // 'P'  filled plus
  ztf: 'triangle-left',      // '<'
  sdss: 'diamond-tall',      // 'd'  thin diamond
  panstarrs: 'y-left',       // '3'  tri_left
  gaia: 'x',                 // 'X'
  custom: 'circle',          // '.'
  ztforce: 'square',         // 's'
  zubercal: 'diamond',       // 'D'
  wise: 'circle',
  neowise: 'square',
}

/** Surveys drawn on the main optical panel. WISE and NEOWISE are excluded:
 *  `plot_lc` leaves them to `plot_wise_lc`, on their own magnitude scale. */
export const OPTICAL_SURVEYS = [
  'ztf', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom', 'ztforce', 'zubercal',
]

export const INFRARED_SURVEYS = ['wise', 'neowise']

/** Drawn on load; the rest start `legendonly` and are one click away.
 *  `ztforce` is ZTF forced photometry and only exists once you have asked for
 *  it, so it rides along with ztf rather than starting hidden. */
export const DEFAULT_OPTICAL_SURVEYS = ['ztf', 'ztforce', 'panstarrs']

/** W3 and W4 sit several magnitudes off W1/W2 in Vega, so they both clutter
 *  the panel and stretch its y range. */
export const DEFAULT_INFRARED_BANDS = ['w1', 'w2']

export function colorFor(band: string, survey?: string): string {
  if (survey === 'wise' || survey === 'neowise') {
    return WISE_BAND_COLOR[band] ?? '#888888'
  }
  return BAND_COLOR[BAND_ALIAS[band] ?? band] ?? '#888888'
}

export function symbolFor(survey: string): string {
  return SURVEY_SYMBOL[survey] ?? 'circle'
}

/** Categorical colours for people, not bands.
 *
 * The same six the Streamlit app assigns to buckets (`lib/constants.py`
 * BUCKET_PALETTE) and that `ui.tsx` uses for badge tones, so a member keeps
 * one colour wherever they appear. Grey is deliberately absent there: it
 * means "unbucketed".
 */
export const CATEGORICAL = [
  '#3b82f6', '#16a34a', '#8b5cf6', '#ea580c', '#dc2626', '#ca8a04',
]

/** Stable colour for the nth member, cycling. */
export function categorical(index: number): string {
  return CATEGORICAL[((index % CATEGORICAL.length) + CATEGORICAL.length)
                     % CATEGORICAL.length]
}
