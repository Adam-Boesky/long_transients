/** W1 - W2 per visit, the quantity `plot_wise_mag_hist` histograms.
 *
 * The analysis code reads a table where `w1_mag` and `w2_mag` are columns of
 * the same row, so the colour is a subtraction. Our light curves are stored
 * one array per band, and the two bands do not have the same length -- an
 * epoch can be measured in W1 and not W2.
 *
 * Colours come from visit-binned photometry, not raw exposures, for the same
 * reason the light-curve panel beside it bins. Measured over one uuid bucket,
 * the propagated per-exposure colour error has median 0.274 and 90th
 * percentile 0.505, against a Stern+2012 threshold of 0.8 -- so a
 * per-exposure distribution straddles that line largely through photometric
 * noise. The AllWISE coadd colour the criterion was calibrated on has a
 * median error of 0.098; binning moves this toward that regime rather than
 * away from it, and makes the two panels describe the same measurements.
 *
 * `binVisits` combines exposures by inverse-variance weighting in flux, which
 * is how AllWISE builds `w?mpro` from single exposures -- so a binned visit
 * is a small coadd rather than an average of magnitudes.
 */
import type { LightCurve } from '../api/types'
import { VISIT_GAP_DAYS, binVisits } from './reductions'
import { INFRARED_SURVEYS } from './palette'

/** Binned points are paired within half a visit rather than on an exact MJD.
 *
 *  Raw exposures could be matched exactly, because WISE reports both bands
 *  from one exposure. Binning breaks that: each band's visit MJD is the mean
 *  over *its own* epochs, so a visit where one band is missing an exposure
 *  gets a slightly different mean. Visits are ~180 days apart, so a 2.5-day
 *  window cannot pair the wrong two.
 */
const PAIR_TOLERANCE_DAYS = VISIT_GAP_DAYS / 2

/** Sources above this are AGN (Stern et al. 2012, ApJ 753 30). Re-exported
 *  from the filter state so the histogram's reference line and the sidebar's
 *  cut cannot drift apart. */
export { W1W2_CUT as AGN_COLOR_CUT } from '../lib/filterState'

/** Epochs whose propagated colour error exceeds this are dropped, matching
 *  `plot_wise_mag_hist(color_err_thresh=0.5)`. */
export const COLOR_ERR_THRESH = 0.5

/** `ax.hist(delta_mag, bins=10)`. */
export const N_BINS = 10

/** One visit's colour, for plotting it against time. */
export interface ColorPoint {
  survey: string
  /** Mean of the paired W1 and W2 visit times. */
  mjd: number
  color: number
  err: number
}

export interface WiseColors {
  /** W1 - W2 for every visit passing the error cut. */
  colors: number[]
  /** The same visits, with their times and errors. */
  points: ColorPoint[]
  /** Visits measured in both bands, before the error cut. */
  nPaired: number
  mean: number | null
  median: number | null
}

export function wiseColors(lc: LightCurve | null): WiseColors {
  const colors: number[] = []
  const points: ColorPoint[] = []
  let nPaired = 0

  for (const survey of INFRARED_SURVEYS) {
    const bands = lc?.surveys?.[survey]
    if (!bands?.w1 || !bands?.w2) continue

    // Each band is binned on its own epochs, exactly as the light-curve
    // panel bins it, so the two charts are built from the same points.
    const w1 = binVisits(survey, bands.w1)
    const w2 = binVisits(survey, bands.w2)

    // Both are mjd-sorted, so one pass pairs them: advance w2 while it is
    // more than a tolerance behind, then take it if it is within one.
    let j = 0
    for (let i = 0; i < w1.mjd.length; i++) {
      const t = w1.mjd[i]
      while (j < w2.mjd.length && w2.mjd[j] < t - PAIR_TOLERANCE_DAYS) j++
      if (j >= w2.mjd.length) break
      if (Math.abs(w2.mjd[j] - t) > PAIR_TOLERANCE_DAYS) continue

      const m1 = w1.mag[i]
      const m2 = w2.mag[j]
      if (!Number.isFinite(m1) || !Number.isFinite(m2)) continue
      nPaired += 1
      // A visit made only of upper limits carries a NaN error and is dropped
      // here -- the same rule as before: a missing error fails the cut rather
      // than counting as zero, which would let an unmeasured visit through as
      // though it were the most precise one.
      // sigma(W1-W2) = sqrt(sigma_W1^2 + sigma_W2^2).
      const e1 = w1.magerr[i]
      const e2 = w2.magerr[j]
      if (!Number.isFinite(e1) || !Number.isFinite(e2)) continue
      const err = Math.hypot(e1, e2)
      if (err >= COLOR_ERR_THRESH) continue
      colors.push(m1 - m2)
      points.push({ survey, mjd: (t + w2.mjd[j]) / 2, color: m1 - m2, err })
    }
  }

  return {
    colors,
    points,
    nPaired,
    mean: colors.length ? colors.reduce((a, b) => a + b, 0) / colors.length : null,
    median: colors.length ? medianOf(colors) : null,
  }
}

function medianOf(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

export interface Bins {
  /** Left edge of each bin. */
  edges: number[]
  counts: number[]
  width: number
}

/** Equal-width bins across the data range, as numpy's `bins=10` does. */
export function histogram(values: number[], nBins = N_BINS): Bins {
  if (values.length === 0) return { edges: [], counts: [], width: 0 }
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  // numpy widens a zero-width range to [x-0.5, x+0.5] rather than dividing
  // by zero, and a single repeated colour is a real case here.
  const [a, b] = lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi]
  const width = (b - a) / nBins
  const counts = new Array(nBins).fill(0)
  for (const v of values) {
    // The last bin is closed on the right, so the maximum lands inside it.
    const k = v === b ? nBins - 1 : Math.floor((v - a) / width)
    counts[k] += 1
  }
  return { edges: Array.from({ length: nBins }, (_, i) => a + i * width),
           counts, width }
}
