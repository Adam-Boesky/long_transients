/** The render-time reductions the stored light curves deliberately omit.
 *
 * `source_db.lightcurves._base.RENDER_TIME_REDUCTIONS` records what the
 * database leaves undone, because each step is lossy and parameterised:
 *
 *   ztf            drop epochs failing `catflags & (6141 | 32768)`
 *   wise, neowise  average exposures into visits, gap <= 5 days
 *
 * Doing them here rather than at write time keeps the raw measurements
 * recoverable, and keeps these knobs in the UI where they belong.
 */
import type { BandSeries } from '../api/types'

/** ZTF Explanatory Supplement 10.3, plus bit 15 (32768): cloud- or
 *  moon-affected epochs (DR23 release notes 9b). */
export const BAD_CATFLAG_MASK = 6141 | 32768

/** WISE revisits a field ~12 times over ~36 h; 5 days spans a visit with
 *  margin while staying far below the ~180-day revisit cadence. */
export const VISIT_GAP_DAYS = 5.0

export interface Epochs {
  mjd: number[]
  mag: number[]
  magerr: number[]
  /** Per epoch, aligned with the arrays above. */
  isUpperLimit: boolean[]
}

/** The magnitude error at SNR = 2, sigma_m = 2.5 / (ln10 * SNR). */
export const SNR2_MAGERR = 2.5 / (Math.LN10 * 2)

/** Is epoch `i` a non-detection?
 *
 * SNR < 2 is the threshold the surveys themselves use, applied uniformly.
 *
 *   WISE / NEOWISE  The Explanatory Supplement is explicit: `w?sigmpro` is
 *     null "if the magnitude is a 95% confidence upper limit or the source
 *     is not measurable", and the magnitude is "the 95% confidence
 *     brightness upper limit if the flux measurement has SNR<2"
 *     (`ph_qual` = U). So a null error *is* the survey saying "limit", and
 *     the SNR-2 test below is not applied: the survey has already made the
 *     call, and re-deriving it from the quoted error second-guesses it.
 *
 *     That mattered. WISE caps `w?sigmpro` at the SNR-2 value, and the cap
 *     it stores (0.54299998) is 1.3e-4 *above* 2.5/(ln10*2) = 0.542868, so
 *     the test re-flagged 267 capped epochs as limits while
 *     `Light_Curve._bin_visit_clusters` -- which this file is supposed to
 *     mirror, and which feeds `wise_w1w2_mean` -- kept them. Same colour,
 *     two answers, up to 0.058 mag apart on the median.
 *
 *     Stern et al. (2012) points the same way: the AGN criterion requires
 *     W2 SNR >= 10, so an epoch anywhere near SNR 2 is excluded by the cut
 *     itself. Splitting 1.9995 from 2.0005 is two orders of magnitude below
 *     the threshold that decides anything.
 *
 *   SDSS  Reports asinh magnitudes (Lupton et al. 1999) precisely so that
 *     low and negative fluxes still have a value, and states "we quote no
 *     upper limits in our photometry" -- so nothing here is flagged and the
 *     threshold is ours to pick. SDSS does publish the asinh magnitude of
 *     zero flux per band (u 24.63, g 25.11, r 24.80, i 24.36, z 22.83), and
 *     our faint tail piles up immediately past it -- 1,903 of 1,927 faint
 *     i-band epochs fall in 24.36-25.00, i.e. negative measured flux, with
 *     median error 1.6 mag. SNR < 2 catches those and the merely-noisy ones
 *     with them.
 *
 *   ZTF / PanSTARRS / Gaia  Detection-only catalogues; no epoch of any of
 *     them is below SNR 2, so the rule is a no-op rather than a guess.
 *
 *   custom  Carries an explicit flag, which wins outright.
 */
export function isUpperLimitAt(survey: string, series: BandSeries,
                               i: number): boolean {
  if (series.is_upper_limit?.[i]) return true
  const e = series.magerr?.[i]
  // A missing error is WISE's way of saying "limit"; elsewhere it means the
  // epoch carries no uncertainty at all, which is not a measurement either.
  if (e === null || e === undefined || !Number.isFinite(e)) return true
  // Surveys that publish their own limits have already answered this.
  if (PUBLISHES_LIMITS.has(survey)) return false
  return e >= SNR2_MAGERR
}

/** Surveys whose catalogue marks non-detections itself, by nulling the error.
 *  For these the null check above is the whole rule. */
const PUBLISHES_LIMITS = new Set(['wise', 'neowise'])

function reorder(survey: string, series: BandSeries, keep: number[]): Epochs {
  return {
    mjd: keep.map((i) => series.mjd[i]),
    mag: keep.map((i) => series.mag[i]),
    magerr: keep.map((i) => series.magerr[i]),
    // Carried through the reduction rather than read off the raw series by
    // the caller: every reduction here drops or merges epochs, so raw
    // indices do not line up with what is drawn.
    isUpperLimit: keep.map((i) => isUpperLimitAt(survey, series, i)),
  }
}

/** Drop ZTF epochs that fail the quality mask. A series with no `catflags`
 *  is returned untouched rather than silently emptied. */
export function applyCatflags(series: BandSeries): Epochs {
  if (!series.catflags) return reorder('ztf', series, series.mjd.map((_, i) => i))
  const keep: number[] = []
  for (let i = 0; i < series.mjd.length; i++) {
    // eslint-disable-next-line no-bitwise
    if ((series.catflags[i] & BAD_CATFLAG_MASK) === 0) keep.push(i)
  }
  return reorder('ztf', series, keep)
}

/** Cluster ids for already-sorted mjds: a new cluster starts wherever the gap
 *  to the *previous* point exceeds `maxGap`.
 *
 *  Measured from the previous point rather than from the cluster's first
 *  point, so a long visit whose total span exceeds the gap is not split even
 *  though every internal gap is small. Mirrors
 *  `Light_Curve._cluster_by_gap`.
 */
function clusterByGap(sortedMjd: number[], maxGap = VISIT_GAP_DAYS): number[] {
  const ids: number[] = []
  let current = -1
  for (let i = 0; i < sortedMjd.length; i++) {
    if (i === 0 || sortedMjd[i] - sortedMjd[i - 1] > maxGap) current++
    ids.push(current)
  }
  return ids
}

/** Combine WISE/NEOWISE exposures into one point per visit.
 *
 *  Inverse-variance weighted in *flux*, matching `Light_Curve._bin_visit_clusters`
 *  and AllWISE's own WPRO methodology (Explanatory Supplement sec2_3b / sec4_4c):
 *  averaged profile-fit magnitudes come from weighted fluxes, not averaged mags.
 *  The error reported is the error *on the mean*, not the typical per-exposure
 *  error -- for the ~11 exposures in a visit those differ by about sqrt(N).
 *
 *  This docstring used to claim the opposite -- a plain `np.nanmean`, 'not an
 *  inverse-variance weighting' -- which described `Light_Curve`'s WISE branch
 *  while the code here did what its NEOWISE branch did. Both now do this.
 */
export function binVisits(survey: string, series: BandSeries,
                          maxGap = VISIT_GAP_DAYS): Epochs {
  const n = series.mjd.length
  if (n === 0) return { mjd: [], mag: [], magerr: [], isUpperLimit: [] }

  // The payload is already mjd-sorted per band, but clustering depends on it,
  // so sort defensively rather than assume.
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => series.mjd[a] - series.mjd[b])
  const mjd = order.map((i) => series.mjd[i])
  const mag = order.map((i) => series.mag[i])
  const magerr = order.map((i) => series.magerr[i])
  const upper = order.map((i) => isUpperLimitAt(survey, series, i))

  const ids = clusterByGap(mjd, maxGap)
  const out: Epochs = { mjd: [], mag: [], magerr: [], isUpperLimit: [] }
  let i = 0
  while (i < n) {
    let j = i
    while (j < n && ids[j] === ids[i]) j++
    out.mjd.push(nanmean(mjd.slice(i, j)))

    // Detections win inside a visit: upper limits are averaged in only when
    // the cluster has none, matching `Light_Curve`'s NEOWISE binning. A
    // plain mean over everything drags a mostly-detected visit toward the
    // limit values, which are bounds rather than measurements.
    const idx: number[] = []
    for (let k = i; k < j; k++) if (!upper[k]) idx.push(k)
    if (idx.length === 0) {
      out.mag.push(nanmean(mag.slice(i, j)))
      out.magerr.push(NaN)
      out.isUpperLimit.push(true)
      i = j
      continue
    }

    // Inverse-variance weighted in *flux*, as AllWISE combines single
    // exposures (Explanatory Supplement sec2_3b / sec4_4c): averaged
    // profile-fit magnitudes come from weighted fluxes, not averaged mags.
    let wSum = 0, wfSum = 0
    const fluxes: number[] = []
    for (const k of idx) {
      const f = 10 ** (-mag[k] / 2.5)
      const fe = magerr[k] * f * Math.LN10 / 2.5
      if (!Number.isFinite(f) || !Number.isFinite(fe) || fe <= 0) continue
      const w = 1 / (fe * fe)
      wSum += w
      wfSum += w * f
      fluxes.push(f)
    }
    if (wSum === 0) {
      out.mag.push(nanmean(idx.map((k) => mag[k])))
      out.magerr.push(NaN)
      out.isUpperLimit.push(false)
      i = j
      continue
    }
    const fMean = wfSum / wSum
    const formal = 1 / Math.sqrt(wSum)
    // AllWISE tracks the scatter across exposures alongside the formal
    // error, since the formal one understates a genuinely variable source.
    // Report whichever is larger.
    const empirical = fluxes.length >= 2 ? stdErr(fluxes) : 0
    const fErr = Math.max(formal, empirical)

    out.mag.push(-2.5 * Math.log10(fMean))
    out.magerr.push((2.5 / Math.LN10) * fErr / fMean)
    out.isUpperLimit.push(false)
    i = j
  }
  return out
}

/** Standard error of the mean, with the sample (ddof=1) standard deviation. */
function stdErr(xs: number[]): number {
  const n = xs.length
  const mean = xs.reduce((a, b) => a + b, 0) / n
  const varr = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
  return Math.sqrt(varr) / Math.sqrt(n)
}

function nanmean(values: number[]): number {
  let sum = 0
  let count = 0
  for (const v of values) {
    if (v !== null && Number.isFinite(v)) { sum += v; count++ }
  }
  return count ? sum / count : NaN
}

/** Everything a survey needs doing to it before it is drawn. */
export function reduceSeries(
  survey: string, series: BandSeries,
  opts: { catflags?: boolean; binVisits?: boolean } = {},
): Epochs {
  const applyCat = opts.catflags ?? true
  const applyBin = opts.binVisits ?? true
  if (survey === 'ztf' && applyCat) return applyCatflags(series)
  if ((survey === 'wise' || survey === 'neowise') && applyBin) {
    return binVisits(survey, series)
  }
  return reorder(survey, series, series.mjd.map((_, i) => i))
}
