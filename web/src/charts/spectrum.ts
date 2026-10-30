/** Rebuilding a spectrum's wavelength axis.
 *
 * The serving tier drops the wavelength array when the grid is analytic --
 * DESI samples linearly in wavelength, SDSS linearly in log10(wavelength)
 * with dloglam = 1e-4 -- and ships `wave_start`/`wave_step` instead. That is
 * a third of the payload for something exactly reconstructible; see
 * `source_db.serving.payload.fit_grid`.
 *
 * An irregular grid falls back to an explicit `wavelength` array, so both
 * shapes have to be handled.
 */
import type { Spectrum } from '../api/types'

/** Wavelength in Angstroms for every pixel, however the payload encoded it. */
export function wavelengthOf(s: Spectrum): number[] {
  if (s.wavelength && s.wavelength.length) {
    return s.wavelength.map((v) => (v == null ? NaN : v))
  }
  if (s.wave_start === undefined || s.wave_step === undefined) return []
  const out = new Array<number>(s.n_pix)
  for (let i = 0; i < s.n_pix; i++) {
    const g = s.wave_start + i * s.wave_step
    out[i] = s.wave_scale === 'log10' ? 10 ** g : g
  }
  return out
}

/** Finite y-range of the *model* curves.
 *
 * `plot_spectrum` sets the y-limits from the smooth pipeline fits before
 * drawing the noisier flux, so a few spikes in the raw data cannot blow out
 * the scale. Reproduced here, with a small margin.
 */
export function modelRange(spectra: Spectrum[]): [number, number] | null {
  let lo = Infinity
  let hi = -Infinity
  for (const s of spectra) {
    for (const v of s.model ?? []) {
      if (v == null || !Number.isFinite(v)) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null
  const pad = (hi - lo) * 0.08 || 1
  return [lo - pad, hi + pad]
}
