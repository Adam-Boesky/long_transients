/** The spectrum panel, overlaying SDSS and DESI as `plot_spectrum` does.
 *
 * Two traces per spectrum, mirroring the matplotlib version: the pipeline
 * `model` drawn solid on top, and the raw `flux` beneath it at low opacity as
 * unlabelled context. The y-range is taken from the *models* before the flux
 * is added, so a few noise spikes cannot blow out the scale -- the same
 * reason `plot_spectrum` sets ylim when it does.
 *
 * Every SDSS spectrum of the source is drawn, with no preference among them:
 * repeat SDSS-V visits mean some sources have dozens. Each is one legend
 * entry (date, class, redshift) toggling its flux and model together, shaded
 * light to dark by date.
 *
 * This is the one panel drawn with `scattergl`. At ~7,781 points per trace
 * and possibly hundreds of traces, SVG would be sluggish; everything else in
 * the app stays on SVG so the page never approaches the browser's WebGL
 * context ceiling.
 */
import { useMemo } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { Spectra, Spectrum } from '../api/types'
import { PlotlyChart } from './PlotlyChart'
import { MODEBAR_GUTTER } from './plotly'
import { modelRange, wavelengthOf } from './spectrum'

/** matplotlib's tab:blue and tab:red, as `plot_spectrum` uses. */
const SURVEY_COLOR: Record<string, string> = {
  sdss: '#1f77b4',
  desi: '#d62728',
}

const SURVEY_LABEL: Record<string, string> = {
  sdss: 'SDSS',
  desi: 'DESI DR1',
}

/** SDSS spectra shade from light to dark blue by date when there are several. */
function sdssShade(i: number, n: number): string {
  if (n <= 1) return SURVEY_COLOR.sdss
  const t = i / (n - 1)
  const mix = (a: number, b: number) => Math.round(a + (b - a) * t)
  return `rgb(${mix(140, 8)}, ${mix(190, 48)}, ${mix(230, 107)})`
}

/** MJD -> YYYY-MM-DD. */
function mjdDate(mjd: number): string {
  return new Date((mjd - 40587) * 86400e3).toISOString().slice(0, 10)
}

function spectrumLabel(s: Spectrum, survey: string): string {
  const parts = [survey]
  if (s.mjd != null) parts.push(mjdDate(s.mjd))
  if (s.spec_class) {
    parts.push(s.redshift != null ? `${s.spec_class} z=${s.redshift.toFixed(3)}` : s.spec_class)
  }
  return parts.join(' · ')
}

/** Repeated between the populated and empty layouts, so they cannot drift. */
const Y_TITLE =
  'F<sub>λ</sub> [10<sup>-17</sup> erg cm<sup>-2</sup> s<sup>-1</sup> Å<sup>-1</sup>]'
const X_TITLE = 'Wavelength [Å]'

/** Stable empty dataset, so the no-data path does not hand Plotly a new
 *  array on every render. */
const EMPTY: Data[] = []

/** Shorter than the light-curve panel: a spectrum is read for shape, not
 *  for the span of a long baseline. */
const HEIGHT = 320

const STYLE = { height: HEIGHT }

export interface SpectrumChartProps {
  spectra: Spectra | null
}

export function SpectrumChart({ spectra }: SpectrumChartProps) {
  // Memoised: a bare `?? []` is a new array every render, so both memos
  // below recomputed on every parent render.
  const items = useMemo(() => spectra?.spectra ?? [], [spectra])

  const data = useMemo<Data[]>(() => {
    const traces: Data[] = []
    // Oldest first, so the shading and the legend both read in date order.
    const sdss = items.filter((s) => s.survey === 'sdss')
      .sort((a, b) => (a.mjd ?? 0) - (b.mjd ?? 0))
    const ordered = [...sdss, ...items.filter((s) => s.survey !== 'sdss')]
    ordered.forEach((s, k) => {
      const color = s.survey === 'sdss'
        ? sdssShade(k, sdss.length) : (SURVEY_COLOR[s.survey] ?? '#666666')
      const label = spectrumLabel(s, SURVEY_LABEL[s.survey] ?? s.survey)
      const group = `${s.survey}-${s.spec_id}`
      const wave = wavelengthOf(s)
      if (wave.length === 0) return

      // Raw flux first so it sits *under* the model, as in plot_spectrum.
      traces.push({
        x: wave, y: s.flux, type: 'scattergl', mode: 'lines',
        line: { color, width: 1 },
        opacity: 0.3,
        name: `${label} flux`,
        legendgroup: group,
        showlegend: false,       // context for the model, not its own entry
        hoverinfo: 'skip',
      } as Data)
      if (s.model) {
        traces.push({
          x: wave, y: s.model, type: 'scattergl', mode: 'lines',
          line: { color, width: 1.4 },
          name: label,
          legendgroup: group,
          hovertemplate:
            `<b>${label}</b><br>%{x:.1f} A<br>%{y:.3f}<extra></extra>`,
        } as Data)
      }
    })
    return traces
  }, [items])

  const layout = useMemo<Partial<Layout>>(() => {
    const range = modelRange(items)
    return {
      height: HEIGHT,
      xaxis: { title: { text: X_TITLE } },
      yaxis: {
        title: { text: Y_TITLE },
        ...(range ? { range, autorange: false } : {}),
      },
      margin: { l: 72, r: MODEBAR_GUTTER, t: 0, b: 46, autoexpand: true },
    }
  }, [items])

  // Draw the frame even with nothing in it, as the light-curve panels do:
  // an empty framed panel says "no spectrum exists", a bare line of text
  // looks like a failure to load.
  const emptyLayout = useMemo<Partial<Layout>>(() => ({
    height: HEIGHT,
    showlegend: false,
    xaxis: { title: { text: X_TITLE }, range: [0, 1],
             showticklabels: false, zeroline: false },
    yaxis: { title: { text: Y_TITLE }, range: [0, 1],
             showticklabels: false, zeroline: false },
    margin: { l: 72, r: MODEBAR_GUTTER, t: 10, b: 46, autoexpand: true },
    annotations: [{
      text: 'Source has no spectrum from SDSS or DESI DR1.',
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      showarrow: false, font: { size: 13, color: '#5d6673' },
    }],
  }), [])

  const noData = data.length === 0

  return (
    <PlotlyChart data={noData ? EMPTY : data}
                 layout={noData ? emptyLayout : layout} style={STYLE} />
  )
}
