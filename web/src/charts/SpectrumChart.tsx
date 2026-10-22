/** The spectrum panel, overlaying SDSS and DESI as `plot_spectrum` does.
 *
 * Two traces per spectrum, mirroring the matplotlib version: the pipeline
 * `model` drawn solid on top, and the raw `flux` beneath it at low opacity as
 * unlabelled context. The y-range is taken from the *models* before the flux
 * is added, so a few noise spikes cannot blow out the scale -- the same
 * reason `plot_spectrum` sets ylim when it does.
 *
 * This is the one panel drawn with `scattergl`. At ~7,781 points per trace
 * and up to four traces, SVG would be sluggish; everything else in the app
 * stays on SVG so the page never approaches the browser's WebGL context
 * ceiling.
 */
import { useMemo } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { Spectra } from '../api/types'
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
    for (const s of items) {
      const color = SURVEY_COLOR[s.survey] ?? '#666666'
      const label = SURVEY_LABEL[s.survey] ?? s.survey
      const wave = wavelengthOf(s)
      if (wave.length === 0) continue

      // Raw flux first so it sits *under* the model, as in plot_spectrum.
      traces.push({
        x: wave, y: s.flux, type: 'scattergl', mode: 'lines',
        line: { color, width: 1 },
        opacity: 0.3,
        name: `${label} flux`,
        showlegend: false,       // context for the model, not its own entry
        hoverinfo: 'skip',
      } as Data)
      if (s.model) {
        traces.push({
          x: wave, y: s.model, type: 'scattergl', mode: 'lines',
          line: { color, width: 1.4 },
          name: label,
          hovertemplate:
            `<b>${label}</b><br>%{x:.1f} A<br>%{y:.3f}<extra></extra>`,
        } as Data)
      }
    }
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
