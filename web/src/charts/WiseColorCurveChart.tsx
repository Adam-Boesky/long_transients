/** W1 - W2 against time: how the colour the histogram summarises evolves.
 *
 * Built from the same visit-binned, error-cut points as `WiseColorChart`
 * (`wiseColor.ts`), so the two panels always describe the same visits. Sits
 * under the WISE light curve on the same date axis and left margin, so a
 * colour change lines up by eye with the flux change that drove it.
 */
import { useMemo } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { LightCurve } from '../api/types'
import { PlotlyChart } from './PlotlyChart'
import { DATE_TICKFORMATS, MODEBAR_GUTTER, mjdToMs } from './plotly'
import { INFRARED_SURVEYS, symbolFor } from './palette'
import { AGN_COLOR_CUT, COLOR_ERR_THRESH, wiseColors } from './wiseColor'

const EMPTY: Data[] = []
const HEIGHT = 260
const STYLE = { height: HEIGHT }
const SURVEY_LABEL: Record<string, string> = { wise: 'WISE', neowise: 'NEOWISE' }

export interface WiseColorCurveChartProps {
  lightcurve: LightCurve | null
}

export function WiseColorCurveChart({ lightcurve }: WiseColorCurveChartProps) {
  const stats = useMemo(() => wiseColors(lightcurve), [lightcurve])

  const { data, layout } = useMemo(() => {
    const pts = stats.points
    if (!pts.length) return { data: EMPTY, layout: null }

    const traces: Data[] = INFRARED_SURVEYS.flatMap((survey) => {
      const p = pts.filter((q) => q.survey === survey)
      if (!p.length) return []
      return [{
        x: p.map((q) => mjdToMs(q.mjd)),
        y: p.map((q) => q.color),
        customdata: p.map((q) => q.mjd),
        error_y: { type: 'data', array: p.map((q) => q.err), visible: true,
                   thickness: 1, width: 0, color: '#22262e' },
        type: 'scatter',
        mode: 'markers',
        name: SURVEY_LABEL[survey] ?? survey,
        marker: { color: '#22262e', symbol: symbolFor(survey), size: 6 },
        hovertemplate: '%{x|%Y-%m-%d} (MJD %{customdata:.4f})<br>'
          + 'W1 - W2 %{y:.3f} ± %{error_y.array:.3f}<extra></extra>',
      } as Data]
    })

    // Fixed ranges, as in the light-curve panel, with the 0.8 cut kept in view.
    const xs = pts.map((q) => mjdToMs(q.mjd))
    const xLo = Math.min(...xs), xHi = Math.max(...xs)
    const xPad = (xHi - xLo) * 0.05 || 30 * 86400000
    const yLo = Math.min(AGN_COLOR_CUT, ...pts.map((q) => q.color - q.err))
    const yHi = Math.max(AGN_COLOR_CUT, ...pts.map((q) => q.color + q.err))
    const yPad = (yHi - yLo) * 0.08 || 0.25

    // A trace rather than a shape so it gets a legend entry, as in the histogram.
    traces.push({
      x: [xLo - xPad, xHi + xPad], y: [AGN_COLOR_CUT, AGN_COLOR_CUT],
      type: 'scatter', mode: 'lines', name: `W1 - W2 = ${AGN_COLOR_CUT}`,
      line: { color: 'red', width: 1.5 }, hoverinfo: 'skip',
    } as Data)

    const lay: Partial<Layout> = {
      height: HEIGHT,
      xaxis: { title: { text: 'Year' }, type: 'date', tickformatstops: DATE_TICKFORMATS,
               range: [xLo - xPad, xHi + xPad], autorange: false },
      yaxis: { title: { text: 'W1 - W2' }, range: [yLo - yPad, yHi + yPad], autorange: false },
      margin: { l: 62, r: MODEBAR_GUTTER, t: 0, b: 46, autoexpand: true },
    }
    return { data: traces, layout: lay }
  }, [stats])

  // Same messages as the histogram, so the two empty panels agree.
  const emptyLayout = useMemo<Partial<Layout>>(() => ({
    height: HEIGHT,
    showlegend: false,
    xaxis: { title: { text: 'Year' }, range: [0, 1], showticklabels: false, zeroline: false },
    yaxis: { title: { text: 'W1 - W2' }, range: [0, 1], showticklabels: false, zeroline: false },
    margin: { l: 62, r: MODEBAR_GUTTER, t: 10, b: 46, autoexpand: true },
    annotations: [{
      text: !lightcurve ? 'No light curve.'
        : stats.nPaired === 0 ? 'Source has no WISE data.'
        : `No WISE visit with σ(W1-W2) < ${COLOR_ERR_THRESH}.`,
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      showarrow: false, font: { size: 13, color: '#5d6673' },
    }],
  }), [lightcurve, stats.nPaired])

  return (
    <div>
      <div style={{ font: '600 12px system-ui', opacity: 0.75, margin: '2px 0 2px 62px' }}>
        W1 - W2 colour
      </div>
      <PlotlyChart data={layout ? data : EMPTY} layout={layout ?? emptyLayout} style={STYLE} />
    </div>
  )
}
