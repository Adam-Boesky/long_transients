/** The W1 - W2 distribution, as `Sources.plot_wise_mag_hist` draws it.
 *
 * Sits beside the WISE light curve because it answers the question that
 * panel raises: the analysis page puts them in the same row, 8 grid columns
 * to 4, which is the 2:1 the classify page uses.
 *
 * Drawn with `bar` over bins computed in `wiseColor.ts` rather than Plotly's
 * `histogram` trace. The partial bundle would need another trace type for
 * that, and matplotlib's `bins=10` is ten equal-width bins over the data
 * range -- cheap to reproduce exactly, and it keeps the binning testable.
 */
import { useMemo } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { LightCurve } from '../api/types'
import { PlotlyChart } from './PlotlyChart'
import { MODEBAR_GUTTER } from './plotly'
import { AGN_COLOR_CUT, COLOR_ERR_THRESH, histogram, wiseColors } from './wiseColor'

const EMPTY: Data[] = []

/** Matches the light-curve panel it sits beside. */
const HEIGHT = 340

const STYLE = { height: HEIGHT }

export interface WiseColorChartProps {
  lightcurve: LightCurve | null
}

export function WiseColorChart({ lightcurve }: WiseColorChartProps) {
  const stats = useMemo(() => wiseColors(lightcurve), [lightcurve])

  const { data, bins } = useMemo(() => {
    const b = histogram(stats.colors)
    if (!b.counts.length) return { data: EMPTY, bins: b }

    // The reference lines are traces, not `layout.shapes`. Shapes cannot
    // carry a legend entry, and the matplotlib version labels these three
    // with `ax.legend()` -- as traces they get that for free, and become
    // toggleable as a bonus.
    const top = Math.max(...b.counts) * 1.05
    const vline = (x: number, name: string, color: string,
                   dash?: 'dash'): Data => ({
      x: [x, x],
      y: [0, top],
      type: 'scatter',
      mode: 'lines',
      name,
      line: { color, width: 1.5, ...(dash ? { dash } : {}) },
      hoverinfo: 'skip',
    } as Data)

    const traces: Data[] = [{
      // Bars sit on bin centres with the bin width, so they tile the axis
      // the way a histogram's do. Kept out of the legend, as the
      // matplotlib histogram is.
      x: b.edges.map((e) => e + b.width / 2),
      y: b.counts,
      width: b.width,
      type: 'bar',
      marker: { color: '#22262e' },
      showlegend: false,
      hovertemplate: 'W1 - W2 %{x:.3f}<br>%{y} visits<extra></extra>',
    } as Data,
      vline(AGN_COLOR_CUT, `W1 - W2 = ${AGN_COLOR_CUT}`, 'red')]

    if (stats.mean !== null) {
      traces.push(vline(stats.mean, `Mean (${stats.mean.toFixed(2)})`, '#16a34a'))
    }
    if (stats.median !== null) {
      traces.push(vline(stats.median, `Median (${stats.median.toFixed(2)})`,
                        '#16a34a', 'dash'))
    }
    return { data: traces, bins: b }
  }, [stats])

  const layout = useMemo<Partial<Layout>>(() => {
    // The 0.8 cut is the point of the panel, so it stays in view even when
    // every epoch sits well below it.
    const lo = Math.min(bins.edges[0] ?? AGN_COLOR_CUT, AGN_COLOR_CUT)
    const hi = Math.max((bins.edges.at(-1) ?? AGN_COLOR_CUT) + bins.width,
                        AGN_COLOR_CUT)
    const pad = (hi - lo) * 0.08 || 0.5
    const top = bins.counts.length ? Math.max(...bins.counts) * 1.05 : 1

    return {
      height: HEIGHT,
      bargap: 0,
      xaxis: { title: { text: 'W1 - W2' },
               range: [lo - pad, hi + pad], autorange: false },
      // Fixed, so the reference lines drawn to `top` do not themselves
      // stretch the axis past the tallest bar.
      yaxis: { title: { text: 'Number' }, range: [0, top], autorange: false },
      margin: { l: 52, r: MODEBAR_GUTTER, t: 0, b: 46, autoexpand: true },
    }
  }, [bins])

  // Same reasoning as the light-curve panel: keep the axes and put the
  // message inside them, so an empty panel reads as "we looked".
  const emptyLayout = useMemo<Partial<Layout>>(() => ({
    height: HEIGHT,
    showlegend: false,
    xaxis: { title: { text: 'W1 - W2' }, range: [0, 1],
             showticklabels: false, zeroline: false },
    yaxis: { title: { text: 'Number' }, range: [0, 1],
             showticklabels: false, zeroline: false },
    margin: { l: 52, r: MODEBAR_GUTTER, t: 10, b: 46, autoexpand: true },
    annotations: [{
      text: !lightcurve ? 'No light curve.'
        : stats.nPaired === 0 ? 'Source has no WISE data.'
        // Epochs exist but every one is too noisy to colour: a different
        // statement from having no WISE coverage, and `plot_wise_mag_hist`
        // distinguishes them too.
        : `No WISE visit with<br>σ(W1-W2) < ${COLOR_ERR_THRESH}.`,
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      showarrow: false, align: 'center',
      font: { size: 13, color: '#5d6673' },
    }],
  }), [lightcurve, stats.nPaired])
  const noData = stats.colors.length === 0

  return <PlotlyChart data={noData ? EMPTY : data}
                      layout={noData ? emptyLayout : layout} style={STYLE} />
}
