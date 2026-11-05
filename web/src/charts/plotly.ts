/** A partial Plotly bundle.
 *
 * `plotly.js/dist/plotly` is ~1 MB gzipped and pulls in every trace type;
 * `react-plotly.js` is worse, with unoptimised builds regularly exceeding
 * 10 MB, and it is barely maintained. Registering only the two trace types
 * this app draws keeps it to a few hundred KB.
 *
 * `scatter` is SVG and `scattergl` is WebGL. Both are registered, but they
 * are not interchangeable: a browser allows only a handful of WebGL contexts
 * (Plotly hits trouble past ~8 charts on a page), so `scattergl` is reserved
 * for the spectrum, which is the only trace big enough to need it.
 */
import type { Config, Layout, TickFormatStops } from 'plotly.js'
import Plotly from 'plotly.js/lib/core'
import bar from 'plotly.js/lib/bar'
import scatter from 'plotly.js/lib/scatter'
import scattergl from 'plotly.js/lib/scattergl'

// `bar` is SVG and tiny; the Progress page's label/author breakdown is the
// only thing that draws one.
Plotly.register([bar, scatter, scattergl])

export default Plotly

/** MJD of the Unix epoch, 1970-01-01T00:00:00Z. */
const MJD_UNIX_EPOCH = 40587
const MS_PER_DAY = 86400000

/** MJD -> milliseconds since the Unix epoch, which is what a Plotly axis of
 *  `type: 'date'` reads. The epochs stay MJD everywhere else -- that is the
 *  unit the database and `Light_Curve` speak -- and are converted only for
 *  display, with the MJD carried alongside in `customdata` so the hover can
 *  still quote it. */
export const mjdToMs = (mjd: number) => (mjd - MJD_UNIX_EPOCH) * MS_PER_DAY

/** Label the axis by year while it spans years, and fall back to finer units
 *  as you zoom, so a single-season view does not repeat the same year label
 *  across every tick. */
export const DATE_TICKFORMATS: TickFormatStops[] = [
  { dtickrange: [null, 1000 * 60 * 60 * 24 * 60], value: '%d %b %Y' },
  { dtickrange: [1000 * 60 * 60 * 24 * 60, 1000 * 60 * 60 * 24 * 550], value: '%b %Y' },
  { dtickrange: [1000 * 60 * 60 * 24 * 550, null], value: '%Y' },
]

/** Breathing room between the legend and the top of the plotting area. */
export const LEGEND_GAP = 6

/** Right margin reserved for the vertical modebar.
 *
 * Plotly pins the modebar to the top-right of the *container*, not the
 * plotting area, so a horizontal one lands on whatever is there: with a wide
 * legend it covers the last entries, and `elementFromPoint` over them returns
 * `modebar-btn` -- the click toggles zoom instead of the trace. Standing it up
 * and reserving a gutter puts it beside the axis, clear of both the legend
 * (which wraps at the narrowed plot width) and the data.
 */
export const MODEBAR_GUTTER = 42

/** Axis titles are bold. `weight` is a real font attribute (integer 1-1000,
 *  or the keywords 'normal'/'bold'); it does not need `<b>` markup in the
 *  title string. */
export const AXIS_TITLE_FONT = { size: 12, weight: 'bold' as const }

/** Shared look: light, uncluttered, and close to the matplotlib figures. */
export const BASE_LAYOUT: Partial<Layout> = {
  font: { family: 'system-ui, sans-serif', size: 12, color: '#22262e' },
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  // `t: 0` is a floor, not a fixed value. The legend calls
  // `Plots.autoMargin`, which reserves `legendHeight * FROM_TL[yanchor]` --
  // and `FROM_TL.bottom` is 1, so with `yanchor: 'bottom'` below the top
  // margin grows to exactly the legend's height, however many rows it wraps
  // to. Leaving `t` unset instead inherits Plotly's default of 100, which is
  // a lot of empty space above a one-row legend.
  margin: { l: 62, r: MODEBAR_GUTTER, t: 0, b: 46, autoexpand: true },
  hovermode: 'closest',
  // Vertical, so it stacks in the gutter rather than running along the top
  // band the legend uses.
  modebar: { orientation: 'v' },
  showlegend: true,
  // Positioned against the *container*, not the paper. With `yref: 'paper'`,
  // y=1 is the top of the plotting area, so the legend renders inside the
  // axes however it is anchored. `yref: 'container'` measures from the top
  // of the whole figure, which is the band `margin.t` reserves -- and
  // PlotlyChart sizes that margin to the legend's measured height, so the
  // two always agree however many rows the legend wraps to.
  legend: {
    orientation: 'h',
    x: 0, xanchor: 'left',
    y: 1, yanchor: 'top', yref: 'container',
  },
  xaxis: {
    title: { font: AXIS_TITLE_FONT },
    showgrid: true, gridcolor: '#eceff3',
    zeroline: false, ticks: 'outside', mirror: true,
    showline: true, linecolor: '#c4cad2', tickcolor: '#c4cad2',
  },
  yaxis: {
    title: { font: AXIS_TITLE_FONT },
    showgrid: true, gridcolor: '#eceff3',
    zeroline: false, ticks: 'outside', mirror: true,
    showline: true, linecolor: '#c4cad2', tickcolor: '#c4cad2',
  },
}

export const BASE_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  // The subset worth keeping: the rest is for chart types we do not draw.
  modeBarButtonsToRemove: [
    'select2d', 'lasso2d', 'toggleSpikelines',
    'hoverCompareCartesian', 'hoverClosestCartesian',
  ],
  toImageButtonOptions: { format: 'png', scale: 2 },
}
