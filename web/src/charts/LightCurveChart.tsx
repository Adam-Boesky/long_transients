/** The light-curve panel.
 *
 * One trace per (survey, band), because that is the unit a legend click
 * should toggle. Colour is the band and marker is the survey, matching
 * `Light_Curve.LC_COLOR_INFO` / `LC_MARKER_INFO` so this reads like the
 * figures it replaces.
 *
 * The render-time reductions live in `reductions.ts` and are applied here
 * rather than in the database: the stored epochs are raw on purpose, and
 * these two cuts are the only reason a stored curve differs from what
 * `Light_Curve.get_lc()` draws.
 *
 * WISE and NEOWISE are not drawn here by default. `plot_lc` leaves them to
 * `plot_wise_lc` because their magnitudes sit on a different scale; pass
 * `surveys={INFRARED_SURVEYS}` for that second panel.
 */
import { useCallback, useMemo } from 'react'
import type { Data, Layout, PlotDatum } from 'plotly.js'
import type { LightCurve } from '../api/types'
import { PlotlyChart } from './PlotlyChart'
import { DATE_TICKFORMATS, MODEBAR_GUTTER, mjdToMs } from './plotly'
import { OPTICAL_SURVEYS, colorFor, symbolFor } from './palette'
import { reduceSeries } from './reductions'

/** Stable empty dataset, so the no-data path does not hand Plotly a new
 *  array on every render. */
const EMPTY: Data[] = []

/** Upper limits are drawn back from the detections, but only a little: on
 *  WISE they are most of the points and their envelope is informative, so
 *  they need to stay legible rather than fade out. */
const UPPER_LIMIT_OPACITY = 0.55

/** The panel's drawn height. A prop once, but every caller took the default. */
const HEIGHT = 340

/** Hoisted out of the component: a fresh object each render re-renders
 *  `PlotlyChart` for nothing. */
const STYLE = { height: HEIGHT }

export interface LightCurveChartProps {
  lightcurve: LightCurve | null
  /** Which surveys to draw. Defaults to the optical set. */
  surveys?: string[]
  /** Of those, the ones shown on load. The rest are drawn `legendonly`: in
   *  the legend, one click from view, and left out of the axis range so they
   *  cannot stretch it. Omit to show everything. */
  defaultSurveys?: string[]
  /** Same, by band -- what separates W1/W2 from W3/W4. */
  defaultBands?: string[]
  title?: string
  /** Optical surveys report AB magnitudes; WISE and NEOWISE report Vega, so
   *  the two panels are not on the same scale and must say so. */
  yAxisTitle?: string
  /** Surveys whose points open their exposure when clicked. */
  clickableSurveys?: string[]
  onEpochClick?: (e: EpochClick) => void
}

/** One clicked point, as drawn (after the render-time reductions). */
export interface EpochClick {
  survey: string
  band: string
  mjd: number
  mag: number
  magerr: number | null
  upper: boolean
}

/** Carried on each trace so a click can say which series it hit. */
interface TraceMeta { survey: string; band: string; magerr: number[]; upper: boolean[] }

export function LightCurveChart({
  lightcurve,
  surveys = OPTICAL_SURVEYS,
  defaultSurveys,
  defaultBands,
  title,
  yAxisTitle = 'Mag',
  clickableSurveys,
  onEpochClick,
}: LightCurveChartProps) {
  const clickable = useMemo(() => new Set(clickableSurveys ?? []), [clickableSurveys])
  const allowed = useMemo(() => new Set(surveys), [surveys])
  const shownSurveys = useMemo(
    () => (defaultSurveys ? new Set(defaultSurveys) : null), [defaultSurveys])
  const shownBands = useMemo(
    () => (defaultBands ? new Set(defaultBands) : null), [defaultBands])

  /** Reduced series plus the data extent, computed together: the axis range
   *  is set explicitly rather than left to `autorange`, which recomputes and
   *  relayouts on every legend toggle (plotly/plotly.js#7489). */
  const { data, bounds } = useMemo(() => {
    if (!lightcurve) return { data: [] as Data[], bounds: null }
    // Each trace carries its own extent, so the axis range can be folded over
    // only the ones actually on screen.
    type Built = { trace: Data; shown: boolean
                   ext: { xLo: number; xHi: number; yLo: number; yHi: number } }
    const built: Built[] = []
    for (const [survey, bands] of Object.entries(lightcurve.surveys)) {
      if (!allowed.has(survey)) continue
      for (const [band, series] of Object.entries(bands)) {
        // The reductions `reduceSeries` applies by default: the ZTF
        // catflags cut and WISE visit binning. Both used to be props so they
        // could be turned off to see the raw epochs, but nothing ever did.
        const e = reduceSeries(survey, series)
        if (e.mjd.length === 0) continue

        const shown = (!shownSurveys || shownSurveys.has(survey))
                   && (!shownBands || shownBands.has(band))
        const times = e.mjd.map(mjdToMs)
        let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity
        for (let i = 0; i < times.length; i++) {
          const x = times[i], y = e.mag[i], err = e.magerr[i] || 0
          if (Number.isFinite(x)) { if (x < xLo) xLo = x; if (x > xHi) xHi = x }
          if (Number.isFinite(y)) {
            if (y - err < yLo) yLo = y - err
            if (y + err > yHi) yHi = y + err
          }
        }

        const color = colorFor(band, survey)
        // From the reduction, not the raw series: `e` may have dropped or
        // merged epochs, so raw indices no longer line up with what is drawn.
        const upper = e.isUpperLimit
        const anyUpper = upper.some(Boolean)
        const hint = clickable.has(survey) ? '<br><i>click to open the exposure</i>' : ''

        built.push({ shown, ext: { xLo, xHi, yLo, yHi }, trace: {
          x: times,
          y: e.mag,
          // The MJD the epoch actually carries, kept for the hover: the axis
          // is a date for reading, but MJD is what you quote and search on.
          customdata: e.mjd,
          error_y: {
            type: 'data', array: e.magerr, visible: true,
            color, thickness: 1, width: 0,
          },
          type: 'scatter',
          mode: 'markers',
          name: `${survey} ${band}`,
          meta: { survey, band, magerr: e.magerr, upper } satisfies TraceMeta,
          // No `legendgroup`: grouping makes one legend click toggle every
          // trace in the group, so clicking "sdss g" would hide all five SDSS
          // bands and redraw them. Per-(survey, band) toggling is both what
          // you want here and a fifth of the work.

          marker: {
            // Upper limits get a down arrow, and sit back: they are bounds,
            // not measurements, and on WISE they outnumber the detections
            // (54% of epochs), so at full strength they read as the curve.
            symbol: anyUpper
              ? upper.map((u) => (u ? 'triangle-down' : symbolFor(survey)))
              : symbolFor(survey),
            color,
            size: 7,
            opacity: anyUpper ? upper.map((u) => (u ? UPPER_LIMIT_OPACITY : 1)) : 1,
            line: { width: survey === 'panstarrs' ? 2 : 1, color },
          },
          // Upper limits have no uncertainty to quote, so the detection
          // template's "± err" would render as "± undefined".
          hovertemplate: anyUpper
            ? upper.map((u) => `<b>${survey} ${band}</b><br>`
                + '%{x|%Y-%m-%d} (MJD %{customdata:.4f})<br>'
                + (u ? '< %{y:.3f} (upper limit)' : '%{y:.3f}')
                + hint + '<extra></extra>')
            : `<b>${survey} ${band}</b><br>`
              + '%{x|%Y-%m-%d} (MJD %{customdata:.4f})<br>'
              + '%{y:.3f} ± %{error_y.array:.3f}' + hint + '<extra></extra>',
        } as Data })
      }
    }

    // A source with nothing in the default set -- only PTF and SDSS, say --
    // would otherwise open as an empty frame with a full legend. Falling back
    // to everything is better than showing nothing.
    const anyShown = built.some((b) => b.shown)
    let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity
    for (const b of built) {
      const on = anyShown ? b.shown : true
      if (!on) (b.trace as { visible?: boolean | 'legendonly' }).visible = 'legendonly'
      else {
        const e = b.ext
        if (e.xLo < xLo) xLo = e.xLo
        if (e.xHi > xHi) xHi = e.xHi
        if (e.yLo < yLo) yLo = e.yLo
        if (e.yHi > yHi) yHi = e.yHi
      }
    }
    const finite = Number.isFinite(xLo) && Number.isFinite(yLo)
    return {
      data: built.map((b) => b.trace),
      bounds: finite ? { xLo, xHi, yLo, yHi } : null,
    }
  }, [lightcurve, allowed, shownSurveys, shownBands, clickable])

  const onPointClick = useCallback((p: PlotDatum) => {
    const meta = (p.data as { meta?: TraceMeta }).meta
    if (!meta || !onEpochClick || !clickable.has(meta.survey)) return
    const i = p.pointIndex
    onEpochClick({ survey: meta.survey, band: meta.band, mjd: Number(p.customdata),
                   mag: Number(p.y), magerr: meta.magerr[i] ?? null,
                   upper: Boolean(meta.upper[i]) })
  }, [onEpochClick, clickable])

  const layout = useMemo<Partial<Layout>>(() => {
    // Explicit ranges, not `autorange`. Autorange makes Plotly recompute the
    // extent and relayout on *every* legend toggle, which is the dominant
    // cost of hiding a band. A descending y range also reverses the axis --
    // brighter is up -- without asking for autorange at all.
    //
    // Fixing the range has a second benefit: the axes no longer jump when a
    // trace is hidden, so what you are comparing stays put.
    const pad = (lo: number, hi: number) => (hi - lo) * 0.05 || 0.5
    const yPad = bounds ? pad(bounds.yLo, bounds.yHi) : 0
    const xPad = bounds ? pad(bounds.xLo, bounds.xHi) : 0
    return {
      height: HEIGHT,
      yaxis: {
        title: { text: yAxisTitle },
        ...(bounds
          ? { range: [bounds.yHi + yPad, bounds.yLo - yPad], autorange: false }
          : { autorange: 'reversed' as const }),
      },
      xaxis: {
        title: { text: 'Year' },
        type: 'date' as const,
        tickformatstops: DATE_TICKFORMATS,
        ...(bounds
          ? { range: [bounds.xLo - xPad, bounds.xHi + xPad], autorange: false }
          : {}),
      },
      // `t: 0` is a floor that autoMargin grows to the legend height; see
      // BASE_LAYOUT. The panel title is a React heading above the chart, not
      // a Plotly title, so it cannot compete for the same margin.
      margin: { l: 62, r: MODEBAR_GUTTER, t: 0, b: 46, autoexpand: true },
    }
  }, [bounds, yAxisTitle])

  // With nothing to plot, still draw the axes and put the message inside
  // them. An empty framed panel reads as "we looked and found nothing",
  // where a bare line of text reads as "this did not load".
  const emptyLayout = useMemo<Partial<Layout>>(() => ({
    height: HEIGHT,
    showlegend: false,
    // Left as a plain numeric axis: with no epochs there is nothing to date,
    // and a `type: 'date'` axis over [0, 1] would sit in January 1970.
    xaxis: { title: { text: 'Year' }, range: [0, 1],
             showticklabels: false, zeroline: false },
    yaxis: { title: { text: yAxisTitle }, range: [1, 0],
             showticklabels: false, zeroline: false },
    margin: { l: 62, r: MODEBAR_GUTTER, t: 10, b: 46, autoexpand: true },
    annotations: [{
      text: lightcurve ? 'No epochs in these surveys.' : 'No light curve.',
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      showarrow: false, font: { size: 13, color: '#5d6673' },
    }],
  }), [yAxisTitle, lightcurve])

  const noData = !lightcurve || data.length === 0

  return (
    <div>
      {title && (
        <div style={{ font: '600 12px system-ui', opacity: 0.75,
                      margin: '2px 0 2px 62px' }}>
          {title}
        </div>
      )}
      <PlotlyChart data={noData ? EMPTY : data}
                   layout={noData ? emptyLayout : layout} style={STYLE}
                   onPointClick={onEpochClick ? onPointClick : undefined} />
    </div>
  )
}
