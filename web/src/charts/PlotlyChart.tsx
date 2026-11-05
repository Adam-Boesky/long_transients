/** plotly.js bound to a React element, following the pattern the maintainers
 *  recommend for hooks (plotly/react-plotly.js#242).
 *
 * Four things here are deliberate rather than incidental:
 *
 *   useLayoutEffect  the draw happens before paint, so a chart never shows a
 *                    frame of stale data while flipping between sources.
 *   callback ref     `useState` rather than `useRef`, so attaching the node
 *                    re-runs the effect. A `useRef` does not notify React,
 *                    which makes remounts silently draw nothing.
 *   Plotly.react     reconciles against what is already drawn instead of
 *                    tearing down, which preserves zoom and -- more
 *                    importantly -- the WebGL context.
 *   Plotly.purge     in cleanup. This is what releases that context; leaking
 *                    them is how a page hits the browser's ~8-context ceiling
 *                    and charts silently stop rendering.
 *
 * `data`, `layout` and `config` are dependencies, so they must be stable
 * references. Pass memoised objects; an inline literal redraws every render.
 *
 * Resizing is left to Plotly's own `responsive: true`, which listens for
 * window resize. An earlier version also ran a `ResizeObserver` on the graph
 * div to catch *container* resizes -- that was a mistake twice over:
 * `Plots.resize` is a full relayout and redraw, and observing the element it
 * writes back to is a feedback loop. Toggling a legend entry reflows the
 * legend, which changes content height, which can toggle a page scrollbar,
 * which changes the observed width -- and the chart redrew until the tab
 * locked up. If container resizing is ever needed, it should observe an
 * outer element and be measured before it goes in.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Config, Data, Layout, PlotDatum, PlotMouseEvent,
              PlotlyHTMLElement } from 'plotly.js'
import Plotly, { BASE_CONFIG, BASE_LAYOUT, LEGEND_GAP } from './plotly'

/** Detach listeners, tolerating an already-purged node.
 *
 * On unmount React destroys layout effects before passive ones, so
 * `Plotly.purge` below has already run and taken the div's EventEmitter
 * methods with it -- calling `removeAllListeners` then throws
 * "e.removeAllListeners is not a function" *during* the unmount commit,
 * which unwinds the whole React tree and blanks the page. Purge has already
 * removed the listeners, so skipping is the correct thing to do.
 */
function offAll(el: PlotlyHTMLElement, event: string) {
  if (typeof el.removeAllListeners === 'function') el.removeAllListeners(event)
}

export interface PlotlyChartProps {
  data: Data[]
  layout?: Partial<Layout>
  config?: Partial<Config>
  style?: React.CSSProperties
  /** Called with the first point under a click. */
  onPointClick?: (point: PlotDatum) => void
}

export function PlotlyChart({ data, layout, config, style, onPointClick }: PlotlyChartProps) {
  const [node, setNode] = useState<HTMLDivElement | null>(null)

  // Held in a ref so a new handler each render does not re-bind the listener.
  const clickRef = useRef(onPointClick)
  useEffect(() => { clickRef.current = onPointClick }, [onPointClick])
  useEffect(() => {
    if (!node) return
    const el = node as unknown as PlotlyHTMLElement
    el.on('plotly_click', (e: PlotMouseEvent) => {
      const p = e.points?.[0]
      if (p && clickRef.current) clickRef.current(p)
    })
    return () => { offAll(el, 'plotly_click') }
  }, [node, data, layout])

  useLayoutEffect(() => {
    if (!node) return
    const merged: Partial<Layout> = {
      ...BASE_LAYOUT,
      ...layout,
      // Merge the axis `title` objects too, so a caller supplying only
      // `title.text` keeps the shared bold font rather than dropping it.
      xaxis: {
        ...BASE_LAYOUT.xaxis, ...(layout?.xaxis ?? {}),
        title: { ...BASE_LAYOUT.xaxis?.title, ...(layout?.xaxis?.title ?? {}) },
      },
      yaxis: {
        ...BASE_LAYOUT.yaxis, ...(layout?.yaxis ?? {}),
        title: { ...BASE_LAYOUT.yaxis?.title, ...(layout?.yaxis?.title ?? {}) },
      },
    }
    void Plotly.react(node, data, merged, { ...BASE_CONFIG, ...config })
    return () => { Plotly.purge(node) }
  }, [node, data, layout, config])

  // Fit the top margin to the legend.
  //
  // Plotly registers the right push (`_pushmargin.legend.t = legend height`)
  // but `doAutoMargin` does not apply it for a full-width horizontal legend
  // anchored at y=1 -- the same legend also pushes `r` by its whole width,
  // and that gets clamped against available space. The result is either a
  // hard-coded default of 100px of dead air, or an overlap at `t: 0`.
  //
  // So take the height Plotly measured and set the margin to it. One
  // relayout after the first draw, and only when the value actually differs
  // by more than a pixel -- `relayout` re-runs this handler, so an
  // unconditional call is a redraw loop.
  useEffect(() => {
    if (!node) return
    type Margin = { l?: number; r?: number; t?: number; b?: number
                    autoexpand?: boolean }
    const el = node as unknown as PlotlyHTMLElement & {
      _fullLayout?: { legend?: { _effHeight?: number }; margin?: Margin }
    }
    const fit = () => {
      const h = el._fullLayout?.legend?._effHeight
      const margin = el._fullLayout?.margin
      if (typeof h !== 'number' || !margin || typeof margin.t !== 'number') return
      const wanted = Math.ceil(h) + LEGEND_GAP
      if (Math.abs(margin.t - wanted) <= 1) return
      // The whole margin, not a dotted key: `relayout` replaces the object
      // it is given, so the other three sides have to come along.
      void Plotly.relayout(node, {
        margin: { l: margin.l, r: margin.r, b: margin.b,
                  autoexpand: margin.autoexpand, t: wanted },
      })
    }
    el.on('plotly_afterplot', fit)
    fit()
    return () => { offAll(el, 'plotly_afterplot') }
  }, [node, data, layout])

  return <div ref={setNode}
              style={{ width: '100%', height: '100%', ...style }} />
}
