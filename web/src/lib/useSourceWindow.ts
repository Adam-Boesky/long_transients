/** A sliding window over the candidate list.
 *
 * The list is ~198k sources and grows every release, so it is not something to
 * hold in the browser: at 65 B/row gzipped the whole thing is ~13 MB per filter
 * change. The server can answer "rows N..N+k of this filter" in tens of
 * milliseconds even at offset 150,000, so the client keeps a window around
 * wherever you are and refills it before you reach an edge.
 *
 * What that buys, against the two things this page has to do well:
 *   stepping   Prev/Next inside the window is a local array read, no request.
 *   filtering  a filter change costs one COUNT and one window (~13 kB), the
 *              same whether the result is 500 rows or 198,000.
 *
 * Positions here are offsets into the *filtered* order, not indices into a
 * local array -- `total` is what the server reports, and `at()` returns
 * undefined for a position not yet loaded rather than pretending it is absent.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { isAbort, listSources } from '../api/client'
import type { SourceQuery, SourceSummary } from '../api/types'
import { useStableValue } from './useStableValue'

/** Rows per request. 200 is ~13 kB gzipped and covers a long burst of arrow
 *  keys without a refill. */
export const WINDOW = 200
/** Refill when within this many rows of an edge, so the fetch lands before
 *  you arrive rather than after. */
const MARGIN = 60

interface WindowState {
  /** Offset of `items[0]` in the filtered order. */
  start: number
  items: SourceSummary[]
}

export interface SourceWindow {
  total: number | null
  /** How many of `total` this author has filed, for the progress readout. */
  labelled: number
  /** The summary at a filtered-order offset, or undefined if not loaded yet. */
  at: (pos: number) => SourceSummary | undefined
  /** True once the first window for the current filter has landed. */
  ready: boolean
  /** Re-fetch the window covering `pos` (after labelling, say). */
  refresh: () => void
}

export function useSourceWindow(query: SourceQuery, position: number): SourceWindow {
  const [total, setTotal] = useState<number | null>(null)
  const [labelled, setLabelled] = useState(0)
  const [win, setWin] = useState<WindowState | null>(null)
  const [ready, setReady] = useState(false)
  // Compared by value: the caller rebuilds the query object every render, and
  // depending on the object itself would refetch whenever anything else on
  // the page re-renders.
  const stable = useStableValue(query)
  const inflight = useRef<AbortController | null>(null)
  const loadedFor = useRef<string>('')

  // A filter change invalidates the window entirely -- offsets mean something
  // different now -- so drop it rather than showing rows from the old order.
  useEffect(() => {
    setWin(null)
    setTotal(null)
    setLabelled(0)
    setReady(false)
    loadedFor.current = ''
  }, [stable])

  const fetchAround = useCallback((pos: number) => {
    const start = Math.max(0, pos - Math.floor(WINDOW / 2))
    const tag = `${stable.catalog}|${stable.q}|${stable.filter}@${start}`
    if (loadedFor.current === tag) return
    loadedFor.current = tag
    inflight.current?.abort()
    const ac = new AbortController()
    inflight.current = ac
    listSources({ ...stable, offset: start, limit: WINDOW }, ac.signal)
      .then((page) => {
        setTotal(page.total)
        setLabelled(page.n_labelled)
        setWin({ start: page.offset, items: page.items })
        setReady(true)
      })
      .catch((e) => { if (!isAbort(e)) console.error(e) })
  }, [stable])

  useEffect(() => {
    if (!win) { fetchAround(position); return }
    const lo = win.start
    const hi = win.start + win.items.length
    const nearStart = position - lo < MARGIN && lo > 0
    const nearEnd = hi - position < MARGIN && (total === null || hi < total)
    if (position < lo || position >= hi || nearStart || nearEnd) fetchAround(position)
  }, [position, win, total, fetchAround])

  const at = useCallback((pos: number) => {
    if (!win) return undefined
    const i = pos - win.start
    return i >= 0 && i < win.items.length ? win.items[i] : undefined
  }, [win])

  const refresh = useCallback(() => {
    loadedFor.current = ''
    fetchAround(position)
  }, [fetchAround, position])

  // Memoised: a bare object literal is a new identity every render, and
  // callers put this in effect deps -- which made the prefetch effect in
  // ClassifyPage re-run on every keystroke in the comment box.
  return useMemo(() => ({ total, labelled, at, ready, refresh }),
                 [total, labelled, at, ready, refresh])
}
