/** ztforce for the current source: its place in the background queue, its
 * result, and the control for binning it.
 *
 * In `components/` rather than `charts/`: it draws no plot. The photometry it
 * produces is handed to `LightCurveChart`, which is what put this beside the
 * charts to begin with.
 *
 * ztforce measures flux at a fixed position in every ZTF science image, so it
 * recovers a curve whether or not anything was ever *detected* there. That
 * takes minutes, so runs go through a queue on the server: labelling a source
 * with the queue's label (normally "Run ztforce") queues it, and the button
 * here queues it at the front. Finished results are stored, so they load at
 * once; while a source is queued or running this polls for its state.
 *
 * The binning window is a display control, not a parameter of the photometry:
 * the API returns ztforce's stack for every window it offers in one payload,
 * so changing the window is instant and never re-runs anything.
 *
 * The parent mounts this with `key={sourceUuid}`, so moving to another source
 * unmounts it -- cancelling any poll in flight -- and remounts it clean. That
 * is why there is no reset-on-source-change effect here.
 */
import { useCallback, useEffect, useState } from 'react'
import { getZtforce, getZtforceQueue, isAbort, runZtforce, setZtforcePaused } from '../api/client'
import type { ZtforceQueue, ZtforceResult } from '../api/types'

/** Matches `Light_Curve._bin_stack`'s default window. The windows on offer
 *  come from the API; 0 is "every epoch", which is what you want when checking
 *  whether a feature is one bad night or a real rise. */
export const DEFAULT_BIN_WINDOW = 365

const POLL_MS = 4000

/** How often the elapsed-time readout redraws. Finer than a second so the
 *  count steps by one; it is interpolated locally between polls. */
const CLOCK_MS = 250

export interface ZtforcePanelProps {
  sourceUuid: string | null
  /** The run's payload, lifted to the parent, which owns the merge into the chart. */
  onResult: (result: ZtforceResult | null) => void
  window: number
  onWindow: (w: number) => void
}

const labelFor = (w: number) =>
  w === 0 ? 'unbinned' : w >= 365 && w % 365 === 0 ? `${w / 365} yr` : `${w} d`

const duration = (s: number) =>
  s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`

const active = (r: ZtforceResult | null) => r?.status === 'queued' || r?.status === 'running'

export function ZtforcePanel({ sourceUuid, onResult, window: binWindow,
                              onWindow }: ZtforcePanelProps) {
  const [result, setResult] = useState<ZtforceResult | null>(null)
  const [queue, setQueue] = useState<ZtforceQueue | null>(null)
  const [err, setErr] = useState<string | null>(null)
  // Wall-clock time of the last payload, and a clock ticking while running: the
  // readout is the server's `elapsed_s` plus the time since that payload, so
  // it counts every second instead of jumping by POLL_MS.
  const [syncedAt, setSyncedAt] = useState(0)
  const [now, setNow] = useState(0)

  const running = result?.status === 'running'
  const polling = active(result)

  /** Shared landing for a payload, from the first load, a poll, or a POST. */
  const settle = useCallback((next: ZtforceResult) => {
    setResult(next)
    const t = Date.now()
    setSyncedAt(t)
    setNow(t)
    onResult(next.stacks ? next : null)
  }, [onResult])

  const refresh = useCallback(async (signal?: AbortSignal) => {
    if (!sourceUuid) return
    const [r, q] = await Promise.all([getZtforce(sourceUuid, signal), getZtforceQueue(signal)])
    settle(r)
    setQueue(q)
    return r
  }, [sourceUuid, settle])

  const fail = useCallback((e: unknown) => {
    if (!isAbort(e)) setErr(e instanceof Error ? e.message : String(e))
  }, [])

  // This source's state, once.
  useEffect(() => {
    if (!sourceUuid) return
    const ac = new AbortController()
    void refresh(ac.signal).catch(fail)
    return () => ac.abort()
  }, [sourceUuid, refresh, fail])

  // Then keep asking while it is queued or running. Separate from the load
  // above, and starting a full interval late, because the two were one effect
  // keyed on `polling`: opening a source that was already running loaded it,
  // flipped `polling` true, and immediately re-ran the whole effect -- two
  // identical round trips back to back, the first of them thrown away.
  // "Run now" still starts the loop, since its POST leaves the source queued.
  useEffect(() => {
    if (!polling) return
    const ac = new AbortController()
    let handle = 0
    const tick = () => {
      void refresh(ac.signal)
        .then((r) => {
          if (active(r ?? null)) handle = globalThis.setTimeout(tick, POLL_MS)
        })
        .catch(fail)
    }
    handle = globalThis.setTimeout(tick, POLL_MS)
    return () => { ac.abort(); globalThis.clearTimeout(handle) }
  }, [polling, refresh, fail])

  useEffect(() => {
    if (!running) return
    const id = globalThis.setInterval(() => setNow(Date.now()), CLOCK_MS)
    return () => globalThis.clearInterval(id)
  }, [running])

  const runNow = useCallback(() => {
    if (!sourceUuid) return
    setErr(null)
    void (async () => {
      try {
        settle(await runZtforce(sourceUuid))
        setQueue(await getZtforceQueue())
      } catch (e) {
        if (!isAbort(e)) setErr(e instanceof Error ? e.message : String(e))
      }
    })()
  }, [sourceUuid, settle])

  const togglePause = useCallback(() => {
    if (!queue) return
    void setZtforcePaused(!queue.paused).then(setQueue).catch((e) => setErr(String(e)))
  }, [queue])

  const status = result?.status ?? 'none'
  const windows = result?.windows ?? []
  const hasStacks = !!result?.stacks
  const nBands = hasStacks
    ? new Set(Object.values(result?.stacks ?? {}).flatMap((w) => Object.keys(w))).size
    : 0
  const muted = { opacity: 0.7 }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10,
                  flexWrap: 'wrap', margin: '2px 0 8px 62px', font: '12px system-ui' }}>
      {status === 'none' && (
        <>
          <button onClick={runNow} disabled={!sourceUuid} style={buttonStyle}>Run ztforce</button>
          {queue && <span style={muted}>or put it in “{queue.label}”</span>}
        </>
      )}

      {status === 'queued' && (
        <>
          <span style={muted}>
            ztforce queued
            {result?.queue_position != null && ` · #${result.queue_position + 1}`}
            {result?.eta_s != null && ` · ~${duration(result.eta_s)}`}
            {queue?.paused && ' · queue paused'}
          </span>
          {(result?.queue_position ?? 0) > 0 && (
            <button onClick={runNow} style={buttonStyle}>Run now</button>
          )}
        </>
      )}

      {status === 'running' && (
        <span style={muted}>
          ztforce running · {duration((result?.elapsed_s ?? 0) + Math.max(0, now - syncedAt) / 1000)}
        </span>
      )}

      {status === 'error' && (
        <>
          <span style={{ color: '#b4232a' }}>ztforce failed: {result?.error}</span>
          <button onClick={runNow} style={buttonStyle}>Retry</button>
        </>
      )}

      {hasStacks && (
        <>
          <span style={{ opacity: 0.75 }}>
            {result?.n_epochs ?? 0} epochs · {nBands} band{nBands === 1 ? '' : 's'}
            {result?.stale && ' · from an older ztforce, re-run queued'}
          </span>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            Bin
            <select value={binWindow}
                    onChange={(e) => onWindow(Number(e.target.value))}
                    style={{ font: 'inherit' }}>
              {windows.map((w) => (
                <option key={w} value={w}>{labelFor(w)}</option>
              ))}
            </select>
          </label>
        </>
      )}

      {status === 'done' && nBands === 0 && <span style={muted}>no usable epochs</span>}

      {queue && (queue.n_queued > 0 || queue.n_running > 0 || queue.paused) && (
        <span style={{ ...muted, marginLeft: 'auto' }}>
          queue: {queue.n_queued} queued · {queue.n_running} running
          {queue.eta_s != null && ` · ~${duration(queue.eta_s)} left`}{' '}
          <button onClick={togglePause} style={{ ...buttonStyle, padding: '1px 6px' }}>
            {queue.paused ? 'Resume' : 'Pause'}
          </button>
        </span>
      )}

      {err && <span style={{ color: '#b4232a' }}>{err}</span>}
    </div>
  )
}

const buttonStyle = { font: 'inherit', cursor: 'pointer', padding: '4px 10px', borderRadius: 5 } as const
