/** Postage stamps: ZTF on the top row, PanSTARRS below, g/r/i left to right.
 *
 * The API renders these from FITS with the same log/99.5 viridis stretch the
 * analysis code uses (`source_db.cutouts.render`), because the ready-made
 * JPEGs the services hand out bury a faint source in the noise. It still
 * returns *addresses*: the six stamps download in parallel on the browser's
 * own connection pool and land in its cache, and the server caches the
 * rendered PNG on disk so the FITS fetch is paid once.
 *
 * `usePrefetchCache` warms that cache for the *next* source while the current
 * one is on screen, which is what makes paging feel instant: by the time you
 * press Next the images are already in the browser's HTTP cache and the
 * `<img>` resolves from memory.
 *
 * The reticle and scale bar are DOM overlays rather than pixels baked into
 * the PNG. That keeps one render per (source, band, stretch) in the server
 * cache instead of one per annotation choice, and the marks stay crisp when
 * a stamp is blown up to full screen.
 */
import { Fragment, useCallback, useEffect, useRef, useState } from 'react'

const CUTOUT_BANDS = ['g', 'r', 'i'] as const

/** survey -> band -> image URL */
export type CutoutImages = Record<string, Record<string, string>>

export interface CutoutGridProps {
  images: CutoutImages | null
  /** Shown while the addresses are still being fetched. */
  loading?: boolean
  /** Flat catalog row, for the PSF/Kron magnitudes beside each stamp. */
  catalog?: Record<string, unknown> | null
  /** survey -> arcsec per pixel, from the API. */
  pixelArcsec?: Record<string, number> | null
  stretch?: string
  onStretch?: (s: string) => void
  percent?: number
  onPercent?: (p: number) => void
  /** Links to the surveys' own image viewers, rendered beside the controls:
   *  "show me more of this image" belongs with the images. */
  links?: { label: string; url: string; help?: string }[]
}

const STRETCHES = ['linear', 'sqrt', 'log', 'asinh']
const PERCENTS = [95, 99, 99.5, 99.9, 100]

// `prefix` is the catalog column prefix: `ZTF_gPSFMag`, `PSTARR_rKronMagErr`.
// PanSTARRS first: it is the deeper, higher-resolution reference, so it is
// the one you read the field off before looking at what ZTF saw.
const ROWS: { survey: string; label: string; prefix: string }[] = [
  { survey: 'panstarrs', label: 'PanSTARRS', prefix: 'PSTARR' },
  { survey: 'ztf', label: 'ZTF', prefix: 'ZTF' },
]

// The magnitude column is a fixed width so that every band cell is the same
// size and the stamps line up vertically between the two rows. Sizing it to
// its content instead let a wider "Kron 20.37 ± 0.09" in one row push that
// row's stamps out of alignment with the other's.
const MAG_COL_W = 116

// The 5-sigma depth each row is measured against. PanSTARRS has no per-source
// limit to read -- these are the survey's stack depths, the same constants
// `PSTARR_UPPER_LIM` substitutes in `filter_fields.only_big_dmag` and in the
// API's `_mean_abs_dmag` -- while ZTF carries its own per-image MAGLIM. Both
// are shown because a non-detection is otherwise a bare dash, which makes the
// table's mean |dmag| impossible to reconstruct: for a ZTF-only source that
// column is |ZTF - 23.3| and nothing on this page said where 23.3 came from.
const PS1_5SIGMA: Record<string, number> = { g: 23.3, r: 23.2, i: 23.1 }

// The two surveys do not cover the same angle -- PanSTARRS is 240 px at
// 0.25" (60.0") and ZTF is 61 px at 1.0" (61.0") -- so the bar is sized from
// each stamp's own pixel count rather than as a fixed fraction of the frame,
// which would read 10.17" on ZTF.
const SCALE_BAR_ARCSEC = 10

// `plot_cutout` draws two short scatter markers -- a '_' offset 3% of the
// image height to the left of the source and a '|' the same distance below
// it -- so they bracket the source without covering it. Matplotlib's default
// marker is 6 pt, which is about 4% of a stamp this size; drawing them any
// longer makes the two ticks meet and read as one 'T' sitting off-centre,
// which is what they looked like before.
const RETICLE_OFFSET_PCT = 3
const RETICLE_LEN_PCT = 4

/** What the hover preview is currently showing. */
interface Magnified {
  url: string
  band: string
  survey: string
  arcsecPerPixel: number | null
  psf: Mag
  kron: Mag
}

export function CutoutGrid({ images, loading, catalog, pixelArcsec,
                            stretch, onStretch, percent, links,
                            onPercent }: CutoutGridProps) {
  // Which stamp the enlarge dialog is showing, if any. Enlarging one at a
  // time beats a full-screen view of all six: the thing you want a closer
  // look at is a single band, with the other five still there to compare.
  const [enlarged, setEnlarged] = useState<Magnified | null>(null)

  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center',
                    fontSize: 12, opacity: 0.85 }}>
        {onStretch && onPercent && (
          <>
            <span style={{ opacity: 0.7 }}>Stretch</span>
            <select value={stretch} onChange={(e) => onStretch(e.target.value)}
                    style={{ font: 'inherit' }}>
              {STRETCHES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select value={percent}
                    onChange={(e) => onPercent(Number(e.target.value))}
                    style={{ font: 'inherit' }}>
              {PERCENTS.map((p) => <option key={p} value={p}>{p}%</option>)}
            </select>
          </>
        )}
        {/* Pushed right so the stretch controls stay where the eye expects
            them and the archive links read as a separate action. */}
        {links && links.length > 0 && (
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {links.map((l) => (
              <a key={l.label} href={l.url} target="_blank" rel="noreferrer"
                 title={l.help}
                 style={{ padding: '2px 8px', borderRadius: 5, fontSize: 12,
                          border: '1px solid var(--border-strong)',
                          textDecoration: 'none', color: 'inherit' }}>
                {l.label}
              </a>
            ))}
          </span>
        )}
      </div>

      {/* Both survey rows share one grid rather than being a flex row each,
          which is what keeps the g/r/i stamps in the two rows in the same
          three columns. */}
      {/* `1fr` stamp columns rather than a fixed pixel size, so the stamps
          grow to fill whatever width the card has; `minmax(0, …)` stops a
          wide image from forcing the track past its share. The magnitude
          columns stay fixed, which is what keeps the two rows aligned. */}
      <div style={{ display: 'grid',
                    gridTemplateColumns:
                      `auto repeat(${CUTOUT_BANDS.length}, minmax(0, 1fr) ${MAG_COL_W}px)`,
                    gap: '10px 8px', alignItems: 'start' }}>
        {ROWS.map((row) => (
          <Fragment key={row.survey}>
            <div style={{ font: '600 11px system-ui', opacity: 0.7,
                          writingMode: 'vertical-rl', transform: 'rotate(180deg)',
                          textAlign: 'center', letterSpacing: 0.4,
                          alignSelf: 'stretch', marginRight: 6 }}>
              {row.label}
            </div>
            {CUTOUT_BANDS.map((band) => (
              <BandCell key={band} band={band}
                        url={images?.[row.survey]?.[band] ?? null}
                        loading={loading}
                        arcsecPerPixel={pixelArcsec?.[row.survey] ?? null}
                        survey={row.label}
                        psf={mag(catalog, row.prefix, band, 'PSF')}
                        kron={mag(catalog, row.prefix, band, 'Kron')}
                        limit={limitFor(catalog, row.prefix, band)}
                        onEnlarge={setEnlarged} />
            ))}
          </Fragment>
        ))}
      </div>
      {enlarged && (
        <EnlargeDialog shot={enlarged} onClose={() => setEnlarged(null)} />
      )}
    </div>
  )
}

/** One stamp, its band label underneath, and its magnitudes to the right. */
function BandCell({ url, band, loading, psf, kron, limit, arcsecPerPixel,
                    survey, onEnlarge }:
                  { url: string | null; band: string
                    loading?: boolean; psf: Mag; kron: Mag
                    limit: number | null
                    arcsecPerPixel: number | null; survey: string
                    onEnlarge: (m: Magnified) => void }) {
  return (
    <>
      <div>
        {/* Reveal is CSS `:hover`, not React state. State got stuck on: while
            the modal is open the rest of the page is inert, so the stamp
            never receives the `mouseleave` and its button stayed visible
            after the dialog closed. The selector cannot desynchronise.
            Positioning context is the stamp alone, so the band caption
            below does not shift where the button sits. */}
        <div className="cutout-frame" style={{ position: 'relative' }}>
          <Cutout url={url} band={band} loading={loading}
                  arcsecPerPixel={arcsecPerPixel} />
          {url && (
            <button
              className="cutout-enlarge"
              onClick={() => onEnlarge({ url, band, survey, arcsecPerPixel, psf, kron })}
              aria-label={`Enlarge ${survey} ${band}`}
              style={{ position: 'absolute', right: 6, bottom: 6, padding: '2px 8px',
                       fontSize: 11, borderRadius: 4, lineHeight: 1.6,
                       background: 'rgba(255,255,255,0.92)' }}>
              Enlarge
            </button>
          )}
        </div>
        {/* Where an x-axis label would sit. */}
        <div style={{ textAlign: 'center', marginTop: 3,
                      font: '600 12px system-ui', opacity: 0.8 }}>
          {band}
        </div>
      </div>
      {/* `alignSelf: stretch` overrides the grid's `alignItems: start` so this
          cell spans the row, which is what lets `marginTop: auto` drop the
          limit to the foot of the stamp rather than leaving it tucked under
          the Kron line. */}
      <div style={{ fontSize: 11, lineHeight: 1.45, opacity: 0.85,
                    paddingTop: 2, whiteSpace: 'nowrap',
                    alignSelf: 'stretch',
                    display: 'flex', flexDirection: 'column' }}>
        <MagBlock label="PSF" m={psf} />
        <MagBlock label="Kron" m={kron} style={{ marginTop: 5 }} />
        {/* Dimmer than the two measurements above: this is the depth the
            stamp reaches, not something measured at this position. */}
        <div style={{ marginTop: 'auto', paddingTop: 6, opacity: 0.6,
                      fontVariantNumeric: 'tabular-nums' }}>
          <span style={{ opacity: 0.6 }}>5&sigma; </span>
          {limit === null ? '—' : `> ${limit.toFixed(2)}`}
        </div>
      </div>
    </>
  )
}

/** One stamp enlarged, in a native modal `<dialog>`.
 *
 * `showModal()` rather than a hand-rolled overlay: the platform then gives us
 * the backdrop, Escape-to-close, focus containment, and rendering in the top
 * layer -- so it cannot be trapped behind a stacking context and needs no
 * z-index at all. `onClose` fires however it was dismissed, including Escape,
 * which is why the parent's state is cleared from there rather than from the
 * close button.
 *
 * Backdrop clicks are wired up by hand because the attribute that does it
 * natively, `closedby="any"`, is not Baseline yet.
 */
function EnlargeDialog({ shot, onClose }:
                       { shot: Magnified; onClose: () => void }) {
  const { url, band, survey, arcsecPerPixel, psf, kron } = shot
  const ref = useRef<HTMLDialogElement | null>(null)
  const [naturalPx, setNaturalPx] = useState(0)

  // Opening has to happen after the element exists; `showModal` on an
  // already-open dialog throws, hence the guard.
  useEffect(() => {
    const d = ref.current
    if (d && !d.open) d.showModal()
  }, [])

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      // With `padding: 0` the dialog box is exactly its content, so a click
      // reported against the dialog itself can only have been on the backdrop.
      onClick={(e) => { if (e.target === ref.current) ref.current?.close() }}
      style={{ padding: 0, border: '1px solid var(--border)', borderRadius: 8,
               background: 'var(--bg)', color: 'inherit', maxWidth: '96vw',
               maxHeight: '96vh' }}>
      <div style={{ padding: 12 }}>
        <div style={{ position: 'relative',
                      // vmin keeps the stamp square and inside the viewport
                      // on any window shape without measuring anything.
                      width: 'min(78vmin, 760px)', height: 'min(78vmin, 760px)',
                      overflow: 'hidden', borderRadius: 4,
                      background: 'var(--bg-soft)' }}>
          <img src={url} alt={`${band} band cutout, enlarged`}
               onLoad={(e) => setNaturalPx((e.target as HTMLImageElement).naturalWidth)}
               ref={(img) => { if (img?.naturalWidth) setNaturalPx(img.naturalWidth) }}
               style={{ display: 'block', width: '100%', height: '100%',
                        objectFit: 'cover', imageRendering: 'pixelated' }} />
          <Reticle />
          {arcsecPerPixel !== null && naturalPx > 0 && (
            <ScaleBar fovArcsec={naturalPx * arcsecPerPixel} />
          )}
        </div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'baseline',
                      marginTop: 8, fontSize: 13 }}>
          <b>{survey} {band}</b>
          <span style={{ opacity: 0.85 }}>
            <span style={{ opacity: 0.6 }}>PSF </span>{fmt(psf)}
          </span>
          <span style={{ opacity: 0.85 }}>
            <span style={{ opacity: 0.6 }}>Kron </span>{fmt(kron)}
          </span>
          {naturalPx > 0 && arcsecPerPixel !== null && (
            <span style={{ opacity: 0.5 }}>
              {naturalPx}&nbsp;px &middot; {(naturalPx * arcsecPerPixel).toFixed(0)}&Prime;
            </span>
          )}
          <button autoFocus onClick={() => ref.current?.close()}
                  style={{ marginLeft: 'auto', padding: '3px 10px' }}>
            Close
          </button>
        </div>
      </div>
    </dialog>
  )
}


/** A magnitude with its error and the SNR implied by that error. */
function MagBlock({ label, m, style }:
                  { label: string; m: Mag; style?: React.CSSProperties }) {
  const snr = snrFromMagErr(m?.err ?? null)
  return (
    <div style={style}>
      <div><span style={{ opacity: 0.6 }}>{label} </span>{fmt(m)}</div>
      {snr !== null && (
        <div style={{ opacity: 0.6, paddingLeft: 2 }}>
          SNR {snr.toFixed(2)}
        </div>
      )}
    </div>
  )
}

/** SNR from the magnitude error alone.
 *
 * `Extracting.utils.get_snr_from_mag` converts to flux and divides, but the
 * flux and the zero point both cancel:
 *
 *     fluxerr = magerr * ln(10) * flux / 2.5
 *     SNR = flux / fluxerr = 2.5 / (ln(10) * magerr)
 *
 * Checked against that function across several zero points. So the ZTF
 * per-band zero point the analysis code threads through is not needed here.
 */
function snrFromMagErr(magerr: number | null): number | null {
  if (magerr === null || !(magerr > 0)) return null
  return 2.5 / (Math.LN10 * magerr)
}

type Mag = { value: number; err: number | null } | null

function mag(catalog: Record<string, unknown> | null | undefined,
             prefix: string, band: string, kind: 'PSF' | 'Kron'): Mag {
  if (!catalog) return null
  const v = catalog[`${prefix}_${band}${kind}Mag`]
  if (typeof v !== 'number' || !Number.isFinite(v)) return null
  const e = catalog[`${prefix}_${band}${kind}MagErr`]
  return { value: v, err: typeof e === 'number' && Number.isFinite(e) ? e : null }
}

/** The 5-sigma limiting magnitude for one (survey, band), or null. */
function limitFor(catalog: Record<string, unknown> | null | undefined,
                  prefix: string, band: string): number | null {
  if (prefix === 'PSTARR') return PS1_5SIGMA[band] ?? null
  const v = catalog?.[`ZTF_${band}_mag_limit`]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function fmt(m: Mag): string {
  if (!m) return '—'
  return m.err === null ? m.value.toFixed(2)
                        : `${m.value.toFixed(2)} ± ${m.err.toFixed(2)}`
}

function Cutout({ url, band, loading, arcsecPerPixel }:
                { url: string | null; band: string
                  loading?: boolean; arcsecPerPixel: number | null }) {
  const [state, setState] = useState<'idle' | 'ok' | 'error'>('idle')
  // The stamp's native pixel count, which sets the scale bar's length.
  const [naturalPx, setNaturalPx] = useState(0)

  // Reset *during render* when the URL changes, not in an effect.
  //
  // An effect runs after the browser has had a chance to fire `onLoad`, and
  // for a cached image that happens immediately -- so `setState('ok')` landed
  // first and the effect then reset it to 'idle', leaving a loaded image
  // hidden behind the spinner forever. Prefetching made every image cache-warm
  // and so made this happen on every navigation. Adjusting state directly
  // during render is React's documented way to derive state from props.
  const [seenUrl, setSeenUrl] = useState(url)
  if (seenUrl !== url) {
    setSeenUrl(url)
    setState('idle')
  }

  // A cached image can also finish before React attaches `onLoad` at all, so
  // the ref checks whether it is already complete on mount.
  const attach = useCallback((img: HTMLImageElement | null) => {
    if (img && img.complete && img.naturalWidth > 0) {
      setState('ok'); setNaturalPx(img.naturalWidth)
    }
  }, [])

  const frame: React.CSSProperties = {
    // Square by declaration rather than by a measured pixel size, so the
    // stamp tracks its grid column and the placeholder reserves the right
    // height before the image has loaded.
    width: '100%', aspectRatio: '1 / 1', position: 'relative',
    border: '1px solid var(--border)', borderRadius: 4,
    background: 'var(--bg-soft)', overflow: 'hidden',
  }

  return (
    <div style={frame}>
      {url && (
        <img
          ref={attach}
          // A fresh element per URL, so a cached swap cannot reuse the old
          // one's completed state.
          key={url}
          src={url}
          alt={`${band} band cutout`}
          // The images are the point of the page, so they are not lazy and
          // are hinted as high priority; `async` decoding keeps a slow
          // decode off the main thread.
          loading="eager"
          decoding="async"
          // React 19 types this natively; it was spread in as a lowercase
          // `fetchpriority` through an `any` cast, which React rejected as
          // an unknown DOM property -- so the hint never reached the image.
          fetchPriority="high"
          onLoad={(e) => {
            setState('ok')
            setNaturalPx((e.target as HTMLImageElement).naturalWidth)
          }}
          onError={() => setState('error')}
          style={{ display: 'block', width: '100%', height: '100%',
                   objectFit: 'cover',
                   // ZTF is 61 px native at this field of view; smoothing it
                   // up to 150 would imply resolution the data does not have.
                   imageRendering: 'pixelated',
                   // Fade in on load rather than flashing a half-drawn image.
                   opacity: state === 'ok' ? 1 : 0,
                   transition: 'opacity 120ms ease-out' }}
        />
      )}
      {state === 'ok' && <Reticle />}
      {state === 'ok' && arcsecPerPixel !== null && naturalPx > 0 && (
        <ScaleBar fovArcsec={naturalPx * arcsecPerPixel} />
      )}
      {(state !== 'ok') && (
        <div style={{ position: 'absolute', inset: 0, display: 'grid',
                      placeItems: 'center', fontSize: 11,
                      color: 'var(--text-dim)' }}>
          {!url ? (loading ? '…' : 'not available')
                : state === 'error' ? 'failed' : '…'}
        </div>
      )}
    </div>
  )
}

/** The red centre marks from `plot_cutout`: a dash to the left of the source
 *  and a bar below it, each offset so neither covers the source itself. */
export function Reticle() {
  const line: React.CSSProperties = { position: 'absolute', background: 'red' }
  return (
    <>
      <div style={{ ...line,
                    left: `${50 - RETICLE_OFFSET_PCT - RETICLE_LEN_PCT / 2}%`,
                    top: '50%', width: `${RETICLE_LEN_PCT}%`, height: 2,
                    transform: 'translateY(-50%)' }} />
      <div style={{ ...line,
                    top: `${50 + RETICLE_OFFSET_PCT - RETICLE_LEN_PCT / 2}%`,
                    left: '50%', height: `${RETICLE_LEN_PCT}%`, width: 2,
                    transform: 'translateX(-50%)' }} />
    </>
  )
}

/** 10" bar in the top-right, as `plot_cutout` draws it.
 *
 * Red rather than matplotlib's black, to match the reticle: the low end of
 * viridis is near-black, so a black bar vanishes into the background on
 * screen in a way it does not in a paper figure.
 */
export function ScaleBar({ fovArcsec }: { fovArcsec: number }) {
  const pct = (SCALE_BAR_ARCSEC / fovArcsec) * 100
  return (
    <>
      <div style={{ position: 'absolute', right: '3%', top: '1%',
                    width: 2, height: `${pct}%`, background: 'red',
                    boxShadow: '0 0 3px rgba(0,0,0,0.9)' }} />
      <div style={{ position: 'absolute', right: '6%',
                    top: `${1 + pct / 2}%`,
                    transform: 'translateY(-50%)',
                    fontSize: 10, color: 'red', whiteSpace: 'nowrap',
                    textShadow: '0 0 3px rgba(0,0,0,0.9)' }}>
        {SCALE_BAR_ARCSEC}&Prime;
      </div>
    </>
  )
}
