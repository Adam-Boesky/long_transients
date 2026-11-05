/** The exposure behind one light-curve point, beside the reference stamp.
 *
 * A native modal `<dialog>`, like the stamp enlarger in `CutoutGrid`, so the
 * backdrop, Escape and focus handling come from the platform. The two images
 * share a grid (the API resamples the epoch onto the reference's pixels), so
 * the reticle marks the same sky in both and they can be compared directly.
 */
import { useEffect, useRef, useState } from 'react'
import { getEpochImage, isAbort } from '../api/client'
import type { EpochImage } from '../api/types'
import type { EpochClick } from '../charts/LightCurveChart'
import { Reticle, ScaleBar } from './CutoutGrid'

const SURVEY_LABEL: Record<string, string> = { ztf: 'ZTF', panstarrs: 'PanSTARRS' }
const REFERENCE_LABEL: Record<string, string> = {
  ztf: 'ZTF deep reference', panstarrs: 'PanSTARRS stack',
}
const MJD_UNIX_EPOCH = 40587

export interface EpochImageDialogProps {
  sourceUuid: string
  epoch: EpochClick & { objectid?: string | null }
  /** The reference stamp for this survey and band, if the source has one. */
  referenceUrl: string | null
  referencePixelArcsec: number | null
  stretch: string
  percent: number
  onClose: () => void
}

export function EpochImageDialog({ sourceUuid, epoch, referenceUrl,
                                   referencePixelArcsec, stretch, percent,
                                   onClose }: EpochImageDialogProps) {
  const ref = useRef<HTMLDialogElement | null>(null)
  const downOnBackdrop = useRef(false)
  const [info, setInfo] = useState<EpochImage | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const d = ref.current
    if (d && !d.open) d.showModal()
  }, [])

  useEffect(() => {
    const ac = new AbortController()
    setInfo(null); setErr(null)
    getEpochImage(sourceUuid, epoch.survey, epoch.band, epoch.mjd,
                  { objectid: epoch.objectid, stretch, percent }, ac.signal)
      .then(setInfo)
      .catch((e) => { if (!isAbort(e)) setErr(e instanceof Error ? e.message : String(e)) })
    return () => ac.abort()
  }, [sourceUuid, epoch, stretch, percent])

  const survey = SURVEY_LABEL[epoch.survey] ?? epoch.survey
  const date = new Date((epoch.mjd - MJD_UNIX_EPOCH) * 86400000).toISOString().slice(0, 10)
  const magText = epoch.upper ? `< ${epoch.mag.toFixed(2)} (upper limit)`
    : `${epoch.mag.toFixed(2)}${epoch.magerr !== null ? ` ± ${epoch.magerr.toFixed(2)}` : ''}`

  return (
    // Closes on a backdrop click only if the press also started there, so the
    // light-curve click that opened the panel cannot land on it and close it.
    <dialog ref={ref} onClose={onClose}
            onPointerDown={(e) => { downOnBackdrop.current = e.target === ref.current }}
            onClick={(e) => {
              if (downOnBackdrop.current && e.target === ref.current) ref.current?.close()
            }}
            style={{ padding: 0, border: '1px solid var(--border)', borderRadius: 8,
                     background: 'var(--bg)', color: 'inherit', maxWidth: '96vw',
                     maxHeight: '96vh' }}>
      <div style={{ padding: 12, font: '13px system-ui' }}>
        <div style={{ display: 'flex', gap: 14, alignItems: 'baseline', marginBottom: 8 }}>
          <b>{survey} {epoch.band}</b>
          <span>{date} <span style={{ opacity: 0.6 }}>(MJD {epoch.mjd.toFixed(4)})</span></span>
          <span>{magText}</span>
          <button autoFocus onClick={() => ref.current?.close()}
                  style={{ marginLeft: 'auto', padding: '3px 10px' }}>Close</button>
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          <Panel key={info?.png ?? 'pending'} title="This exposure"
                 url={info?.png ?? null}
                 pixelArcsec={info?.pixel_arcsec ?? null}
                 placeholder={err ?? (epoch.survey === 'ztf'
                   ? 'Finding the exposure at IRSA… (can take ~10 s)'
                   : 'Finding the exposure at STScI…')}
                 isError={err !== null} />
          <Panel title={REFERENCE_LABEL[epoch.survey] ?? 'Reference'}
                 url={referenceUrl} pixelArcsec={referencePixelArcsec}
                 placeholder={`No ${survey} ${epoch.band} reference for this source.`}
                 isError={false} />
        </div>

        {info && (
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'baseline',
                        marginTop: 8, opacity: 0.85 }}>
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
              {info.image_id}
            </span>
            {Object.entries(info.details).filter(([, v]) => v !== null).map(([k, v]) => (
              <span key={k}><span style={{ opacity: 0.6 }}>{k} </span>
                {typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(2) : v}</span>
            ))}
            <a href={info.fits_url} target="_blank" rel="noreferrer" style={{ marginLeft: 'auto' }}>
              FITS cutout</a>
            <a href={info.full_url} target="_blank" rel="noreferrer">
              {epoch.survey === 'ztf' ? 'Full image (FITS)' : 'Open at STScI'}</a>
          </div>
        )}
      </div>
    </dialog>
  )
}

function Panel({ title, url, pixelArcsec, placeholder, isError }:
               { title: string; url: string | null; pixelArcsec: number | null
                 placeholder: string; isError: boolean }) {
  const [naturalPx, setNaturalPx] = useState(0)
  const [failed, setFailed] = useState(false)
  const side = 'min(44vmin, 460px)'
  return (
    <div>
      <div style={{ font: '600 12px system-ui', opacity: 0.75, marginBottom: 4 }}>{title}</div>
      <div style={{ position: 'relative', width: side, height: side, overflow: 'hidden',
                    borderRadius: 4, background: 'var(--bg-soft)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {url && !failed ? (
          <>
            <img src={url} alt={title}
                 onLoad={(e) => setNaturalPx((e.target as HTMLImageElement).naturalWidth)}
                 onError={() => setFailed(true)}
                 style={{ position: 'absolute', inset: 0, width: '100%', height: '100%',
                          objectFit: 'cover', imageRendering: 'pixelated' }} />
            {naturalPx === 0 && <span style={{ opacity: 0.55 }}>Loading image…</span>}
            {naturalPx > 0 && <Reticle />}
            {naturalPx > 0 && pixelArcsec !== null && (
              <ScaleBar fovArcsec={naturalPx * pixelArcsec} />
            )}
          </>
        ) : (
          <span style={{ padding: 16, textAlign: 'center', opacity: isError ? 0.9 : 0.55,
                         color: isError ? 'var(--danger, #b42318)' : undefined }}>
            {failed ? 'The survey returned no pixels at this position.' : placeholder}
          </span>
        )}
      </div>
    </div>
  )
}
