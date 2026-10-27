/** Thin typed wrapper over the FastAPI backend.
 *
 * No data-fetching library: every call here is a plain GET keyed by source,
 * and the one thing that actually matters for responsiveness -- having the
 * next source ready before it is asked for -- is a `prefetch` the caller
 * drives explicitly, not something a cache heuristic should be guessing at.
 */
import type {
  Annotation, CatalogSchema, ColumnStats, CoordListPreview, CoordListSummary, Cutouts, EpochImage, FilterNode, FilterPreview,
  FilterSchema, GroupMember, Label, LightCurve, Progress, SourceDetail, SourcePage,
  SourceQuery, Spectra, TableAnnotations, TablePayload, TableQuery, ZtforceQueue,
  ZtforceResult,
} from './types'

// Same-origin by default: both `vite dev` and `vite preview` proxy /api to
// the backend, so relative fetches work unchanged in production behind one
// host and no CORS entry is needed per port.
export const API_BASE = import.meta.env.VITE_API_BASE ?? ''

/** `fetch`, plus the headers every call to this API wants.
 *
 * Only one so far. ngrok's free tier answers browser requests with an
 * "you are about to visit" interstitial until you click through it, and a
 * `/api/...` call answered with that page's HTML fails in `res.json()` as
 * `Unexpected token '<'` -- a parse error that reads as a bug in the data
 * layer when the cause is the tunnel in front of it. The header suppresses
 * it; ngrok only checks that it is present, and every other host ignores an
 * unknown request header, so this costs nothing when not tunnelled.
 *
 * Not a complete answer: cutout `<img>` tags cannot carry headers. Those are
 * covered by the cookie ngrok sets once you click through, which is already
 * in place by the time the page runs, since loading the page is what you
 * clicked through to reach.
 */
function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  // Via `Headers` rather than spreading, so this keeps working if a caller
  // ever passes a `Headers` instance instead of a plain object.
  const headers = new Headers(init.headers)
  headers.set('ngrok-skip-browser-warning', 'true')
  return fetch(url, { ...init, headers })
}

export class ApiError extends Error {
  // Fields declared explicitly rather than as constructor parameter
  // properties: the template enables `erasableSyntaxOnly`, which forbids
  // syntax that cannot be erased by a type-stripping transform.
  readonly status: number
  readonly url: string

  constructor(status: number, url: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.url = url
  }
}

/** True for the DOMException a cancelled fetch throws.
 *
 * React StrictMode mounts, unmounts and remounts in development, so an
 * effect that aborts on cleanup *always* rejects its first request. Treating
 * that as a failure renders an error the remount never clears -- which looks
 * exactly like a hung page. Callers should ignore it.
 */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function qs(params: object): string {
  const out = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    // Repeated keys, not a joined string: FastAPI reads `list[str]` from
    // `?k=a&k=b`, and `String(['a','b'])` would arrive as the single
    // label "a,b".
    if (Array.isArray(v)) {
      for (const item of v) if (item !== '') out.append(k, String(item))
      continue
    }
    out.set(k, String(v))
  }
  const s = out.toString()
  return s ? `?${s}` : ''
}

async function get<T>(path: string, params: object = {},
                      signal?: AbortSignal): Promise<T> {
  const url = `${API_BASE}${path}${qs(params)}`
  const res = await apiFetch(url, { signal })
  if (!res.ok) {
    throw new ApiError(res.status, url,
      `${res.status} ${res.statusText} for ${path}`)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, signal?: AbortSignal): Promise<T> {
  const url = `${API_BASE}${path}`
  const res = await apiFetch(url, { method: 'POST', signal })
  if (!res.ok) {
    throw new ApiError(res.status, url,
      `${res.status} ${res.statusText} for ${path}`)
  }
  return res.json() as Promise<T>
}

// ── sources ───────────────────────────────────────────────────────────────────

export const listSources = (q: SourceQuery = {}, signal?: AbortSignal) =>
  get<SourcePage>('/api/sources', q, signal)

/** Where `target` sits in the current filtered order, or null if the filters
 *  exclude it. The client only holds a window, so it cannot find this itself. */
export const locateSource = (target: string, q: SourceQuery = {},
                             signal?: AbortSignal) =>
  get<{ offset: number | null }>('/api/sources-locate', { ...q, target }, signal)

/** Every detection of the same object (rows from overlapping fields that
 *  share a `group_id`), this one included. Empty when it has no group. */
export const getGroup = (uuid: string, signal?: AbortSignal) =>
  get<{ members: GroupMember[] }>(
    `/api/sources/${encodeURIComponent(uuid)}/group`, {}, signal)

/** Offset of the next source `author` has not labelled, searching from
 *  `after` and wrapping once. Null when everything in the filter is done. */
export const nextUnlabelled = (author: string, after: number,
                               q: SourceQuery = {}, signal?: AbortSignal) =>
  get<{ offset: number | null; source_uuid: string | null }>(
    '/api/sources-next-unlabelled', { ...q, author, after }, signal)

// ── filters ───────────────────────────────────────────────────────────────────

export const getFilterSchema = (signal?: AbortSignal) =>
  get<FilterSchema>('/api/filters/columns', {}, signal)

export const getColumnStats = (column: string, signal?: AbortSignal) =>
  get<ColumnStats>(`/api/filters/stats/${encodeURIComponent(column)}`, {}, signal)

/** How many sources a filter matches, and how many each clause keeps alone.
 *
 *  A POST because a filter document has no length limit here, unlike the URL
 *  the list endpoint reads it from. */
export async function previewFilter(doc: FilterNode | null,
                                    scope: SourceQuery = {},
                                    signal?: AbortSignal) {
  // `scope` is the rest of the sidebar. Counted together so the panel and the
  // page header cannot report different numbers for the same selection.
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(scope)) {
    if (v !== undefined && v !== null && v !== '' && k !== 'filter') {
      qs.set(k, String(v))
    }
  }
  const res = await apiFetch(`${API_BASE}/api/filters/preview?${qs}`, {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(doc),
  })
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
  return await res.json() as FilterPreview
}

export const getSource = (uuid: string, signal?: AbortSignal) =>
  get<SourceDetail>(`/api/sources/${encodeURIComponent(uuid)}`, {}, signal)



// ── chart payloads ────────────────────────────────────────────────────────────
//
// Served gzipped with `Content-Encoding: gzip`; the browser inflates them
// transparently, so these are ordinary JSON fetches.

export const getLightCurve = (uuid: string, signal?: AbortSignal) =>
  get<LightCurve>(`/api/sources/${encodeURIComponent(uuid)}/lightcurve`, {}, signal)

export const getSpectra = (uuid: string, signal?: AbortSignal) =>
  get<Spectra>(`/api/sources/${encodeURIComponent(uuid)}/spectra`, {}, signal)

/** Image *addresses*, not images: the browser fetches the PNGs itself.
 *
 * The stamps are rendered from FITS server-side with the same log/99.5
 * viridis stretch the analysis code uses, so `stretch` and `percent` are
 * part of the address -- changing either is a different URL and cannot come
 * back from a stale cache.
 */
export const getCutouts = (uuid: string,
                           opts: { stretch?: string; percent?: number } = {},
                           signal?: AbortSignal) =>
  get<Cutouts>(`/api/sources/${encodeURIComponent(uuid)}/cutouts`,
               opts as Record<string, string | number>, signal)
    // The API returns same-origin paths. Prefix them only when the frontend
    // has been pointed at a different host, so `<img src>` resolves there too.
    .then((c) => API_BASE ? {
      ...c,
      images: Object.fromEntries(Object.entries(c.images).map(([survey, bands]) =>
        [survey, Object.fromEntries(Object.entries(bands)
          .map(([band, url]) => [band, `${API_BASE}${url}`]))])),
    } : c)

/** Find the exposure behind one light-curve point. Rejects with the server's
 *  own reason (no exposure matched, the survey's service is down) so the
 *  panel can say which. */
export async function getEpochImage(
  uuid: string, survey: string, band: string, mjd: number,
  opts: { objectid?: string | null; stretch?: string; percent?: number } = {},
  signal?: AbortSignal,
): Promise<EpochImage> {
  const path = `/api/sources/${encodeURIComponent(uuid)}/epoch/${survey}/${band}`
  const url = `${API_BASE}${path}${qs({ mjd, ...opts })}`
  const res = await apiFetch(url, { signal })
  if (!res.ok) {
    const detail = await res.json().then((b) => b?.detail).catch(() => null)
    throw new ApiError(res.status, url, typeof detail === 'string' ? detail
      : `${res.status} ${res.statusText} for ${path}`)
  }
  const e = await res.json() as EpochImage
  return API_BASE ? { ...e, png: `${API_BASE}${e.png}` } : e
}

// ── annotations ───────────────────────────────────────────────────────────────


export async function putAnnotation(
  uuid: string, author: string,
  body: { label?: string | null; confidence?: string | null;
          note?: string | null; tags?: string[]; status?: string | null },
): Promise<Annotation> {
  const url = `${API_BASE}/api/sources/${encodeURIComponent(uuid)}`
    + `/annotations/${encodeURIComponent(author)}`
  const res = await apiFetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags: [], ...body }),
  })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
  return res.json()
}

// ── coordinate lists ──────────────────────────────────────────────────────────

/** A JSON request whose error carries the server's `detail`, which says what to fix. */
async function send<T>(method: string, path: string, body?: object): Promise<T> {
  const url = `${API_BASE}${path}`
  const res = await apiFetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const detail = await res.json().then((b) => b.detail).catch(() => null)
    throw new ApiError(res.status, url, detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json()
}

/** What positions (the text of a .txt) would match, without saving anything. */
export const previewCoordList = (text: string, radiusArcsec: number) =>
  send<CoordListPreview>('POST', '/api/coord-lists/preview',
                         { text, radius_arcsec: radiusArcsec })

/** Match and save a list everyone can see. */
export const createCoordList = (text: string, name: string, radiusArcsec: number) =>
  send<CoordListSummary>('POST', '/api/coord-lists',
                         { text, name, radius_arcsec: radiusArcsec })

export const listCoordLists = (signal?: AbortSignal) =>
  get<CoordListSummary[]>('/api/coord-lists', {}, signal)

export const getCoordList = (id: string, signal?: AbortSignal) =>
  get<CoordListSummary>(`/api/coord-lists/${encodeURIComponent(id)}`, {}, signal)

/** Anyone may rename or delete a list; deleting only hides it. */
export const renameCoordList = (id: string, name: string) =>
  send<CoordListSummary>('PATCH', `/api/coord-lists/${encodeURIComponent(id)}`, { name })

export const deleteCoordList = (id: string) =>
  send<{ deleted: string }>('DELETE', `/api/coord-lists/${encodeURIComponent(id)}`)

// ── labels ────────────────────────────────────────────────────────────────────
// Shared by everyone, in the order they were added.

export const getLabels = (signal?: AbortSignal) =>
  get<Label[]>('/api/labels', {}, signal)

/** Throws an `ApiError` with status 409 if the label exists in any case. */
export async function addLabel(name: string, author: string): Promise<Label> {
  const url = `${API_BASE}/api/labels`
  const res = await apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, author }),
  })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
  return res.json()
}

/** Recolour a label for everyone; null goes back to the default. */
export async function setLabelColor(name: string,
                                    color: string | null): Promise<Label> {
  const url = `${API_BASE}/api/labels/${encodeURIComponent(name)}`
  const res = await apiFetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ color }),
  })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
  return res.json()
}

/** Throws an `ApiError` with status 409 if any source is filed under it. */
export async function deleteLabel(name: string): Promise<void> {
  const url = `${API_BASE}/api/labels/${encodeURIComponent(name)}`
  const res = await apiFetch(url, { method: 'DELETE' })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
}

/** Throws an `ApiError` with status 409 if `names` is not exactly the
 *  current labels -- someone added one since you loaded them. */
export async function reorderLabels(names: string[]): Promise<Label[]> {
  const url = `${API_BASE}/api/labels/order`
  const res = await apiFetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
  return res.json()
}

export const getTable = (q: TableQuery = {}, signal?: AbortSignal) =>
  get<TablePayload>('/api/table', q, signal)

/** Verdicts for the table, fetched apart from the catalog half.
 *
 *  They are the only part that changes, so keeping them separate is what lets
 *  the catalog half be cached and handed over gzipped untouched. The page
 *  joins the two on `source_uuid`. */
export const getTableAnnotations = (signal?: AbortSignal) =>
  get<TableAnnotations>('/api/table/annotations', {}, signal)

export const getProgress = (signal?: AbortSignal) =>
  get<Progress>('/api/progress', {}, signal)

/** Only the part of `/api/stats` the member picker reads: everyone who has
 *  filed anything, label or comment, with how many sources. */
export const getAuthors = (signal?: AbortSignal) =>
  get<{ by_author: Record<string, number> }>('/api/stats', {}, signal)
    .then((s) => s.by_author)

export const getCatalogSchema = (signal?: AbortSignal) =>
  get<CatalogSchema>('/api/catalog/schema', {}, signal)



// ── ztforce ───────────────────────────────────────────────────────────────────
// Runs go through a server-side queue: labelling a source queues it, and these
// read its state and results, or ask for it to run now.

export const getZtforce = (uuid: string, signal?: AbortSignal) =>
  get<ZtforceResult>(`/api/sources/${encodeURIComponent(uuid)}/ztforce`, {}, signal)

/** Queue this source at the front (no-op if it already has a current result). */
export const runZtforce = (uuid: string, signal?: AbortSignal) =>
  post<ZtforceResult>(`/api/sources/${encodeURIComponent(uuid)}/ztforce`, signal)

export const getZtforceQueue = (signal?: AbortSignal) =>
  get<ZtforceQueue>('/api/ztforce/queue', {}, signal)

export async function setZtforcePaused(paused: boolean): Promise<ZtforceQueue> {
  const url = `${API_BASE}/api/ztforce/queue`
  const res = await apiFetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paused }),
  })
  if (!res.ok) throw new ApiError(res.status, url, `${res.status} ${res.statusText}`)
  return res.json()
}
