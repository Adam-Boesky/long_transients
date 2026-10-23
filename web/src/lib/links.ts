/** Links out to the survey archives for the source on screen.
 *
 * Ported verbatim from `classify_app.lib.external`, whose docstring records
 * that these URL shapes were checked against the live services rather than
 * taken from memory -- so they are copied, not reconstructed.
 *
 *   ZTF   the application's URL API (api=searchPos), which lands on the
 *         Results tab with an image preview. Deliberately not IBE, which
 *         returns exposure metadata and no picture.
 *   PS1   ps1cutouts, asking for stacks and warps, colour plus grizy.
 *   TNS   always a positional cone search, never by designation: our
 *         designation comes from a crossmatch, and linking straight to it
 *         would assert an identification the link cannot show you.
 */

/** Loose enough for a slightly disagreeing reported position, tight enough
 *  not to dredge up neighbours. */
export const TNS_SEARCH_ARCSEC = 5
/** Only needs to be big enough that an exposure containing the position
 *  comes back; the sources are point-like. */
const ZTF_SEARCH_ARCSEC = 10
/** ~1 arcmin at the PS1 pixel scale. */
const PS1_CUTOUT_SIZE = 240

export interface ExternalLink {
  label: string
  url: string
  help: string
}

function ztfImagesUrl(ra: number, dec: number): string {
  const q = new URLSearchParams({
    api: 'searchPos',
    ra: ra.toFixed(6),
    dec: dec.toFixed(6),
    sr: `${ZTF_SEARCH_ARCSEC}s`,
    intersect: 'OVERLAPS',
    execute: 'true',
  })
  return `https://irsa.ipac.caltech.edu/applications/ztf/?${q}`
}

function panstarrsImagesUrl(ra: number, dec: number): string {
  // `filter` and `filetypes` repeat, so this is built as pairs.
  const q = new URLSearchParams([
    ['pos', `${ra.toFixed(6)} ${dec.toFixed(6)}`],
    ['filter', 'color'],
    ['filter', 'g'], ['filter', 'r'], ['filter', 'i'],
    ['filter', 'z'], ['filter', 'y'],
    ['filetypes', 'stack'],
    ['filetypes', 'warp'],
    ['size', String(PS1_CUTOUT_SIZE)],
    ['output_size', '0'],
    ['verbose', '0'],
    ['autoscale', '99.5'],
  ])
  return `https://ps1images.stsci.edu/cgi-bin/ps1cutouts?${q}`
}

function tnsSearchUrl(ra: number, dec: number): string {
  const q = new URLSearchParams({
    ra: ra.toFixed(6),
    decl: dec.toFixed(6),
    radius: String(TNS_SEARCH_ARCSEC),
    coords_unit: 'arcsec',
  })
  return `https://www.wis-tns.org/search?&${q}`
}

/** Links to the survey archives' own image viewers.
 *
 * Separate from `tnsLink` because these belong beside the cutouts -- they are
 * the "show me more of this image" action -- while TNS answers a different
 * question and sits with the classifications. */
export function imageLinks(ra: number | null, dec: number | null): ExternalLink[] {
  if (ra === null || dec === null) return []
  return [
    { label: 'ZTF images', url: ztfImagesUrl(ra, dec),
      help: "IRSA's ZTF image search at this position: science, reference and "
            + 'deep-reference products, with previews.' },
    { label: 'PanSTARRS images', url: panstarrsImagesUrl(ra, dec),
      help: 'PS1 stack and single-epoch warp cutouts, colour and grizy. '
            + 'The warps make for a long page.' },
  ]
}

/** The TNS cone search, which belongs with the other classifications: it is
 *  asking the same question they answer -- has anyone already typed this? */
export function tnsLink(ra: number | null, dec: number | null): ExternalLink | null {
  if (ra === null || dec === null) return null
  return { label: 'Search TNS', url: tnsSearchUrl(ra, dec),
           help: `TNS cone search within ${TNS_SEARCH_ARCSEC}" of this position.` }
}
