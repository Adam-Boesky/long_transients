"""Full-pipeline injection-recovery efficiency test.

Extends the extraction-level efficiency notebook (Notebooks/efficiency.ipynb)
by running injected sources through the complete in_ztf filter chain:

  inject (g+r+i) → extract → cross-match with PanSTARRS → quality filters
  → only_big_dmag → at_least_2_bands → no_nearby_source

Survival is tracked at each step per band, plus the final multi-band combined
count.  Results are saved to Data/n_retrieved_pipeline.pkl with shape:

  {float(mag): {step_key: [count_iter0, count_iter1, ...]}}

where step_key is one of:
  '{band}_extracted', '{band}_in_ztf', '{band}_sep_filter', '{band}_ext_artifact_filter',
  '{band}_snr_filter', '{band}_shape_filter', '{band}_psf_filter',
  '{band}_dec_filter', '{band}_only_big_dmag', 'at_least_2_bands', 'no_nearby_source'

NOTE: Full PSF fitting is run (include_psf=True) to match the real pipeline exactly —
PSFFlags drive psf_fit_filter, and PSFMag drives extended_source_artifact_filter.  SNR
uses Kron flux.  Gaia parallax/PM filters are skipped because injected sources have
no Gaia counterpart and auto-pass.

Usage:
    conda run -n long_transients python -m scripts.efficiency.run [--n_iter N] [--output PATH]
"""

import os
import sys
import pickle
import argparse
import warnings
import multiprocessing
import numpy as np

sys.path.append('/Users/adamboesky/Research/long_transients')
warnings.filterwarnings('ignore')

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from Extracting.Source_Extractor import Source_Extractor
from Extracting.cross_match import associate_tables, collapse_nonunique_srcs
from Extracting.utils import get_data_path
from Source_Analysis.filter_fields import (
    Filters,
    build_kde_envelopes,
    load_or_build_kde_envelopes,
    remove_mask,
    PSTARR_UPPER_LIM,
)
from scripts.efficiency.inject import inject_multiband, BANDS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIELD = 582
CCDID = 2
QUADRANT = 1
QUAD_NAME = f'{str(FIELD).zfill(6)}_{str(CCDID).zfill(2)}_{QUADRANT}'
QUAD_DIR = os.path.join(
    get_data_path(), f'ztf_data_injected_sources/{QUAD_NAME}'
)
IMG_DIR = os.path.join(get_data_path(), 'ztf_data')
OUTPUT_PICKLE = os.path.join(get_data_path(), 'n_retrieved_pipeline_fixed_sep.pkl')

N_INJECTED = 100
INJECTION_MAGS = np.arange(16, 25, 1.0)
MIN_DEC = -29.5
SIGMA_BOUNDARY = 3.0

# Sources within this many arcsec of >N_NEARBY_MAX other candidates are dropped
N_NEARBY_MAX = 5
NEARBY_ARCSEC = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count(tab, coords, y_col, x_col, tol=1.0):
    """Count how many injected coords appear in tab using pixel proximity."""
    if len(tab) == 0:
        return 0
    zy = np.asarray(tab[y_col], dtype=float)
    zx = np.asarray(tab[x_col], dtype=float)
    return int(sum(
        np.any((np.abs(zy - c[0]) < tol) & (np.abs(zx - c[1]) < tol))
        for c in coords
    ))


def count_raw(tab, coords, tol=1.0):
    """Count injected sources in a pre-association SEP table (x/y columns)."""
    return _count(tab, coords, y_col='y', x_col='x', tol=tol)


def count_assoc(tab, coords, tol=1.0):
    """Count injected sources in a post-association table (ZTF_x/ZTF_y cols)."""
    return _count(tab, coords, y_col='ZTF_y', x_col='ZTF_x', tol=tol)


def surviving_indices(tab, coords, tol=1.0):
    """Return the set of coord indices that appear in tab (post-association)."""
    if len(tab) == 0:
        return set()
    zy = np.asarray(tab['ZTF_y'], dtype=float)
    zx = np.asarray(tab['ZTF_x'], dtype=float)
    return {
        i for i, c in enumerate(coords)
        if np.any((np.abs(zy - c[0]) < tol) & (np.abs(zx - c[1]) < tol))
    }


def raw_surviving_indices(tab, coords, tol=1.0):
    """Return coord indices appearing in a pre-association SEP table (x/y cols)."""
    if len(tab) == 0:
        return set()
    sy = np.asarray(tab['y'], dtype=float)
    sx = np.asarray(tab['x'], dtype=float)
    return {
        i for i, c in enumerate(coords)
        if np.any((np.abs(sy - c[0]) < tol) & (np.abs(sx - c[1]) < tol))
    }


def _add_psf_proxy_cols(tab, band):
    """Approximate missing PSF mag columns with Kron mag values.

    Using Kron mags as a proxy lets us skip the expensive PSF fitting step
    while still exercising the SNR, shape, and delta-mag filters.
    PSFFlags are set to 0 so psf_fit_filter passes all sources.
    """
    kron_mag = tab[f'ZTF_{band}KronMag'] if f'ZTF_{band}KronMag' in tab.colnames else np.full(len(tab), np.nan)
    kron_err = tab[f'ZTF_{band}KronMagErr'] if f'ZTF_{band}KronMagErr' in tab.colnames else np.full(len(tab), np.nan)
    for col, val in [
        (f'ZTF_{band}PSFMag',    kron_mag),
        (f'ZTF_{band}PSFMagErr', kron_err),
        (f'ZTF_{band}PSFFlags',  np.zeros(len(tab))),
        ('ZTF_qfit',             np.zeros(len(tab))),  # 0.0 = perfect fit proxy
    ]:
        if col not in tab.colnames:
            tab[col] = val
    return tab


def _add_metadata_cols(tab):
    """Add fieldid/ccdid/qid float columns required by Filters.filter()."""
    tab['fieldid'] = np.full(len(tab), float(FIELD))
    tab['ccdid']   = np.full(len(tab), float(CCDID))
    tab['qid']     = np.full(len(tab), float(QUADRANT))
    return tab


def _ensure_pstarr_cols(tab):
    """Ensure all PSTARR band mag columns exist (NaN for in_ztf sources)."""
    for b in BANDS:
        for col in [
            f'PSTARR_{b}PSFMag', f'PSTARR_{b}PSFMagErr',
            f'PSTARR_{b}KronMag', f'PSTARR_{b}KronMagErr',
        ]:
            if col not in tab.colnames:
                tab[col] = np.full(len(tab), np.nan)
    return tab


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

def load_shared_resources() -> dict:
    """Load resources that are reused across all Monte Carlo iterations.

    Returns a dict with keys:
        pstarr_tab   – cleaned PanSTARRS table for this quadrant
        wcss         – {band: WCS}
        nan_masks    – {band: np.ndarray}
        envelopes    – {band: Envelope_KDE} built from in_both sources
    """
    import pickle as pkl

    # PanSTARRS catalog — collapse duplicates as the normal pipeline does
    pstarr_tab = Table.read(os.path.join(QUAD_DIR, 'PSTARR.ecsv'))
    pstarr_tab.remove_column('primaryDetection')
    pstarr_tab = collapse_nonunique_srcs(pstarr_tab)

    # Per-band WCS and NaN masks
    wcss, nan_masks = {}, {}
    for band in BANDS:
        wcs_path  = os.path.join(QUAD_DIR, f'WCSs/ZTF_{band}_wcs.pkl')
        mask_path = os.path.join(QUAD_DIR, f'nan_masks/ZTF_{band}_nan_mask.npy')
        if os.path.exists(wcs_path):
            with open(wcs_path, 'rb') as f:
                wcss[band] = pkl.load(f)
        if os.path.exists(mask_path):
            nan_masks[band] = np.load(mask_path)

    # KDE envelopes — built from in_both sources in the existing associated tables
    # and cached under Data/kde_envelopes/{QUAD_NAME}/ for subsequent runs.
    in_both_tabs = {}
    for band in BANDS:
        fpath = os.path.join(QUAD_DIR, f'{band}_associated.ecsv')
        if not os.path.exists(fpath):
            continue
        tab = Table.read(fpath)
        # Handle upper-limit flag column expected by filter helpers
        if f'ZTF_{band}_upper_lim_flag' not in tab.colnames:
            tab[f'ZTF_{band}_upper_lim_flag'] = False
        in_both_tabs[band] = remove_mask(tab[tab['Catalog_Flag'] == 0])

    envelopes = load_or_build_kde_envelopes(QUAD_NAME, in_both_tabs)
    print(f'KDE envelopes ready for bands: {list(envelopes.keys())}')

    return {
        'pstarr_tab': pstarr_tab,
        'wcss':       wcss,
        'nan_masks':  nan_masks,
        'envelopes':  envelopes,
    }


# ---------------------------------------------------------------------------
# Per-magnitude worker (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

# Shared state injected into each worker process via pool initializer
_WORKER_SHARED: dict | None = None
_WORKER_BAND_EPSFS: dict | None = None
_WORKER_COORDS: np.ndarray | None = None


def _worker_init(shared: dict, band_epsfs: dict, coords: np.ndarray) -> None:
    global _WORKER_SHARED, _WORKER_BAND_EPSFS, _WORKER_COORDS
    import warnings as _w
    _w.filterwarnings('ignore')
    _WORKER_SHARED = shared
    _WORKER_BAND_EPSFS = band_epsfs
    _WORKER_COORDS = coords


def _process_one_mag(task_args):
    """Process a single injection magnitude.  Runs in a worker process.

    Returns (mag_key, mag_results_dict).
    """
    mag, seed, band_to_fpath, save_plots, plot_dir = task_args

    shared    = _WORKER_SHARED
    band_epsfs = _WORKER_BAND_EPSFS
    coords    = _WORKER_COORDS

    pstarr_tab = shared['pstarr_tab']
    wcss       = shared['wcss']
    nan_masks  = shared['nan_masks']
    envelopes  = shared['envelopes']

    mag_key        = float(mag)
    mag_results    = {}
    dmag_survivors: dict[str, set] = {b: set() for b in BANDS}
    g_dropout_indices: dict = {}
    g_extra_info: dict = {}

    for band in BANDS:
        fpath = band_to_fpath.get(band)
        if fpath is None or band not in wcss or band not in nan_masks:
            continue
        if band not in envelopes:
            continue

        # ---- 1. Extract -----------------------------------------------
        se = Source_Extractor(fpath, band)
        se.set_sources_for_psf(pstarr_tab)
        if band in band_epsfs:
            se.epsf = band_epsfs[band]
        ztf_tab = se.get_data_table(include_psf=True, include_kron=True)
        mag_results[f'{band}_extracted'] = count_raw(ztf_tab, coords)

        # ---- 2. Cross-match with PanSTARRS ----------------------------
        assoc = associate_tables(ztf_tab, pstarr_tab, nan_masks[band], wcss[band])
        assoc = remove_mask(assoc)

        # ---- 3. Isolate in_ztf sources and prepare columns ------------
        in_ztf  = assoc[assoc['Catalog_Flag'] == 1].copy()
        in_both = assoc[assoc['Catalog_Flag'] == 0].copy()
        in_ztf  = _add_psf_proxy_cols(in_ztf, band)
        in_ztf  = _ensure_pstarr_cols(in_ztf)
        in_ztf  = _add_metadata_cols(in_ztf)
        if f'ZTF_{band}_mag_limit' not in in_ztf.colnames:
            in_ztf[f'ZTF_{band}_mag_limit'] = float(se.maglimit or np.nan)
        mag_results[f'{band}_in_ztf'] = count_assoc(in_ztf, coords)

        # ---- 4. Apply filter chain ------------------------------------
        filt = Filters()
        tabs = {band: in_ztf}

        tabs = filt.filter(tabs, 'sep_extraction_filter')
        mag_results[f'{band}_sep_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs = filt.filter(tabs, 'extended_source_artifact_filter')
        mag_results[f'{band}_ext_artifact_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs = filt.filter(tabs, 'snr_filter', snr_min=5)
        mag_results[f'{band}_snr_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs = filt.filter(tabs, 'shape_filter')
        mag_results[f'{band}_shape_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs = filt.filter(tabs, 'psf_fit_filter')
        mag_results[f'{band}_psf_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs = filt.filter(tabs, 'dec_greater_than', min_dec=MIN_DEC)
        mag_results[f'{band}_dec_filter'] = count_assoc(tabs.get(band, Table()), coords)

        if band in tabs:
            tabs, _ = filt.filter(
                tabs, 'only_big_dmag',
                sigma_boundary=SIGMA_BOUNDARY,
                envelopes=envelopes,
            )
        mag_results[f'{band}_only_big_dmag'] = count_assoc(tabs.get(band, Table()), coords)
        dmag_survivors[band] = surviving_indices(tabs.get(band, Table()), coords)

        # ---- 5. Stash g-band dropout index sets for plotting ----------
        if save_plots and band == 'g':
            all_idxs      = set(range(N_INJECTED))
            idx_extracted = raw_surviving_indices(ztf_tab, coords)
            idx_in_both   = surviving_indices(in_both, coords)
            idx_in_ztf    = surviving_indices(in_ztf, coords)
            _filt2 = Filters()
            _tabs2 = {band: in_ztf.copy()}
            _tabs2 = _filt2.filter(_tabs2, 'sep_extraction_filter')
            idx_sep = surviving_indices(_tabs2.get(band, Table()), coords)
            if band in _tabs2:
                _tabs2 = _filt2.filter(_tabs2, 'extended_source_artifact_filter')
            idx_sat = surviving_indices(_tabs2.get(band, Table()), coords)
            if band in _tabs2:
                _tabs2 = _filt2.filter(_tabs2, 'snr_filter', snr_min=5)
            idx_snr = surviving_indices(_tabs2.get(band, Table()), coords)
            if band in _tabs2:
                _tabs2 = _filt2.filter(_tabs2, 'shape_filter')
            idx_shape = surviving_indices(_tabs2.get(band, Table()), coords)
            if band in _tabs2:
                _tabs2 = _filt2.filter(_tabs2, 'psf_fit_filter')
            idx_psf = surviving_indices(_tabs2.get(band, Table()), coords)
            if band in _tabs2:
                _tabs2 = _filt2.filter(_tabs2, 'dec_greater_than', min_dec=MIN_DEC)
            idx_dec = surviving_indices(_tabs2.get(band, Table()), coords)
            idx_dmag = dmag_survivors[band]

            g_dropout_indices = {
                'not_extracted':          all_idxs - idx_extracted,
                'flagged_by_cross_match': idx_extracted - idx_in_ztf - idx_in_both,
                'matched_to_panstarrs':   idx_in_both,
                'failed_sep_flags':       idx_in_ztf - idx_sep,
                'failed_ext_artifact':      idx_sep - idx_sat,
                'failed_snr':             idx_sat - idx_snr,
                'failed_shape':           idx_snr - idx_shape,
                'failed_psf_fit':         idx_shape - idx_psf,
                'failed_dec':             idx_psf - idx_dec,
                'failed_delta_mag':       idx_dec - idx_dmag,
                'recovered':              idx_dmag,
            }

            g_extra_info = {'nan_mask': nan_masks.get(band)}

            if g_dropout_indices['failed_sep_flags']:
                zy_f = np.asarray(in_ztf['ZTF_y'], dtype=float)
                zx_f = np.asarray(in_ztf['ZTF_x'], dtype=float)
                flag_map = {}
                for si in g_dropout_indices['failed_sep_flags']:
                    c = coords[si]
                    dists = np.hypot(zy_f - c[0], zx_f - c[1])
                    j = int(np.argmin(dists))
                    if dists[j] < 3.0:
                        flag_map[si] = int(in_ztf['ZTF_sepExtractionFlag'][j])
                g_extra_info['failed_sep_flags'] = flag_map

            if g_dropout_indices['failed_psf_fit'] and 'ZTF_qfit' in in_ztf.colnames:
                zy_f = np.asarray(in_ztf['ZTF_y'], dtype=float)
                zx_f = np.asarray(in_ztf['ZTF_x'], dtype=float)
                qfit_map = {}
                for si in g_dropout_indices['failed_psf_fit']:
                    c = coords[si]
                    dists = np.hypot(zy_f - c[0], zx_f - c[1])
                    j = int(np.argmin(dists))
                    if dists[j] < 3.0:
                        qfit_map[si] = float(in_ztf['ZTF_qfit'][j])
                g_extra_info['failed_psf_fit'] = qfit_map

            if g_dropout_indices['matched_to_panstarrs'] and 'g' in wcss:
                zy_b = np.asarray(in_both['ZTF_y'], dtype=float)
                zx_b = np.asarray(in_both['ZTF_x'], dtype=float)
                pstarr_map = {}
                for si in g_dropout_indices['matched_to_panstarrs']:
                    c = coords[si]
                    dists = np.hypot(zy_b - c[0], zx_b - c[1])
                    j = int(np.argmin(dists))
                    if dists[j] < 3.0:
                        pra  = float(in_both['PSTARR_ra'][j])
                        pdec = float(in_both['PSTARR_dec'][j])
                        px, py = wcss['g'].all_world2pix([[pra, pdec]], 0)[0]
                        pstarr_map[si] = (float(py), float(px))
                g_extra_info['matched_to_panstarrs'] = pstarr_map

    # ---- 6. Save dropout cutout plots (g-band, seed=0 only) ----------
    if save_plots and g_dropout_indices:
        from astropy.io import fits
        from astropy.wcs.utils import proj_plane_pixel_scales
        from scripts.efficiency.plot_dropouts import plot_dropout_cutouts
        fpath_orig = os.path.join(QUAD_DIR, 'orig_g.fits')
        fpath_inj  = band_to_fpath.get('g')
        if fpath_inj and os.path.exists(fpath_orig):
            orig_img = fits.open(fpath_orig)[0].data.byteswap().newbyteorder()
            inj_img  = fits.open(fpath_inj)[0].data.byteswap().newbyteorder()
            pixel_scale_arcsec = float(
                proj_plane_pixel_scales(wcss['g'])[0] * 3600
            )
            plot_dropout_cutouts(
                mag=mag, seed=seed, coords=coords,
                orig_img=orig_img, inj_img=inj_img,
                dropout_index_sets=g_dropout_indices,
                extra_info=g_extra_info,
                pixel_scale_arcsec=pixel_scale_arcsec,
                out_dir=plot_dir,
            )

    # ---- 7. at_least_2_bands -----------------------------------------
    mag_results['at_least_2_bands'] = sum(
        1 for i in range(N_INJECTED)
        if sum(i in dmag_survivors[b] for b in BANDS) >= 2
    )

    # ---- 8. no_nearby_source -----------------------------------------
    survivors_2band = [
        i for i in range(N_INJECTED)
        if sum(i in dmag_survivors[b] for b in BANDS) >= 2
    ]
    if len(survivors_2band) == 0:
        mag_results['no_nearby_source'] = 0
    else:
        surviving_ras, surviving_decs = [], []
        for i in survivors_2band:
            c = coords[i]
            for b in BANDS:
                fpath_b = band_to_fpath.get(b)
                if fpath_b is None or b not in wcss:
                    continue
                ra, dec = wcss[b].all_pix2world([[c[1], c[0]]], 0)[0]
                surviving_ras.append(ra)
                surviving_decs.append(dec)
                break
        if len(surviving_ras) == 0:
            mag_results['no_nearby_source'] = 0
        else:
            sky = SkyCoord(surviving_ras, surviving_decs, unit='deg')
            seps = sky.separation(sky[:, None])
            nearby_counts = np.sum(seps.arcsecond < NEARBY_ARCSEC, axis=1) - 1
            mag_results['no_nearby_source'] = int(np.sum(nearby_counts <= N_NEARBY_MAX))

    return mag_key, mag_results


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def run_iteration(seed: int, shared: dict, plot_dir: str | None = None) -> dict:
    """Run one full injection-recovery iteration.

    Args:
        seed: Random seed (also used as the iteration index).
        shared: Dict returned by load_shared_resources().
        plot_dir: If given, save per-filter dropout cutout PDFs for the g-band
            under <plot_dir>/mag_<XX.X>/.  Only pass this for seed=0.

    Returns:
        results: {float(mag): {step_key: int}}
    """
    np.random.seed(seed)

    # Wipe any stale PDFs from previous runs before writing any new plots,
    # so that filter categories with zero dropouts don't leave old files behind.
    if plot_dir is not None:
        import shutil
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

    pstarr_tab = shared['pstarr_tab']
    wcss       = shared['wcss']
    nan_masks  = shared['nan_masks']
    envelopes  = shared['envelopes']

    # Inject into all 3 bands at the same pixel positions
    coords, mag_to_band_to_fpath = inject_multiband(
        field=FIELD, ccdid=CCDID, quadrant=QUADRANT,
        outdir=QUAD_DIR, img_dir=IMG_DIR,
        n_injected=N_INJECTED, injection_mags=INJECTION_MAGS,
    )

    # Pre-build one EPSF per band from the original (un-injected) image and
    # reuse it for every injected magnitude.  Building the EPSF is the
    # dominant cost (~7 min/band), so caching it reduces the per-magnitude
    # work from O(EPSF build) to O(PSF photometry only).
    band_epsfs: dict[str, object] = {}
    for band in BANDS:
        orig_fpath = os.path.join(QUAD_DIR, f'orig_{band}.fits')
        if not os.path.exists(orig_fpath):
            continue
        if band not in wcss or band not in nan_masks or band not in envelopes:
            continue
        print(f'[seed={seed}] Pre-building EPSF for band {band} from original image...')
        se_orig = Source_Extractor(orig_fpath, band)
        band_epsfs[band] = se_orig.build_epsf_cache(pstarr_tab)
        print(f'[seed={seed}] EPSF for band {band} cached.')

    # Dispatch all magnitudes in parallel — each is fully independent.
    save_plots = plot_dir is not None
    n_workers = min(len(INJECTION_MAGS), max(1, (os.cpu_count() or 4) - 1))
    print(f'[seed={seed}] Processing {len(INJECTION_MAGS)} magnitudes with {n_workers} workers...')
    tasks = [
        (float(mag), seed, mag_to_band_to_fpath[float(mag)], save_plots, plot_dir)
        for mag in INJECTION_MAGS
    ]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(shared, band_epsfs, coords),
    ) as pool:
        mag_result_list = pool.map(_process_one_mag, tasks)

    return {mag_key: mag_results for mag_key, mag_results in mag_result_list}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run full-pipeline injection-recovery efficiency test.'
    )
    parser.add_argument('--n_iter', type=int, default=10,
                        help='Number of Monte Carlo iterations (default: 10).')
    parser.add_argument('--output', type=str, default=OUTPUT_PICKLE,
                        help='Path to write results pickle.')
    parser.add_argument('--plot_dir', type=str, default='Figures/filter_dropouts',
                        help='Directory for per-filter dropout cutout PDFs (seed=0 only). '
                             'Pass empty string to disable.')
    args = parser.parse_args()

    print('Loading shared resources...')
    shared = load_shared_resources()

    n_retrieved: dict = {}

    plot_dir = args.plot_dir if args.plot_dir else None

    for i in range(args.n_iter):
        if i % max(1, args.n_iter // 5) == 0:
            print(f'Iteration {i} / {args.n_iter}')

        iter_results = run_iteration(
            seed=i, shared=shared,
            plot_dir=plot_dir if i == 0 else None,
        )

        for mag, step_counts in iter_results.items():
            if mag not in n_retrieved:
                n_retrieved[mag] = {}
            for step, count in step_counts.items():
                n_retrieved[mag].setdefault(step, []).append(count)

    with open(args.output, 'wb') as f:
        pickle.dump(n_retrieved, f)

    print(f'Saved results to {args.output}')

    # Print a quick summary
    print('\nMean recovery fraction at each step (g-band, all mags):')
    mags = sorted(n_retrieved.keys())
    steps = [k for k in n_retrieved[mags[0]].keys() if k.startswith('g_') or k in ('at_least_2_bands', 'no_nearby_source')]
    header = f"{'mag':>6}  " + '  '.join(f'{s:>20}' for s in steps)
    print(header)
    for mag in mags:
        row = f'{mag:>6.1f}  '
        for step in steps:
            vals = n_retrieved[mag].get(step, [0])
            row += f'{np.mean(vals)/N_INJECTED:>20.3f}  '
        print(row)


if __name__ == '__main__':
    main()
