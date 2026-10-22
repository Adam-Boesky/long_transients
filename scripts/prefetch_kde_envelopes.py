"""Pre-populate KDE envelope caches for all gri fields.

Self-contained: inlines only the functions needed to run the quality-filter
pipeline and build KDE envelopes, avoiding the full filter_fields.py import
chain (which pulls in Sources → sparcl and other heavy optional deps).

Safe to run in parallel with the main filter job — both produce identical
deterministic output, so a write race at most wastes one computation.
"""

import os
import sys
import numpy as np

from typing import Dict, Iterable, Union
from astropy.table import Table, MaskedColumn, vstack
from concurrent.futures import ProcessPoolExecutor, as_completed
from KDEpy import TreeKDE

DATA_PATH = '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients'
EXTRACTED_CATALOG_DIR = 'catalog_results/field_results'
BANDS = ('g', 'r', 'i')
MIN_DEC = -29.5


# ---------------------------------------------------------------------------
# Utilities inlined from filter_fields.py / Extracting/utils.py
# ---------------------------------------------------------------------------

def _is_flag(flags: np.ndarray, flag: Union[int, Iterable[int]]) -> np.ndarray:
    flags = np.nan_to_num(flags, nan=0).astype(int)
    if isinstance(flag, int):
        return np.bitwise_and(flags, flag) != 0
    mask = np.zeros_like(flags, dtype=bool)
    for f in flag:
        mask |= (np.bitwise_and(flags, f) != 0)
    return mask


def remove_mask(tab: Table) -> Table:
    _INT64_COLUMNS = {'ZTF_objID', 'PSTARR_PanSTARR_ID'}
    tab = Table(tab, masked=True)
    for col in tab.colnames:
        column = tab[col]
        if col in _INT64_COLUMNS:
            tab[col] = column.astype(object)
            tab[col][column.mask] = None
        elif np.issubdtype(column.dtype, np.number):
            tab[col] = column.astype(float)
            tab[col][column.mask] = np.nan
        else:
            tab[col] = column.astype(object)
            tab[col][column.mask] = np.nan
    return Table(tab, masked=False)


def get_snr_from_mag(mag, magerr, zp=25.0):
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    flux = 10 ** ((zp - mag) / 2.5)
    fluxerr = flux * np.abs(magerr) * np.log(10) / 2.5
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(fluxerr > 0, flux / fluxerr, np.nan)


# ---------------------------------------------------------------------------
# KDE envelope (mirrors filter_fields.py exactly)
# ---------------------------------------------------------------------------

class Envelope_KDE:
    def __init__(self, mag_grid, dmag_grid, kde_ys):
        self.mag_grid = mag_grid
        self.dmag_grid = dmag_grid
        self.kde_ys = kde_ys

    def get_percentile(self, mag_grid, percentile):
        scalar = isinstance(percentile, (int, float))
        percentiles = [percentile] if scalar else list(percentile)
        cdf = np.cumsum(self.kde_ys, axis=1) / np.sum(self.kde_ys, axis=1, keepdims=True)
        results = np.array([
            [np.interp(p, cdf[i], self.dmag_grid) for i in range(len(self.mag_grid))]
            for p in percentiles
        ])
        results = np.array([
            np.interp(mag_grid, self.mag_grid, results[i])
            for i in range(len(percentiles))
        ])
        return results[0] if scalar else results


def build_kde_envelopes(
    tabs: Dict[str, Table],
    mag_grid: np.ndarray = np.arange(14, 24, 0.25),
    dmag_grid: np.ndarray = np.linspace(-5, 5, 500),
    half_bin_width: float = 0.5,
    min_dmagerr: float = 0.01,
) -> Dict[str, Envelope_KDE]:
    envelopes = {}
    for band, tab in tabs.items():
        finite_mask = (
            np.isfinite(tab[f'PSTARR_{band}PSFMag']) &
            np.isfinite(tab[f'ZTF_{band}PSFMag'])
        )
        tab = tab[finite_mask]

        pstarr_mag = np.array(tab[f'PSTARR_{band}PSFMag'])
        pstarr_magerr = np.array(tab[f'PSTARR_{band}PSFMagErr'])
        ztf_mag = np.array(tab[f'ZTF_{band}PSFMag'])
        ztf_magerr = np.array(tab[f'ZTF_{band}PSFMagErr'])

        mag = np.mean([pstarr_mag, ztf_mag], axis=0)
        dmag = ztf_mag - pstarr_mag
        dmagerr = np.maximum(np.sqrt(pstarr_magerr**2 + ztf_magerr**2), min_dmagerr)
        finite_mask = np.isfinite(dmagerr) & np.isfinite(dmag) & (dmagerr < 100.0)

        kde_ys = []
        for mag_center in mag_grid:
            mag_mask = np.abs(mag - mag_center) < half_bin_width
            mag_mask &= finite_mask
            if np.sum(mag_mask) <= 1:
                kde_ys.append(np.full(len(dmag_grid), np.nan))
                continue
            y_kde = TreeKDE(bw=dmagerr[mag_mask]).fit(dmag[mag_mask]).evaluate(dmag_grid)
            kde_ys.append(y_kde)

        envelopes[band] = Envelope_KDE(mag_grid, dmag_grid, np.array(kde_ys))
    return envelopes


def load_or_build_kde_envelopes(field_name: str, tabs: Dict[str, Table]) -> Dict[str, Envelope_KDE]:
    cache_dir = os.path.join(DATA_PATH, 'kde_envelopes', field_name)
    bands = list(tabs.keys())

    if all(os.path.exists(os.path.join(cache_dir, f'{b}.npz')) for b in bands):
        envelopes = {}
        for band in bands:
            data = np.load(os.path.join(cache_dir, f'{band}.npz'))
            envelopes[band] = Envelope_KDE(data['mag_grid'], data['dmag_grid'], data['kde_ys'])
        return envelopes

    envelopes = build_kde_envelopes(tabs)
    os.makedirs(cache_dir, exist_ok=True)
    for band, env in envelopes.items():
        np.savez(os.path.join(cache_dir, f'{band}.npz'),
                 mag_grid=env.mag_grid, dmag_grid=env.dmag_grid, kde_ys=env.kde_ys)
    return envelopes


# ---------------------------------------------------------------------------
# Quality-filter steps (mirrors the throwaway _qfilters pipeline in filter_field)
# ---------------------------------------------------------------------------

def _sep_extraction_filter(tabs):
    bad_flags = [4, 8, 16, 32, 64, 128]
    return {b: t[~_is_flag(t['ZTF_sepExtractionFlag'], bad_flags)] for b, t in tabs.items()}


def _snr_filter_both_cat(tabs):
    """Mirrors snr_filter(both_cat=True): returns good, ztf_only, pstarr_only."""
    snr_min = 5.0
    good, ztf_only, pstarr_only = {}, {}, {}
    for band, tab in tabs.items():
        zp = np.nan_to_num(tab[f'ZTF_{band}_zero_pt_mag'], nan=25.0)

        psf_mask = (
            np.logical_or(
                np.logical_and(
                    get_snr_from_mag(tab[f'PSTARR_{band}PSFMag'], tab[f'PSTARR_{band}PSFMagErr'], zp=25) > snr_min,
                    get_snr_from_mag(tab[f'ZTF_{band}PSFMag'],   tab[f'ZTF_{band}PSFMagErr'],   zp=zp)  > snr_min,
                ),
                np.logical_or(
                    np.logical_and(np.isnan(get_snr_from_mag(tab[f'PSTARR_{band}PSFMag'], tab[f'PSTARR_{band}PSFMagErr'], zp=25)),
                                   get_snr_from_mag(tab[f'ZTF_{band}PSFMag'], tab[f'ZTF_{band}PSFMagErr'], zp=zp) > snr_min),
                    np.logical_and(get_snr_from_mag(tab[f'PSTARR_{band}PSFMag'], tab[f'PSTARR_{band}PSFMagErr'], zp=25) > snr_min,
                                   np.isnan(get_snr_from_mag(tab[f'ZTF_{band}PSFMag'], tab[f'ZTF_{band}PSFMagErr'], zp=zp))),
                ),
            )
        )
        kron_mask = (
            np.logical_or(
                np.logical_and(
                    get_snr_from_mag(tab[f'PSTARR_{band}KronMag'], tab[f'PSTARR_{band}KronMagErr'], zp=25) > snr_min,
                    get_snr_from_mag(tab[f'ZTF_{band}KronMag'],   tab[f'ZTF_{band}KronMagErr'],   zp=zp)  > snr_min,
                ),
                np.logical_or(
                    np.logical_and(np.isnan(get_snr_from_mag(tab[f'PSTARR_{band}KronMag'], tab[f'PSTARR_{band}KronMagErr'], zp=25)),
                                   get_snr_from_mag(tab[f'ZTF_{band}KronMag'], tab[f'ZTF_{band}KronMagErr'], zp=zp) > snr_min),
                    np.logical_and(get_snr_from_mag(tab[f'PSTARR_{band}KronMag'], tab[f'PSTARR_{band}KronMagErr'], zp=25) > snr_min,
                                   np.isnan(get_snr_from_mag(tab[f'ZTF_{band}KronMag'], tab[f'ZTF_{band}KronMagErr'], zp=zp))),
                ),
            )
        )
        both_mask = psf_mask | kron_mask

        ztf_psf_snr   = get_snr_from_mag(tab[f'ZTF_{band}PSFMag'],  tab[f'ZTF_{band}PSFMagErr'],  zp=zp)
        ztf_kron_snr  = get_snr_from_mag(tab[f'ZTF_{band}KronMag'], tab[f'ZTF_{band}KronMagErr'], zp=zp)
        pstarr_psf_snr  = get_snr_from_mag(tab[f'PSTARR_{band}PSFMag'],  tab[f'PSTARR_{band}PSFMagErr'],  zp=25)
        pstarr_kron_snr = get_snr_from_mag(tab[f'PSTARR_{band}KronMag'], tab[f'PSTARR_{band}KronMagErr'], zp=25)

        ztf_mask    = ((ztf_psf_snr > snr_min)   | (ztf_kron_snr > snr_min))   & ~both_mask
        pstarr_mask = ((pstarr_psf_snr > snr_min) | (pstarr_kron_snr > snr_min)) & ~both_mask

        good[band]      = tab[both_mask]
        ztf_only[band]  = tab[ztf_mask]
        pstarr_only[band] = tab[pstarr_mask]

    return good, ztf_only, pstarr_only


def _shape_filter(tabs):
    good = {}
    for band, tab in tabs.items():
        mask = (tab['ZTF_a'] / tab['ZTF_b']) < 2
        mask &= ((tab['ZTF_a'] / tab['ZTF_b']) < 1.35) | (tab['ZTF_tnpix'] < 200)
        mask |= np.isnan(tab['ZTF_a'])
        good[band] = tab[mask]
    return good


def _pstarr_not_saturated(tabs, ztf_mag_min=13.5):
    good = {}
    for band, tab in tabs.items():
        mask = (
            ~_is_flag(tab[f'PSTARR_{band}infoFlag'], 128) &
            ~_is_flag(tab[f'PSTARR_{band}infoFlag'], 4096) &
            ~_is_flag(tab[f'PSTARR_{band}infoFlag2'], 4096)
        )
        mask &= (tab[f'ZTF_{band}PSFMag'] > ztf_mag_min) | np.isnan(tab[f'ZTF_{band}PSFMag'])
        good[band] = tab[mask]
    return good


def _psf_fit_filter(tabs):
    bad_flags = [2, 8, 16, 32]
    good = {}
    for band, tab in tabs.items():
        mask = (tab[f'ZTF_{band}PSFFlags'] == 0) | np.isnan(tab[f'ZTF_{band}PSFFlags'])
        mask &= ~_is_flag(tab[f'ZTF_{band}PSFFlags'], bad_flags)
        good[band] = tab[mask]
    return good


def _dec_greater_than(tabs, min_dec=MIN_DEC):
    return {b: t[t['dec'] > min_dec] for b, t in tabs.items()}


# ---------------------------------------------------------------------------
# Per-field KDE build
# ---------------------------------------------------------------------------

def _kde_cached(field_name: str) -> bool:
    cache_dir = os.path.join(DATA_PATH, 'kde_envelopes', field_name)
    return all(os.path.exists(os.path.join(cache_dir, f'{b}.npz')) for b in BANDS)


def build_kde_for_field(field_name: str):
    if _kde_cached(field_name):
        print(f'[{field_name}] cache hit, skipping', flush=True)
        return

    print(f'[{field_name}] loading tables...', flush=True)
    tables = {}
    for band in BANDS:
        path = os.path.join(DATA_PATH, f'{EXTRACTED_CATALOG_DIR}/{field_name}_{band}.hdf5')
        try:
            tables[band] = Table.read(path, path='data')
        except FileNotFoundError:
            print(f'[{field_name}] WARNING: missing band {band}, skipping', flush=True)
            return

    # Upper-limit and mask-removal (mirrors filter_field)
    for band in BANDS:
        for b in BANDS:
            if f'PSTARR_{b}PSFMag' not in tables[band].colnames:
                tables[band][f'PSTARR_{b}PSFMag'] = -999 * np.ones(len(tables[band]))
        tab = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)

    # Quality-filter pipeline (mirrors the throwaway _qfilters block in filter_field)
    all_q = {b: t.copy() for b, t in tables.items()}
    all_q = _sep_extraction_filter(all_q)
    all_q, ztf_low, pstarr_low = _snr_filter_both_cat(all_q)

    tabs0 = {b: tables[b][tables[b]['Catalog_Flag'] == 0] for b in tables}
    processed_bands = list(tables.keys())
    for band in processed_bands:
        if band in ztf_low:
            ztf_low[band]['Catalog_Flag'] = 1
        if band in pstarr_low:
            pstarr_low[band]['Catalog_Flag'] = 2
        if band not in all_q:
            all_q[band] = tabs0[band][:0].copy()
        if band not in ztf_low:
            ztf_low[band] = tabs0[band][:0].copy()
            ztf_low[band]['Catalog_Flag'] = 1
        if band not in pstarr_low:
            pstarr_low[band] = tabs0[band][:0].copy()
            pstarr_low[band]['Catalog_Flag'] = 2
    all_q = {b: vstack([all_q[b], ztf_low[b], pstarr_low[b]]) for b in processed_bands}
    del ztf_low, pstarr_low, tabs0, tables

    all_q = _shape_filter(all_q)
    all_q = _pstarr_not_saturated(all_q)
    all_q = _psf_fit_filter(all_q)
    all_q = _dec_greater_than(all_q)

    in_both = {b: t[t['Catalog_Flag'] == 0] for b, t in all_q.items()}
    del all_q

    print(f'[{field_name}] building KDE envelopes...', flush=True)
    load_or_build_kde_envelopes(field_name, in_both)
    del in_both
    print(f'[{field_name}] done', flush=True)


def main():
    # If a field name is passed as an argument, process just that one field.
    # This is used by the SLURM job array (prefetch_kde_one_field.sh).
    if len(sys.argv) > 1:
        build_kde_for_field(sys.argv[1])
        return

    g_fields = np.load(os.path.join(DATA_PATH, 'g_imaged_fields.npy'))
    r_fields = np.load(os.path.join(DATA_PATH, 'r_imaged_fields.npy'))
    i_fields = np.load(os.path.join(DATA_PATH, 'i_imaged_fields.npy'))
    gri_fields = np.intersect1d(np.intersect1d(g_fields, r_fields), i_fields)
    fields = [str(f).zfill(6) for f in sorted(gri_fields)]

    todo = [f for f in fields if not _kde_cached(f)]
    print(f'{len(todo)} / {len(fields)} fields need KDE pre-population', flush=True)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(build_kde_for_field, f): f for f in todo}
        for future in as_completed(futures):
            field = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'ERROR: {field} failed with {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
