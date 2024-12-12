import numpy as np
import schemdraw
import pandas as pd

from typing import Dict, Union
from astropy.io import ascii
from astropy.table import Table
from schemdraw.flow import Start, Arrow, Box
from schemdraw.elements import Annotate
from decimal import Decimal, ROUND_HALF_UP

from Source_Analysis.Sources import Sources
from Extracting.utils import get_snr_from_mag

BANDS = ['g', 'r', 'i']
CATALOG_KEY = {0: 'ZTF and Pan-STARRS', 1: 'ZTF', 2: 'Pan-STARRS', 3: 'Out of Coverage'}


def _is_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    return np.bitwise_and(flags.astype(int), [flag for _ in range(len(flags))]) != 0


def format_two_sig_figs(num):
    d = Decimal(str(num))
    # Determine how many digits we need to shift to get two significant figures.
    # adjusted() gives the position of the most significant digit.
    # For two significant figures, the exponent = (# of sig figs - adjusted_position - 1)
    shift = 2 - d.adjusted() - 1
    # Create a quantization pattern like '1e-1', '1e0', '1e-2', etc.
    quant = Decimal('1e{}'.format(-shift))
    rounded = d.quantize(quant, rounding=ROUND_HALF_UP)

    return str(rounded)


def catalog_filter(tab: Table, copy: bool = True, **kwargs) -> Table:
    # 0 — Both catalogs
    # 1 — ZTF, not Pan-STARRS, and not nan in either
    # 2 — Pan-STARRS, not ZTF, and not nan in either
    # 3 — Nan in either Pan-STARRS or ZTF
    if copy: tab = tab.copy()
    mask = np.isin(tab[f'Catalog_Flag'], [0, 1, 2])
    return tab[mask]


def psf_fit_filter(tabs: Table, copy: bool = True) -> Table:
    # Flags here https://photutils.readthedocs.io/en/stable/api/photutils.psf.PSFPhotometry.html#photutils.psf.PSFPhotometry.__call__
    for band in tabs.keys():
        tab = tabs[band]
        mask = np.logical_or(tab[f'ZTF_{band}PSFFlags'] == 0, np.isnan(tab[f'ZTF_{band}PSFFlags']))
        mask = np.logical_or(mask, tab[f'ZTF_{band}PSFFlags'] == 4)
        tabs[band] = tab[mask]

    return tabs


def sep_extraction_filter(tabs: Dict[str, Table], copy: bool = True, **kwargs) -> Table:
    # flags here: https://sextractor.readthedocs.io/en/latest/Flagging.html
    for band in tabs.keys():
        tab = tabs[band]
        okay_flags = [1, 2, 3]
        mask = np.logical_or(tab[f'ZTF_sepExtractionFlag'] == 0, np.isnan(tab[f'ZTF_sepExtractionFlag']))
        for f in okay_flags:
            mask = np.logical_or(mask, _is_flag(tab[f'ZTF_sepExtractionFlag'], f))
        tabs[band] = tab[mask]

    return tabs


def snr_filter(tabs: Table, snr_min: float = 5, copy: bool = True) -> Table:
    for band in tabs.keys():
        tab = tabs[band]

        # PSF
        pstarr_psf_snr = get_snr_from_mag(tab[f'PSTARR_{band}PSFMag'], tab[f'PSTARR_{band}PSFMagErr'], zp=25)
        ztf_psf_snr = get_snr_from_mag(tab[f'ZTF_{band}PSFMag'], tab[f'ZTF_{band}PSFMagErr'], zp=25)
        psf_mask = np.logical_or(
            np.logical_and((pstarr_psf_snr > snr_min), (ztf_psf_snr > snr_min)),
            np.logical_or(
                np.logical_and(np.isnan(pstarr_psf_snr), (ztf_psf_snr > snr_min)),
                np.logical_and((pstarr_psf_snr > snr_min), np.isnan(ztf_psf_snr)),
            ),
        )

        # Kron
        ztf_kron_snr = get_snr_from_mag(tab[f'ZTF_{band}KronMag'], tab[f'ZTF_{band}KronMagErr'], zp=25)
        pstarr_kron_snr = get_snr_from_mag(tab[f'PSTARR_{band}KronMag'], tab[f'PSTARR_{band}KronMagErr'], zp=25)
        kron_mask = np.logical_or(
            np.logical_and((pstarr_kron_snr > snr_min), (ztf_kron_snr > snr_min)),
            np.logical_or(
                np.logical_and(np.isnan(pstarr_kron_snr), (ztf_kron_snr > snr_min)),
                np.logical_and((pstarr_kron_snr > snr_min), np.isnan(ztf_kron_snr)),
            ),
        )

        # Total
        mask = np.logical_and(psf_mask, kron_mask)
        tabs[band] = tab[mask]

    return tabs

def shape_filter(tabs: Table, copy: bool = True, **kwargs) -> Table:
    for band in tabs.keys():
        tab = tabs[band]
        good_shape_mask = (tab['ZTF_a'] / tab['ZTF_b']) < 2                                                 # a/b < 1.75
        good_shape_mask &= np.logical_or((tab['ZTF_a'] / tab['ZTF_b']) < 1.35, tab['ZTF_tnpix'] < 200)      # a/b < 1.5 for big sources
        good_shape_mask |= np.isnan(tab['ZTF_a'])
        tabs[band] = tab[good_shape_mask]

    return tabs


# def preliminary_filter(original_tab: Table, band: str) -> Table:
#     tab = original_tab.copy()
#     for filt in [catalog_filter, psf_fit_filter, sep_extraction_filter, snr_filter, shape_filter]:
#         tab = filt(tab, band=band, copy=False)
#         print(f'{filt.__name__} filtered out {len(original_tab) - len(tab)} / {len(original_tab)} = {(len(original_tab) - len(tab)) / len(original_tab):.4f} sources from {band} band.')
#     return tab


def remove_mask(tab: Table) -> Table:
    tab = Table(tab, masked=True)

    # Convert masked entries to np.nan
    for col in tab.colnames:
        column = tab[col]
        if np.issubdtype(column.dtype, np.number):
            # Convert column to float if numeric to allow np.nan
            tab[col] = column.astype(float)
            tab[col][column.mask] = np.nan
        else:
            # Convert to object dtype for non-numeric columns
            tab[col] = column.astype(object)
            tab[col][column.mask] = np.nan

    return Table(tab, masked=False)


def only_big_dmag(tabs: Dict[str, Table], mag_thresh: int = 5) -> Table:
    """Filter out sources with a delta mag > `mag_thresh`"""
    min_mag = 15.0
    for band in tabs.keys():
        tab = tabs[band]

        # Get the 3-sigma boundaries on delta mag
        mag_bin_edges = np.arange(min_mag, np.nanmax(tab[f'PSTARR_{band}PSFMag']) + 1, step=1)
        bin_means, bin_stds = [], []
        both_mask = (tab['Catalog_Flag'] == 0) & (tab[f'PSTARR_{band}PSFMag'] != -999)
        outside_mask = np.ones(len(tab), dtype=bool)
        for i in range(len(mag_bin_edges) - 1):
            lower, upper = mag_bin_edges[i], mag_bin_edges[i+1]
            bin_mask = both_mask & (tab[f'PSTARR_{band}PSFMag'] > lower) & (tab[f'PSTARR_{band}PSFMag'] < upper)
            bin_means.append(np.nanmean(tab[bin_mask][f"ZTF_{band}PSFMag"] - tab[bin_mask][f"PSTARR_{band}PSFMag"]))
            bin_stds.append(np.nanstd(tab[bin_mask][f"ZTF_{band}PSFMag"] - tab[bin_mask][f"PSTARR_{band}PSFMag"]))

            outside_mask[bin_mask] = np.abs(tab[bin_mask][f"ZTF_{band}PSFMag"] - tab[bin_mask][f"PSTARR_{band}PSFMag"] - bin_means[-1]) > mag_thresh * bin_stds[-1]

        # Masks
        both_mask = (tab['Catalog_Flag'] == 0) & (tab[f'PSTARR_{band}PSFMag'] != -999)
        gtr_mag_min_mask = tab[f'PSTARR_{band}PSFMag'] > min_mag
        outside_mask &= both_mask
        outside_mask &= gtr_mag_min_mask  # ignore anything less than some magnitude

        tabs[band] = tab[outside_mask]

    return tabs


# def big_dmag_in_n_bands(srcs: Dict[str, Sources], n: int = 2) -> Dict[str, Sources]:
#     """Filter out sources that are not detected in at least `n` bands."""
#     for band, band_srcs in srcs.items():
#         dmags = [band_srcs.data[f'ZTF_{band}PSFMag'] - band_srcs.data[f'PSTARR_{band}PSFMag'] for band in ['g', 'r', 'i']]

#         mask = np.sum(~np.isnan([band_srcs.data[f'ZTF_{band}PSFMag'] for band in ['g', 'r', 'i']]), axis=0) >= n
#         print(mask)
#         srcs[band] = band_srcs[mask]

#     return srcs


def at_least_n_bands(sources: Dict[str, Sources], n: int = 2) -> Dict[str, Sources]:
    for band in sources.keys():

        # Masks for whether source is in a band
        in_g = ~np.isnan(sources[band].data[f'ZTF_g_ra'])
        in_r = ~np.isnan(sources[band].data[f'ZTF_r_ra'])
        in_i = ~np.isnan(sources[band].data[f'ZTF_i_ra'])
        mask = np.sum([in_g, in_r, in_i], axis=0) >= n

        # Mask and annotate the filter info
        for i, (g_flag, r_flag, i_flag) in enumerate(zip(in_g, in_r, in_i)):
            sources[band][i].filter_info['in_bands'] = list(np.array(['g', 'r', 'i'])[[g_flag, r_flag, i_flag]])
        sources[band] = sources[band][mask]

    return sources


def parallax_filter(sources: Dict[str, Sources]) -> Dict[str, Sources]:
    for band in sources.keys():
        mask = np.ones(len(sources[band]), dtype=bool)
        for i, src in enumerate(sources[band]):
            if len(src.GAIA_info) > 0:
                mask[i] = src.GAIA_info[0]['parallax_over_error'] < 5.0

        sources[band] = sources[band][mask]

    return sources


def proper_motion_filter(sources: Dict[str, Sources]) -> Dict[str, Sources]:
    for band in sources.keys():
        mask = np.ones(len(sources[band]), dtype=bool)
        for i, src in enumerate(sources[band]):
            if len(src.GAIA_info) > 0:
                mask[i] = src.GAIA_info[0]['pm'] / (src.GAIA_info[0]['pmra_error'] + src.GAIA_info[0]['pmdec_error']) < 5.0

        sources[band] = sources[band][mask]

    return sources


def add_filt(filt_name: str, tabs: Dict[str, Union[Table, Sources]], d: schemdraw.Drawing, init_counts: pd.Series, start: bool = False) -> schemdraw.Drawing:
    """Add filter to the flowchart."""
    # Make the label
    stats = ''
    counts = {band: len(tab) for band, tab in tabs.items()}  # pd.Series(tab['Catalog_Flag']).value_counts()
    counts_norm = {band: counts[band] / init_counts[band] if band in counts.keys() else 0.0 for band in init_counts.keys()}
    for band in init_counts.keys():
        ct = counts[band] if band in counts.keys() else 0
        ct_norm = counts_norm[band] if band in counts_norm.keys() else 0.0
        stats += f'{band}: {ct:,} ({format_two_sig_figs(ct_norm)})\n'

    # Add to graph
    if start:
        d += Start().label(stats)
    else:
        arrow = Arrow().down(d.unit/2).label(filt_name)
        d += arrow
        # Annotate the arrow with the filter name
        # d += Annotate().at(arrow.center).delta(dx=0.5)
        d += Box().label(stats)

    return d


def add_final_filt(filt_name: str, srcs: Sources, d: schemdraw.Drawing) -> schemdraw.Drawing:
    """Add final filter to the flowchart."""
    arrow = Arrow().down(d.unit/2).label(filt_name)
    d += arrow
    d += Box().label(f'Total: {len(srcs.data):,}')

    return d


def filter_tables():
    # Load in the tables
    print('Loading tables...')
    g_tab = ascii.read('/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results/000499_g.ecsv', format='ecsv')
    r_tab = ascii.read('/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results/000499_r.ecsv', format='ecsv')
    i_tab = ascii.read('/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results/000499_i.ecsv', format='ecsv')
    tables = {'g': g_tab, 'r': r_tab, 'i': i_tab}

    # Set values <=0 to the upper limit for ZTF
    for band in tables.keys():
        tab = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]   # 4 means flux was negative
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)


    ########## Filtering for sources detected in both! ##########
    print(f'Building flowchart for {CATALOG_KEY[0]} graph...')
    tabs = {band: tab.copy()[tab['Catalog_Flag'] == 0] for band, tab in tables.items()}

    with schemdraw.Drawing() as d:

        # Initial stats
        init_counts = {band: len(tab) for band, tab in tabs.items()}
        add_filt('Initial Counts', tabs, d, init_counts=init_counts, start=True)

        # Drop all sources with bad SEP extraction flags
        tabs = sep_extraction_filter(tabs)
        add_filt('SEP Extraction Flags', tabs, d, init_counts=init_counts)

        # Drop all sources with snr < 5
        tabs = snr_filter(tabs, snr_min=5)
        add_filt('SNR > 5', tabs, d, init_counts=init_counts)

        # Axis ratio filter
        tabs = shape_filter(tabs)
        add_filt('Axis Ratio', tabs, d, init_counts=init_counts)

        # Drop bad PSF fits
        tabs = psf_fit_filter(tabs)
        add_filt('PSF Fit', tabs, d, init_counts=init_counts)

        # Delta mag > n sigma
        dmag_sigma = 5
        tabs = only_big_dmag(tabs, mag_thresh=dmag_sigma)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', tabs, d, init_counts=init_counts)

        # Converting to sources
        sources = {
            band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=tabs) for band, tab in tabs.items()
        }

        # Check for big dmag in >1 bands
        sources = at_least_n_bands(sources, n=2)
        add_filt(f'Big dmag in >=2 bands', sources, d, init_counts=init_counts)

        # TESTING!!!
        for band, srcs in sources.items():
            srcs.save(f'/Users/adamboesky/Research/long_transients/Data/filter_testing/0_{band}_pre_gaia.ecsv')

        # Check for parallax
        sources = parallax_filter(sources)
        add_filt(f'Parallax', sources, d, init_counts=init_counts)

        # Check for proper motion
        sources = proper_motion_filter(sources)
        add_filt(f'Proper Motion', sources, d, init_counts=init_counts)

        # Save the image
        d.save(f'Figures/0_filtering_flowchart.pdf')
        for band, srcs in sources.items():
            srcs.save(f'/Users/adamboesky/Research/long_transients/Data/filter_testing/0_{band}.ecsv')


    ########## Filtering for sources in both! ##########
    tabs = {band: tab.copy()[tab['Catalog_Flag'] == 0] for band, tab in tables.items()}


if __name__ == '__main__':
    filter_tables()
