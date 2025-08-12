import os
import sys
import shutil
import pickle
import numpy as np
import schemdraw
import pandas as pd

from typing import Dict, List, Iterable, Tuple, Union, Optional
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from schemdraw.flow import Start, Arrow, Box, Decision
from schemdraw.elements import Element
from decimal import Decimal, ROUND_HALF_UP

sys.path.append('/Users/adamboesky/Research/long_transients')

from Source_Analysis.Sources import Sources, MANDATORY_SOURCE_COLUMNS
from Extracting.utils import get_snr_from_mag, get_data_path, load_ecsv
from concurrent.futures import ProcessPoolExecutor

BANDS = ['g', 'r', 'i']
CATALOG_KEY = {0: 'ZTF and Pan-STARRS', 1: 'ZTF', 2: 'Pan-STARRS', 3: 'Out of Coverage'}
PSTARR_UPPER_LIM = {'g': 23.3, 'r': 23.2, 'i': 23.1}


def _is_flag(flags: np.ndarray, flag: Union[int, Iterable[int]]) -> np.ndarray:
    """
    Check if one or more bitwise flags are set in the flags array.

    Args:
        flags: np.ndarray of integer flags.
        flag: int or iterable of ints, the flag(s) to check.

    Returns:
        np.ndarray of bool, True where any of the specified flags are set.
    """
    flags = flags.astype(int)
    if isinstance(flag, int):
        return np.bitwise_and(flags, flag) != 0
    else:
        # If flag is an iterable, check if any of the flags are set
        mask = np.zeros_like(flags, dtype=bool)
        for f in flag:
            mask |= (np.bitwise_and(flags, f) != 0)
        return mask


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


def remove_mask(tab: Table, *args, **kwargs) -> Table:
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


def get_merged_tab_coords(tabs: Iterable[Table], max_arcsec: float = 1.0) -> Table:
    """Given an iterable of astropy Table objects, return a table of their associated ra and decs."""
    if not isinstance(tabs, list):
        tabs = list(tabs)

    # Need to start with a non-empty table
    if len(tabs) == 0:
        return Table(data={'ra': [], 'dec': []})
    while True:
        merged_tab = tabs[0].copy()
        if len(merged_tab) == 0:
            if len(tabs) > 1:
                tabs = tabs[1:]
            else:
                return merged_tab
        else:
            break
    merged_tab = merged_tab[['ra', 'dec']]

    for tab in tabs[1:]:

        # Skip if tab is empty
        if len(tab) == 0:
            continue

        # Get angular separations
        merged_coords = SkyCoord(merged_tab['ra'], merged_tab['dec'], unit='deg')
        tab_coords = SkyCoord(tab['ra'], tab['dec'], unit='deg')
        idx, sep2d, _ = match_coordinates_sky(tab_coords, merged_coords)

        # Associate
        same_src_mask = sep2d.arcsecond <= max_arcsec
        merged_tab = vstack((merged_tab, tab[~same_src_mask][['ra', 'dec']]))

    return merged_tab


class Filters():
    def __init__(self, filter_stat_fname: Optional[str] = None):
        self.filters = {
            'catalog_filter': self.catalog_filter,
            'psf_fit_filter': self.psf_fit_filter,
            'sep_extraction_filter': self.sep_extraction_filter,
            'snr_filter': self.snr_filter,
            'shape_filter': self.shape_filter,
            'only_big_dmag': self.only_big_dmag,
            'at_least_n_bands': self.at_least_n_bands,
            'parallax_filter': self.parallax_filter,
            'proper_motion_filter': self.proper_motion_filter,
            'no_nearby_source_filter': self.no_nearby_source_filter,
            'pstarr_mag_less_than': self.pstarr_mag_less_than,
            'dec_greater_than': self.dec_greater_than,
            'pstarr_not_saturated': self.pstarr_not_saturated,
            'pstarr_quality_filter': self.pstarr_quality_filter,
        }
        self.reset_filter_stats()
        self.filter_stat_fname = filter_stat_fname

    def reset_filter_stats(self):
        """Reset the filter statistics."""
        self.filter_stats = pd.DataFrame(data={
            'filter': [],
            'n_g_before': [],
            'n_g_after': [],
            'n_r_before': [],
            'n_r_after': [],
            'n_i_before': [],
            'n_i_after': [],
            'n_gri_before': [],
            'n_gri_after': [],
            'n_gr_before': [],
            'n_gr_after': [],
            'n_gi_before': [],
            'n_gi_after': [],
            'n_ri_before': [],
            'n_ri_after': [],
        })

        self.filtered_out: Dict[str, Table] = {
            'g': Table(data={'ra': [], 'dec': [], 'filter': []}, dtype=[float, float, str]),
            'r': Table(data={'ra': [], 'dec': [], 'filter': []}, dtype=[float, float, str]),
            'i': Table(data={'ra': [], 'dec': [], 'filter': []}, dtype=[float, float, str]),
        }

    def save_filter_stats(self):
        """Save the filtration statistics to a csv file."""
        # Fill nan values with an empty string
        if 'branch' in self.filter_stats.columns:
            self.filter_stats['branch'] = self.filter_stats['branch'].fillna('')

        # Save stats
        self.filter_stats.to_csv(self.filter_stat_fname, index=False)

    def save_filtered_out(self, out_dir: str, cat: int, overwrite: bool = True):
        """Save the filtered out sources to a file."""
        for band in ('g', 'r', 'i'):
            self.filtered_out[band].write(
                os.path.join(out_dir, f'{cat}_{band}_filtered_out.ecsv'),
                format='ascii.ecsv',
                overwrite=overwrite,
            )

    def filter(
            self,
            tabs: Union[Dict[str, Union[Table, Sources]], Sources],
            filt_name: str,
            *args,
            branch: str = '',
            **kwargs,
        ) -> Union[Dict[str, Union[Table, Sources]], Sources]:
        """Filter the tables using the specified filter."""
        # If we are given something of length zero, just return it and log all zeros
        if len(tabs) == 0:

            # Update the filter stats with all zeros
            all_zeros = {col: [0] for col in self.filter_stats.columns if col not in ('filter', 'branch')}
            all_zeros.update({'filter': [filt_name]})
            all_zeros.update({'branch': [branch]})
            self.filter_stats = pd.concat(
                (
                    self.filter_stats,
                    pd.DataFrame(data=all_zeros),
                )
            )

            # Save filtration stats if filename is given
            if self.filter_stat_fname is not None:
                self.save_filter_stats()

            if filt_name == 'only_big_dmag':
                return {}, {}, {}
            return tabs

        # Make into a dict if merged
        is_merged = isinstance(tabs, Sources)
        if is_merged:
            tabs = {'merged': tabs}

        # Filtration stats
        before_counts = {}
        for band, tab in tabs.items():
            
            # at_least_n_bands merges the bands so the counts before are the origin numbers.
            # If the catalog is merged, we cound the number of sources in each combbination of bands.
            # If the catalog is not merged, we simply count the number of sources in each band.
            if filt_name == 'at_least_n_bands':
                for band in ('gri', 'gr', 'gi', 'ri'):
                    before_counts[band] = len(tab)
            elif band == 'merged':
                before_counts['gri'] = np.sum(tab.in_g & tab.in_r & tab.in_i)
                before_counts['gr'] = np.sum(tab.in_g & tab.in_r & ~tab.in_i)
                before_counts['gi'] = np.sum(tab.in_g & ~tab.in_r & tab.in_i)
                before_counts['ri'] = np.sum(~tab.in_g & tab.in_r & tab.in_i)
            else:
                before_counts[band] = len(tab)

        # Filter
        good_tabs, bad_tabs, *other_returns = self.filters[filt_name](tabs, *args, **kwargs)

        # Add bad_tabs to filtered out
        for band, tab in bad_tabs.items():
            if band == 'merged':
                if len(tab.data) > 0:
                    tab_shortened = tab.data[['ra', 'dec']]
                    tab_shortened['filter'] = filt_name
                    self.filtered_out['g'] = vstack((self.filtered_out['g'], tab_shortened[tab.in_g]))
                    self.filtered_out['r'] = vstack((self.filtered_out['r'], tab_shortened[tab.in_r]))
                    self.filtered_out['i'] = vstack((self.filtered_out['i'], tab_shortened[tab.in_i]))
            elif len(tab) > 0:
                tab = tab[['ra', 'dec']]
                tab['filter'] = filt_name
                self.filtered_out[band] = vstack((self.filtered_out[band], tab))

        # Get the counts after filtration
        after_counts = {}
        for band, tab in good_tabs.items():
            if band == 'merged':
                after_counts['gri'] = np.sum(tab.in_g & tab.in_r & tab.in_i)
                after_counts['gr'] = np.sum(tab.in_g & tab.in_r & ~tab.in_i)
                after_counts['gi'] = np.sum(tab.in_g & ~tab.in_r & tab.in_i)
                after_counts['ri'] = np.sum(~tab.in_g & tab.in_r & tab.in_i)
            else:
                after_counts[band] = len(tab)

        # Update filtration stats
        self.filter_stats = pd.concat(
            (
                self.filter_stats,
                pd.DataFrame(
                    data={
                        'filter': filt_name,
                        'n_g_before': [before_counts.get('g', 0)],
                        'n_g_after': [after_counts.get('g', 0)],
                        'n_r_before': [before_counts.get('r', 0)],
                        'n_r_after': [after_counts.get('r', 0)],
                        'n_i_before': [before_counts.get('i', 0)],
                        'n_i_after': [after_counts.get('i', 0)],
                        'n_gri_before': [before_counts.get('gri', 0)],
                        'n_gri_after': [after_counts.get('gri', 0)],
                        'n_gr_before': [before_counts.get('gr', 0)],
                        'n_gr_after': [after_counts.get('gr', 0)],
                        'n_gi_before': [before_counts.get('gi', 0)],
                        'n_gi_after': [after_counts.get('gi', 0)],
                        'n_ri_before': [before_counts.get('ri', 0)],
                        'n_ri_after': [after_counts.get('ri', 0)],
                        'branch': [branch],
                    }
                )
            )
        )

        # Save filtration stats if filename is given
        if self.filter_stat_fname is not None:
            self.save_filter_stats()

        # Drop tables if they are empty
        bands = list(good_tabs.keys())
        for band in bands:
            if len(good_tabs[band]) == 0:
                good_tabs.pop(band)

        # If merged, just return the table
        if is_merged:
            good_tabs = good_tabs.get('merged', Sources(ras=[], decs=[]))
            bad_tabs = bad_tabs.get('merged', Table())

        if len(other_returns) == 0:
            return good_tabs
        return tuple([good_tabs] + other_returns)

    def catalog_filter(self, tab: Table, copy: bool = True, **kwargs) -> Table:
        # 0 — Both catalogs
        # 1 — ZTF, not Pan-STARRS, and not nan in either
        # 2 — Pan-STARRS, not ZTF, and not nan in either
        # 3 — Nan in either Pan-STARRS or ZTF
        if copy: tab = tab.copy()
        mask = np.isin(tab[f'Catalog_Flag'], [0, 1, 2])
        return tab[mask]

    def psf_fit_filter(self, tabs: Dict[str, Table], copy: bool = True, *args, **kwargs) -> Table:
        # Flags here https://photutils.readthedocs.io/en/stable/api/photutils.psf.PSFPhotometry.html#photutils.psf.PSFPhotometry.__call__
        good_tabs = {}
        bad_tabs = {}
        bad_flags = [2, 8, 16, 32]
        for band in tabs.keys():
            tab = tabs[band]
            mask = np.logical_or(tab[f'ZTF_{band}PSFFlags'] == 0, np.isnan(tab[f'ZTF_{band}PSFFlags']))
            mask &= ~_is_flag(tab[f'ZTF_{band}PSFFlags'], bad_flags)
            good_tabs[band] = tab[mask]
            bad_tabs[band] = tab[~mask]

        return good_tabs, bad_tabs

    def sep_extraction_filter(self, tabs: Dict[str, Table], copy: bool = True, *args, **kwargs) -> Table:
        # flags here: https://sextractor.readthedocs.io/en/latest/Flagging.html
        good_tabs = {}
        bad_tabs = {}
        bad_flags = [4, 8, 16, 32, 64, 128]
        for band in tabs.keys():
            tab = tabs[band]
            mask = np.logical_or(tab['ZTF_sepExtractionFlag'] == 0, np.isnan(tab['ZTF_sepExtractionFlag']))
            mask &= ~_is_flag(tab['ZTF_sepExtractionFlag'], bad_flags)
            good_tabs[band] = tab[mask]
            bad_tabs[band] = tab[~mask]

        return good_tabs, bad_tabs


    def snr_filter(
            self,
            tabs: Dict[str, Table],
            snr_min: float = 5,
            both_cat: bool = False,
            copy: bool = True,
            *args,
            **kwargs,
        ) -> Table:
        good_tabs = {}
        bad_tabs = {}

        ztf_tabs = {}
        pstarr_tabs = {}
        for band in tabs.keys():
            tab = tabs[band]

            if both_cat:
                # PSF
                pstarr_psf_snr = get_snr_from_mag(
                    tab[f'PSTARR_{band}PSFMag'],
                    tab[f'PSTARR_{band}PSFMagErr'],
                    zp=25,
                )
                ztf_psf_snr = get_snr_from_mag(
                    tab[f'ZTF_{band}PSFMag'],
                    tab[f'ZTF_{band}PSFMagErr'],
                    zp=np.nan_to_num(tab[f'ZTF_{band}_zero_pt_mag'], nan=25),
                )
                psf_mask = np.logical_or(
                    np.logical_and((pstarr_psf_snr > snr_min), (ztf_psf_snr > snr_min)),
                    np.logical_or(
                        np.logical_and(np.isnan(pstarr_psf_snr), (ztf_psf_snr > snr_min)),
                        np.logical_and((pstarr_psf_snr > snr_min), np.isnan(ztf_psf_snr)),
                    ),
                )

                # Kron
                ztf_kron_snr = get_snr_from_mag(
                    tab[f'ZTF_{band}KronMag'],
                    tab[f'ZTF_{band}KronMagErr'],
                    zp=np.nan_to_num(tab[f'ZTF_{band}_zero_pt_mag'], nan=25),
                )
                pstarr_kron_snr = get_snr_from_mag(
                    tab[f'PSTARR_{band}KronMag'],
                    tab[f'PSTARR_{band}KronMagErr'],
                    zp=25)
                kron_mask = np.logical_or(
                    np.logical_and((pstarr_kron_snr > snr_min), (ztf_kron_snr > snr_min)),
                    np.logical_or(
                        np.logical_and(np.isnan(pstarr_kron_snr), (ztf_kron_snr > snr_min)),
                        np.logical_and((pstarr_kron_snr > snr_min), np.isnan(ztf_kron_snr)),
                    ),
                )

                # Get all the sources for which exclusively ZTF *OR* Pan-STARRS has SNR > 5
                ztf_psf_mask = (ztf_psf_snr > snr_min) & np.logical_not(psf_mask)
                ztf_kron_mask = (ztf_kron_snr > snr_min) & np.logical_not(kron_mask)
                ztf_mask = np.logical_or(ztf_psf_mask, ztf_kron_mask)
                pstarr_psf_mask = (pstarr_psf_snr > snr_min) & np.logical_not(psf_mask)
                pstarr_kron_mask = (pstarr_kron_snr > snr_min) & np.logical_not(kron_mask)
                pstarr_mask = np.logical_or(pstarr_psf_mask, pstarr_kron_mask)

                # Fill in tabs
                ztf_tabs[band] = tab[ztf_mask]
                pstarr_tabs[band] = tab[pstarr_mask]

            else:
                # PSF
                pstarr_psf_snr = get_snr_from_mag(
                    tab[f'PSTARR_{band}PSFMag'],
                    tab[f'PSTARR_{band}PSFMagErr'],
                    zp=25,
                )
                ztf_psf_snr = get_snr_from_mag(
                    tab[f'ZTF_{band}PSFMag'],
                    tab[f'ZTF_{band}PSFMagErr'],
                    zp=np.nan_to_num(tab[f'ZTF_{band}_zero_pt_mag'], nan=25),
                )
                psf_mask = np.logical_or(
                    np.logical_or((pstarr_psf_snr > snr_min), (ztf_psf_snr > snr_min)),
                    np.logical_or(
                        np.logical_and(np.isnan(pstarr_psf_snr), (ztf_psf_snr > snr_min)),
                        np.logical_and((pstarr_psf_snr > snr_min), np.isnan(ztf_psf_snr)),
                    ),
                )

                # Kron
                ztf_kron_snr = get_snr_from_mag(
                    tab[f'ZTF_{band}KronMag'],
                    tab[f'ZTF_{band}KronMagErr'],
                    zp=np.nan_to_num(tab[f'ZTF_{band}_zero_pt_mag'], nan=25),
                )
                pstarr_kron_snr = get_snr_from_mag(
                    tab[f'PSTARR_{band}KronMag'],
                    tab[f'PSTARR_{band}KronMagErr'],
                    zp=25,
                )
                kron_mask = np.logical_or(
                    np.logical_or((pstarr_kron_snr > snr_min), (ztf_kron_snr > snr_min)),
                    np.logical_or(
                        np.logical_and(np.isnan(pstarr_kron_snr), (ztf_kron_snr > snr_min)),
                        np.logical_and((pstarr_kron_snr > snr_min), np.isnan(ztf_kron_snr)),
                    ),
                )

            # Total
            mask = np.logical_and(psf_mask, kron_mask)
            good_tabs[band] = tab[mask]
            bad_tabs[band] = tab[~mask]

        if both_cat:
            return good_tabs, bad_tabs, ztf_tabs, pstarr_tabs
        else:
            return good_tabs, bad_tabs

    def shape_filter(self, tabs: Table, copy: bool = True, *args, **kwargs) -> Table:
        good_tabs = {}
        bad_tabs = {}
        for band in tabs.keys():
            tab = tabs[band]
            good_shape_mask = (tab['ZTF_a'] / tab['ZTF_b']) < 2                                                 # a/b < 2.0
            good_shape_mask &= np.logical_or((tab['ZTF_a'] / tab['ZTF_b']) < 1.35, tab['ZTF_tnpix'] < 200)      # a/b < 1.35 for big sources
            good_shape_mask |= np.isnan(tab['ZTF_a'])
            good_tabs[band] = tab[good_shape_mask]
            bad_tabs[band] = tab[~good_shape_mask]

        return good_tabs, bad_tabs

    def only_big_dmag(
            self,
            tabs: Dict[str, Table],
            mag_thresh: int = 5,
            upper_lim: Optional[str] = None,
            bin_means: Optional[Dict[str, List[float]]] = None,
            bin_stds: Optional[Dict[str, List[float]]] = None,
            *args,
            **kwargs,
        ) -> Tuple[Table, Dict[str, List[float]], Dict[str, List[float]]]:
        """Filter out sources with a delta mag > `mag_thresh`"""
        if upper_lim is not None and upper_lim not in ('ZTF', 'PSTARR'):
            raise ValueError('Upper limit must be None or `PSTARR` or `ZTF`.')

        if len(tabs) == 0:
            return {}, {}, {}, {}

        good_tabs = {}
        bad_tabs = {}

        # Check if we are given bins to use
        given_bins = bin_means is not None and bin_stds is not None
        if not given_bins:
            bin_means, bin_stds = {}, {}

        min_mag = 15.0
        for band in tabs.keys():
            tab = tabs[band]

            # Get the mags that we are going to use based on the string
            if upper_lim == 'PSTARR':
                pstarr_mags = np.ones(len(tab[f'PSTARR_{band}PSFMag'])) * PSTARR_UPPER_LIM[band]
                ztf_mags = np.array(tab[f'ZTF_{band}PSFMag'])
            elif upper_lim == 'ZTF':
                ztf_mags = np.array(tab[f'ZTF_{band}_mag_limit'])
                pstarr_mags = np.array(tab[f'PSTARR_{band}PSFMag'])
            else:
                pstarr_mags = np.array(tab[f'PSTARR_{band}PSFMag'])
                ztf_mags = np.array(tab[f'ZTF_{band}PSFMag'])

            # Fill in the panstarrs upper limit values  # TODO: THINK ABOUT THIS
            upper_lims = pstarr_mags == -999
            # pstarr_mags[upper_lims] = PSTARR_UPPER_LIM[band]

            # Get the n-sigma boundaries on delta mag
            mag_bin_edges = np.arange(min_mag, np.nanmax(np.concatenate((pstarr_mags, ztf_mags))) + 1, step=1)
            if not given_bins:
                bin_means[band], bin_stds[band] = [], []
            outside_mask = np.ones(len(tab), dtype=bool)
            for i in range(len(mag_bin_edges) - 1):
                lower, upper = mag_bin_edges[i], mag_bin_edges[i+1]
                bin_mask = (~upper_lims) & (pstarr_mags > lower) & (pstarr_mags < upper)
                if not given_bins:
                    bin_means[band].append(np.nanmean(ztf_mags[bin_mask] - pstarr_mags[bin_mask]))
                    bin_stds[band].append(np.nanstd(ztf_mags[bin_mask] - pstarr_mags[bin_mask]))

                outside_mask[bin_mask] = np.abs(ztf_mags[bin_mask] - pstarr_mags[bin_mask] - bin_means[band][-1]) > mag_thresh * bin_stds[band][-1]

            # Masks
            gtr_mag_min_mask = pstarr_mags > min_mag
            outside_mask &= gtr_mag_min_mask  # ignore anything less than some magnitude
            outside_mask &= (~upper_lims)

            good_tabs[band] = tab[outside_mask]
            bad_tabs[band] = tab[outside_mask]

        return good_tabs, bad_tabs, bin_means, bin_stds

    def at_least_n_bands(
            self,
            sources: Union[Sources, Dict[str, Sources]],
            n: int = 2,
            *args,
            **kwargs,
        ) -> Union[Sources, Dict[str, Sources]]:
        good_sources = {}
        bad_sources = {}
        for band in sources.keys():

            # Annotate the filter info
            band_mask = np.hstack(
                (
                    sources[band].in_g.reshape(-1,1),
                    sources[band].in_r.reshape(-1, 1),
                    sources[band].in_i.reshape(-1, 1),
                )
            )
            in_bands = [np.array(['g', 'r', 'i'])[m] for m in band_mask]
            for i, src_in_bands in enumerate(in_bands):
                sources[band][i].filter_info['in_bands'] = list(src_in_bands)

            # Mask for sources with n bands >= n
            enough_bands_mask = np.sum(band_mask, axis=1) >= n
            good_sources[band] = sources[band][enough_bands_mask]
            bad_sources[band] = sources[band][~enough_bands_mask]

        return good_sources, bad_sources

    def _parallax_in_srcs(self, srcs: Sources) -> Tuple[Sources, Sources]:
        mask = np.ones(len(srcs), dtype=bool)
        for i, src in enumerate(srcs):
            if len(src.GAIA_info) > 0:
                mask[i] = (
                    (src.GAIA_info[0]['parallax_over_error'] < 5.0)
                    or src.GAIA_info['parallax_over_error'].mask[0]
                )
        good_srcs = srcs[mask]
        bad_srcs = srcs[~mask]
        return good_srcs, bad_srcs

    def parallax_filter(
        self,
        sources: Union[Sources, Dict[str, Sources]],
        *args,
        **kwargs,
    ) -> Union[Tuple[Dict[str, Sources], Dict[str, Sources]], Tuple[Sources, Sources]]:
        if isinstance(sources, Dict):
            good_sources = {}
            bad_sources = {}
            for band in sources.keys():
                good_sources[band], bad_sources[band] = self._parallax_in_srcs(sources[band])
            return good_sources, bad_sources
        else:
            return self._parallax_in_srcs(sources)

    def _proper_motion_in_srcs(self, srcs: Sources) -> Tuple[Sources, Sources]:
        mask = np.ones(len(srcs), dtype=bool)
        for i, src in enumerate(srcs):
            if len(src.GAIA_info) > 0:
                mask[i] = (src.GAIA_info[0]['pm'] / (src.GAIA_info[0]['pmra_error'] + src.GAIA_info[0]['pmdec_error']) < 5.0) or \
                    (src.GAIA_info['pm'].mask[0] or \
                    src.GAIA_info['pmra_error'].mask[0] or \
                    src.GAIA_info['pmdec_error'].mask[0]
                    )

        good_srcs = srcs[mask]
        bad_srcs = srcs[~mask]

        return good_srcs, bad_srcs

    def proper_motion_filter(
            self,
            sources: Union[Sources, Dict[str, Sources]],
            *args,
            **kwargs,
        ) -> Union[Tuple[Dict[str, Sources], Dict[str, Sources]], Sources]:
        if isinstance(sources, Dict):
            good_sources = {}
            bad_sources = {}
            for band in sources.keys():
                good_sources[band], bad_sources[band] = self._proper_motion_in_srcs(sources[band])
            return good_sources, bad_sources
        else:
            return self._proper_motion_in_srcs(sources)
    
    def _no_nearby_source_in_srcs(self, srcs: Sources, n_nearby_max: int) -> Sources:
        all_coords = srcs.coords
        seps = all_coords.separation(all_coords[:, None])
        nearby_counts = np.sum(seps.arcsecond < 200, axis=1)
        mask = nearby_counts <= n_nearby_max
        good_srcs = srcs[mask]
        bad_srcs = srcs[~mask]

        return good_srcs, bad_srcs

    def no_nearby_source_filter(
            self,
            sources: Union[Sources, Dict[str, Sources]],
            n_nearby_max: int = 5,
            *args,
            **kwargs,
        ) -> Union[Tuple[Dict[str, Sources], Dict[str, Sources]], Sources]:
        if isinstance(sources, Dict):
            good_sources = {}
            bad_sources = {}
            for band in sources.keys():
                good_sources[band], bad_sources[band] = self._no_nearby_source_in_srcs(
                    sources[band],
                    n_nearby_max=n_nearby_max,
                )
            return good_sources, bad_sources
        else:
            return self._no_nearby_source_in_srcs(
                    sources,
                    n_nearby_max=n_nearby_max,
            )

    def pstarr_mag_less_than(self, tabs: Dict[str, Table], max_mag: float, *args, **kwargs) -> Table:
        good_tabs = {}
        bad_tabs = {}
        for band in tabs.keys():

            # In all three bands, the mag is less than max_mag
            mask = (
                (tabs[band][['PSTARR_gPSFMag', 'PSTARR_rPSFMag', 'PSTARR_iPSFMag']].to_pandas().values < max_mag)
                & (tabs[band][['PSTARR_gPSFMag', 'PSTARR_rPSFMag', 'PSTARR_iPSFMag']].to_pandas().values > 0)
            ).sum(axis=1) == 3

            good_tabs[band] = tabs[band][mask]
            bad_tabs[band] = tabs[band][~mask]

        return good_tabs, bad_tabs

    def dec_greater_than(self, tabs: Dict[str, Table], min_dec: float = -29.5, *args, **kwargs) -> Tuple[Dict[str, Table], Dict[str, Table]]:
        good_tabs = {}
        bad_tabs = {}
        for band in tabs.keys():
            mask = tabs[band]['dec'] > min_dec
            good_tabs[band] = tabs[band][mask]
            bad_tabs[band] = tabs[band][~mask]

        return good_tabs, bad_tabs

    def pstarr_not_saturated(self, tabs: Dict[str, Table], ztf_mag_min: float = 13.5, *args, **kwargs) -> Tuple[Dict[str, Table], Dict[str, Table]]:
        """Make sure the Pan-STARRS sources are not saturated. Will check both flags and make sure that ZTF mag > 13.5.
        Flags defined here: https://outerspace.stsci.edu/display/PANSTARRS/PS1+Detection+Flags
        """
        good_tabs = {}
        bad_tabs = {}
        for band in tabs.keys():

            # Make the mask
            mask = (
                ~_is_flag(tabs[band][f'PSTARR_{band}infoFlag'], 128) &
                ~_is_flag(tabs[band][f'PSTARR_{band}infoFlag'], 4096) &
                ~_is_flag(tabs[band][f'PSTARR_{band}infoFlag2'], 4096)
            )
            mask &= (
                (tabs[band][f'ZTF_{band}PSFMag'] > ztf_mag_min) |
                np.isnan(tabs[band][f'ZTF_{band}PSFMag'])
            )

            # Filter tabs
            good_tabs[band] = tabs[band][mask]
            bad_tabs[band] = tabs[band][~mask]

        return good_tabs, bad_tabs

    def pstarr_quality_filter(self, tabs: Dict[str, Table], *args, **kwargs) -> Tuple[Dict[str, Table], Dict[str, Table]]:
        """Filter out Pan-STARRS sources with bad quality flags or object info flags.
        Flags defined here: https://outerspace.stsci.edu/display/PANSTARRS/PS1+Object+Flags#PS1ObjectFlags-ColumnXinfoFlag(Xoneofg,r,i,z,y)inStackObjectThin
        """
        good_tabs = {}
        bad_tabs = {}
        for band in tabs.keys():

            # Make the mask - keep sources that do NOT have the bad flags
            mask = ~np.logical_or(
                _is_flag(tabs[band][f'PSTARR_qualityFlag'], [1, 2, 128]),
                _is_flag(tabs[band][f'PSTARR_objInfoFlag'], [8388608, 16777216, 1073741824]),
            )

            # Filter tabs
            good_tabs[band] = tabs[band][mask]
            bad_tabs[band] = tabs[band][~mask]

        return good_tabs, bad_tabs


def associate_in_btwn_distance(table1: Table, table2: Table, min_sep: float = 1.0, max_sep: float = 3.0) -> Table:
    """Associate tables using a min and max angular separation requirement.
    
    Returns:
        1. Table associated with second table in between min_sep and max_sep.
        2. table1 but without all the rows from the assocatiated table.
    """
    coords1 = SkyCoord(ra=table1['ra'], dec=table1['dec'], unit='deg')
    coords2 = SkyCoord(ra=table2['ra'], dec=table2['dec'], unit='deg')
    idx, sep2d, _ = match_coordinates_sky(coords1, coords2)
    mask = (sep2d.arcsecond <= max_sep) & (sep2d.arcsecond > min_sep)

    # Fill in the values appropriately
    associated_table = table1[mask]
    for col in table2.colnames:
        try:
            if col not in ('Catalog', 'ra', 'dec', 'ZTF_g_upper_lim_flag', 'ZTF_r_upper_lim_flag', 'ZTF_i_upper_lim_flag'):
                if np.any(np.logical_not(np.isnan(table2[idx[mask]][col].value))):
                    associated_table[col] = table2[idx[mask]][col]
        except:
            print(col)

    # Add the angular separations
    associated_table['association_separation_arcsec'] = sep2d[mask].arcsecond

    return associated_table, table1[np.logical_not(mask)]


def add_filt_from_counts(
        filt_name: str,
        counts: Dict[str, int],
        d: schemdraw.Drawing,
        init_counts: pd.Series,
        previous_element: Optional[Element] = None,
        start: bool = False,
    ) -> Tuple[schemdraw.Drawing, Element]:
    """Add filter to the flowchart."""
    # Set up some variables about the bands we are writing stats for
    indiv_bands = ('g', 'r', 'i')
    bands_are_combined = len(np.intersect1d(indiv_bands, tuple(counts.keys()))) == 0

    # Make the stats label
    stats = ''
    if bands_are_combined:
        for band, ct in counts.items():
            stats += f'{band}: {ct:,}\n'
    else:
        counts_norm = {band: 0.0 if band not in counts.keys() or init_counts[band] == 0 else counts[band] / init_counts[band] for band in indiv_bands}
        for band in init_counts.keys():
            ct = counts[band] if band in counts.keys() else 0
            ct_norm = counts_norm[band] if band in counts_norm.keys() else 0.0
            stats += f'{band}: {ct:,} ({format_two_sig_figs(ct_norm)})\n'
    stats = stats[:-1]

    # Add to graph
    if start:
        d += Start().label(stats)
        new_box = None
    else:
        if previous_element is not None:
            arrow = Arrow().down(d.unit/2).at(previous_element.S).label(filt_name)
        else:
            arrow = Arrow().down(d.unit/2).label(filt_name)
        d += arrow

        # Annotate the arrow with the filter name
        new_box = Box().label(stats)
        d += new_box

    return d, new_box


def add_filt(
        filt_name: str,
        tabs: Dict[str, Union[Table, Sources]],
        d: schemdraw.Drawing,
        init_counts: pd.Series,
        previous_element: Optional[Element] = None,
        start: bool = False,
    ) -> Tuple[schemdraw.Drawing, Element]:
    """Add filter to the flowchart."""
    # Get the counts
    counts = {band: len(tab) for band, tab in tabs.items()}

    # Add the filter
    return add_filt_from_counts(
        filt_name,
        counts,
        d,
        init_counts,
        previous_element,
        start,
    )

def add_decision_from_counts(
        filt_name: str,
        counts_yes: Dict[str, int],
        counts_no: Dict[str, int],
        d: schemdraw.Drawing,
        init_counts: pd.Series,
        previous_element: Optional[Element] = None,
    ) -> Tuple[schemdraw.Drawing, Element, Element]:
    """Add decision to flow chart."""
    # Make the labels
    stats_yes = ''
    counts_norm_yes = {band: counts_yes[band] / init_counts[band] if band in counts_yes.keys() else 0.0 for band in ('g', 'r', 'i')}
    for band in init_counts.keys():
        ct = counts_yes[band] if band in counts_yes.keys() else 0
        ct_norm = counts_norm_yes[band] if band in counts_norm_yes.keys() else 0.0
        stats_yes += f'{band}: {ct:,} ({format_two_sig_figs(ct_norm)})\n'
    stats_no = ''
    counts_norm_no = {band: counts_no[band] / init_counts[band] if band in counts_no.keys() else 0.0 for band in ('g', 'r', 'i')}
    for band in init_counts.keys():
        ct = counts_no[band] if band in counts_no.keys() else 0
        ct_norm = counts_norm_no[band] if band in counts_norm_no.keys() else 0.0
        stats_no += f'{band}: {ct:,} ({format_two_sig_figs(ct_norm)})\n'

    # Draw arrow from previous and add decision
    if previous_element is not None:
        arrow = Arrow().down(d.unit/2).at(previous_element.S)
    else:
        arrow = Arrow().down(d.unit/2)
    d += arrow
    decision = Decision(w=5, h=3.9, E='Yes', S='No').label(filt_name)
    d += decision

    # Add the yes and no boxes
    arrow = Arrow().right(d.unit*3/4).at(decision.E)
    d += arrow
    yes_box = Box().label(stats_yes)
    d += yes_box
    arrow = Arrow().down(d.unit/2).at(decision.S)
    d += arrow
    no_box = Box().label(stats_no)
    d += no_box

    return d, yes_box, no_box


def add_decision(
        filt_name: str,
        tabs_yes: Dict[str, Union[Table, Sources]],
        tabs_no: Dict[str, Union[Table, Sources]],
        d: schemdraw.Drawing,
        init_counts: pd.Series,
        previous_element: Optional[Element] = None,
    ) -> Tuple[schemdraw.Drawing, Element, Element]:
    """Add decision to flow chart."""
    # Make the labels
    counts_yes = {band: len(tab) for band, tab in tabs_yes.items()}  # pd.Series(tab['Catalog_Flag']).value_counts()
    counts_no = {band: len(tab) for band, tab in tabs_no.items()}  # pd.Series(tab['Catalog_Flag']).value_counts()

    return add_decision_from_counts(
        filt_name,
        counts_yes,
        counts_no,
        d,
        init_counts,
        previous_element,
    )


def add_final_filt(filt_name: str, srcs: Sources, d: schemdraw.Drawing) -> schemdraw.Drawing:
    """Add final filter to the flowchart."""
    arrow = Arrow().down(d.unit/2).label(filt_name)
    d += arrow
    d += Box().label(f'Total: {len(srcs.data):,}')

    return d


def combine_stats(stats_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Function to combine (sum) the stats of several given flowchart dataframes."""
    combined_df = stats_dfs[0]

    # Ensure that the filters and branches of each df are the same
    for idx, df in enumerate(stats_dfs[1:], start=1):
        filter_equal = np.all(combined_df['filter'].to_numpy() == df['filter'].to_numpy())
        branch_equal = np.all(combined_df['branch'].to_numpy() == df['branch'].to_numpy())
        if not (filter_equal and branch_equal):
            # Find where the mismatch is
            filter_mismatch = combined_df['filter'].to_numpy() != df['filter'].to_numpy()
            branch_mismatch = combined_df['branch'].to_numpy() != df['branch'].to_numpy()
            mismatch_indices = np.where(~(filter_equal & branch_equal))[0] if (not filter_equal or not branch_equal) else []
            msg = f"DataFrame at index {idx} does not match the first DataFrame in filters and/or branches.\n"
            if not filter_equal:
                mismatched_filters = list(zip(combined_df['filter'][filter_mismatch], df['filter'][filter_mismatch]))
                msg += f"Filter mismatch at indices {np.where(filter_mismatch)[0].tolist()}: {mismatched_filters}\n"
            if not branch_equal:
                mismatched_branches = list(zip(combined_df['branch'][branch_mismatch], df['branch'][branch_mismatch]))
                msg += f"Branch mismatch at indices {np.where(branch_mismatch)[0].tolist()}: {mismatched_branches}\n"
            raise ValueError(msg)

    # Combine the stats
    for stats_df in stats_dfs[1:]:
        combined_df = combined_df.groupby(['filter', 'branch']).sum() + stats_df.groupby(['filter', 'branch']).sum()

    return combined_df


def create_filter_flowchart(stats_df: pd.DataFrame, decision: Optional[Dict[str, str]] = None) -> schemdraw.Drawing:
    """Create a flowchart given a dataframe with the statistics of a filtration.

    You can easily save a flowchart by doing create_filter_flowchart(...).save('path/to/chart.pdf')
    """
    # Set nan branch to empty string
    if 'branch' in stats_df.columns:
        stats_df['branch'] = stats_df['branch'].fillna('')

    # If not already, group by filters and branch
    if set(stats_df.index.names) != {'branch', 'filter'}:
        stats_df = stats_df.groupby(['filter', 'branch']).sum()

    # Sort the dataframe sequentially
    stats_df.sort_values(
        ['branch',
         'n_g_before', 'n_r_before', 'n_i_before', 'n_g_after', 'n_r_after', 'n_i_after',
         'n_gri_before', 'n_gr_before', 'n_gi_before', 'n_ri_before', 'n_gri_after', 'n_gr_after', 'n_gi_after', 'n_ri_after',
        ],
        ascending=[
            True,
            False, False, False, False, False, False,
            False, False, False, False, False, False, False, False,
        ],
        inplace=True,
    )

    # Try to automatically detect a decision structure in the graph
    if decision is None:
        all_branches = stats_df.index.get_level_values('branch')
        no_branch = all_branches[~all_branches.isin(['in_both', ''])]
        if len(no_branch) == 0:
            no_branch = None
        else:
            no_branch = no_branch[0]
        decision = {
            'text': r'Within 1-3$^{\prime\prime}$',
            'yes': 'in_both',
            'no': no_branch,
        }

    with schemdraw.Drawing(show=False) as d:

        # Initial stats
        init_counts = stats_df.loc[(slice(None), ''), ['n_g_before', 'n_r_before', 'n_i_before']].iloc[0]
        init_counts = init_counts.rename(index={'n_g_before': 'g', 'n_r_before': 'r', 'n_i_before': 'i'}).astype(int)

        # Add initial counts to the flowchart
        d, previous_element = add_filt_from_counts(
            'Initial Counts',
            init_counts.to_dict(),
            d,
            init_counts=init_counts,
            start=True,
        )

        # Iterate through the filters and add them to the flowchart
        branched = False
        prev_branch = ''
        unmerged_before_labs = ['n_g_before', 'n_r_before', 'n_i_before']
        merged_before_labs = ['n_gri_before', 'n_gr_before', 'n_gi_before', 'n_ri_before']
        for (filter_name, branch), row in stats_df.iterrows():

            # Check whether sources have been merged to >=2 bands yet
            # we'll do this by checking if all the g, r, i counts are 0 (at which point we'd expect to be logging in the
            # several-band columns)
            if np.sum(row[['n_g_after', 'n_r_after', 'n_i_after']]) == 0:
                before_labs = merged_before_labs
            else:
                before_labs = unmerged_before_labs
            after_labs = [lab.replace('before', 'after') for lab in before_labs]

            # Grab the counts from the row
            counts = row[after_labs].rename(
                index={lab: lab.replace('n_', '').replace('_after', '') for lab in after_labs}
            ).astype(int).to_dict()

            # Deal with decisions in the graph
            if branch in decision.values() and not branched:
                decision_text = decision['text']
                yes_counts = stats_df.loc[(slice(None), decision['yes']), before_labs].iloc[0].rename(
                    index={
                        lab: lab.replace('n_', '').replace('_after', '').replace('_before', '') for lab in before_labs
                    }
                ).astype(int).to_dict()
                no_counts = stats_df.loc[(slice(None), decision['no']), before_labs].iloc[0].rename(
                    index={
                        lab: lab.replace('n_', '').replace('_after', '').replace('_before', '') for lab in before_labs
                    }
                ).astype(int).to_dict()
                d, previous_element, other_branch_element = add_decision_from_counts(
                    decision_text,
                    yes_counts,
                    no_counts,
                    d,
                    init_counts=init_counts,
                    previous_element=previous_element,
                )
                branched = True

            # Handle whether we want to use the previous element, or start the other branch
            if (prev_branch != branch) and (prev_branch != ''):
                previous_element = other_branch_element
            prev_branch = branch

            # Add an element to the schemdraw figure
            filter_map = {
                'sep_extraction_filter': 'SEP extraction flags',
                'snr_filter': r'$\rm{SNR} > 5$',
                'shape_filter': 'Axis ratio',
                'psf_fit_filter': 'PSF fit',
                'only_big_dmag': r'$\Delta \rm{mag} > 5 \sigma$',
                'at_least_n_bands': r'$\Delta \rm{mag} > 5 \sigma$ in $>1$ band',
                'no_nearby_source_filter': r'$<5$ sources within $200^{\prime\prime}$',
                'proper_motion_filter': 'Proper motion ' + r'$\frac{\rm{value}}{\sigma} < 5$',
                'parallax_filter': 'Parallax ' + r'$\frac{\rm{value}}{\sigma} < 5$',
                'pstarr_mag_less_than': 'Pan-STARRS ' + r'$\rm{mag} < 22.5$',
                'dec_greater_than': r'$\rm{Dec} > -29.5$',
                'pstarr_quality_filter': 'Pan-STARRS quality flags',
            }

            if filter_name in filter_map:
                d, previous_element = add_filt_from_counts(
                    filter_map[filter_name],
                    counts,
                    d,
                    init_counts=init_counts,
                    previous_element=previous_element,
                )

        return d


def filter_field(field_name: str, overwrite: bool = False, store_pre_gaia: bool = False):
    """Filter a field, save flowchart plots, and save candidates."""
    # Load in the tables
    print('Loading tables...')
    tables = {}
    bands = ('g', 'r', 'i')
    for band in bands:
        try:
            tables[band] = load_ecsv(os.path.join(get_data_path(), f'catalog_results/field_results/{field_name}_{band}.ecsv'))
        except FileNotFoundError:
            print(f'Warning: Band {band} not available for field {field_name}...')

    # Set values <=0 to the upper limit for ZTF and add mag cols for Pan-STARRS
    for band in tables.keys():

        # Make sure mag cols are in the table
        for b in ('g', 'r', 'i'):
            if f'PSTARR_{b}PSFMag' not in tables[band].colnames:
                tables[band][f'PSTARR_{b}PSFMag'] = -999 * np.ones(len(tables[band]))

        tab: Table = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]   # 4 means flux was negative
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)

    # Get the stored 5-sigma delta mags
    with open(os.path.join(get_data_path(), '5sigma_delta_mags.pkl'), 'rb') as f:
        delta_mag_5sigma = pickle.load(f)
    bin_means, bin_stds = delta_mag_5sigma['means'], delta_mag_5sigma['stds']

    # Delete and recreate field filter directory
    filter_result_dirpath = os.path.join(get_data_path(), f'filter_results/{field_name}')
    if os.path.exists(filter_result_dirpath):
        if overwrite:
            print(f'Overwriting {filter_result_dirpath}/')
            shutil.rmtree(filter_result_dirpath)
        else:
            print(f'Field {field_name} already exists. Use `overwrite=True` to overwrite it.')
            return
    os.makedirs(filter_result_dirpath, exist_ok=True)


    ################################################################################
    ################################################################################
    ############### FILTERING FOR SOURCES DETECTED IN BOTH CATALOGS ###############
    ################################################################################
    ################################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '0_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[0]} graph...')
    tabs = {band: tab.copy()[tab['Catalog_Flag'] == 0] for band, tab in tables.items()}

    #---------------------------------------------------------------#
    # Initial stats
    init_counts = {band: len(tab) for band, tab in tabs.items()}

    # Drop all sources with bad SEP extraction flags
    tabs = filters.filter(tabs, 'sep_extraction_filter')

    # Drop all sources with snr < 5
    tabs, ztf_tabs_low_snr, pstarr_tabs_low_snr = filters.filter(tabs, 'snr_filter', snr_min=5, both_cat=True)

    # Axis ratio filter
    tabs = filters.filter(tabs, 'shape_filter')

    # Pan-STARRS not saturated
    tabs = filters.filter(tabs, 'pstarr_not_saturated')

    # Drop bad PSF fits
    tabs = filters.filter(tabs, 'psf_fit_filter')

    # Drop sources with dec < -29.5
    min_dec = -29.5
    tabs = filters.filter(tabs, 'dec_greater_than', min_dec=min_dec)

    # Delta mag > n sigma
    dmag_sigma = 5
    tabs, _, _ = filters.filter(tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds)

    # Converting to sources
    merged_coords = get_merged_tab_coords(tabs.values())
    sources = Sources(ras=merged_coords['ra'], decs=merged_coords['dec'], field_catalogs=tabs, verbose=0)

    # Check for big dmag in >1 bands
    sources = filters.filter(sources, 'at_least_n_bands', n=2)

    # Store pre-gaia filteration if requested
    if store_pre_gaia:
        sources.save(os.path.join(filter_result_dirpath, f'0_pre_gaia.ecsv'))

    # Check for proper motion
    sources = filters.filter(sources, 'proper_motion_filter')

    # Check for parallax
    sources = filters.filter(sources, 'parallax_filter')
    #---------------------------------------------------------------#

    # Save the sources and flowchart figure
    sources.save(os.path.join(filter_result_dirpath, f'0.ecsv'))
    d = create_filter_flowchart(filters.filter_stats)
    d.save(os.path.join(filter_result_dirpath, '0_flowchart.pdf'))

    # Save the filtered out tables
    filters.save_filtered_out(filter_result_dirpath, 0)

    ################################################################################
    ################################################################################
    ############### FILTERING FOR SOURCES DETECTED IN ZTF ONLY ###############
    ################################################################################
    ################################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '1_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[1]} graph...')
    in_ztf_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 1] for band, tab in tables.items()}
    in_pstarr_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 2] for band, tab in tables.items()}

    # Concat SNR < 5 non-detections
    in_ztf_tabs = {band: vstack([in_ztf_tabs[band], ztf_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}
    in_pstarr_tabs = {band: vstack([in_pstarr_tabs[band], pstarr_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}

    #---------------------------------------------------------------#
    # Initial stats
    init_counts = {band: len(tab) for band, tab in in_ztf_tabs.items()}

    # Drop all sources with bad SEP extraction flags
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'sep_extraction_filter')

    # Drop all sources with snr < 5
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'snr_filter', snr_min=5)

    # Axis ratio filter
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'shape_filter')

    # Drop bad PSF fits
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'psf_fit_filter')

    # Drop sources with dec < -29.5
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'dec_greater_than', min_dec=min_dec)

    # Double check ZTF sources that are a little more than 1 arcsec from PanSTARRS sources
    min_sep, max_sep = 1.0, 3.0
    in_both_tabs = {}
    for band, tab in in_ztf_tabs.items():
        in_both_tabs[band], in_ztf_tabs[band] = associate_in_btwn_distance(in_ztf_tabs[band], in_pstarr_tabs[band], min_sep=min_sep, max_sep=max_sep)
        print(f'Associated {len(in_both_tabs[band])} / {len(in_ztf_tabs[band]) + len(in_both_tabs[band])} more sources in {band} band between {min_sep} and {max_sep} from eachother.')

    ######################################################################
    ##### DEAL WITH THE SOURCES THAT ARE POTENTIALLY IN BOTH CATALOGS#####
    ######################################################################
    branch = 'in_both'        

    # Drop all sources with snr < 5
    in_both_tabs = filters.filter(in_both_tabs, 'snr_filter', snr_min=5, branch=branch)
    
    # Delta mag > n sigma
    in_both_tabs, _, _ = filters.filter(in_both_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds, branch=branch)

    # Converting to sources
    merged_coords_in_both = get_merged_tab_coords(in_both_tabs.values(), max_arcsec=3.0)
    sources_in_both = Sources(ras=merged_coords_in_both['ra'], decs=merged_coords_in_both['dec'], field_catalogs=in_both_tabs, verbose=0)

    # Check for big dmag in >1 bands
    sources_in_both = filters.filter(sources_in_both, 'at_least_n_bands', n=2, branch=branch)

    # Store pre-gaia filteration if requested
    if store_pre_gaia:
        sources_in_both.save(os.path.join(filter_result_dirpath, f'1_in_both_pre_gaia.ecsv'))

    # Check for proper motion
    sources_in_both = filters.filter(sources_in_both, 'proper_motion_filter', branch=branch)

    # Check for parallax
    sources_in_both = filters.filter(sources_in_both, 'parallax_filter', branch=branch)
    sources_in_both.save(os.path.join(filter_result_dirpath, f'1_wide_association.ecsv'))

    ######################################################################
    ##### DEAL WITH THE SOURCES THAT ARE NOT IN BOTH CATALOGS#####
    ######################################################################
    branch = 'in_ztf'

    # Delta mag > n sigma
    in_ztf_tabs, _, _ = filters.filter(in_ztf_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, upper_lim='PSTARR', bin_means=bin_means, bin_stds=bin_stds, branch=branch)

    # Converting to sources
    merged_coords_in_ztf = get_merged_tab_coords(in_ztf_tabs.values(), max_arcsec=3.0)
    sources_in_ztf = Sources(ras=merged_coords_in_ztf['ra'], decs=merged_coords_in_ztf['dec'], field_catalogs=in_ztf_tabs, verbose=0)

    # Check for big dmag in >1 bands
    sources_in_ztf = filters.filter(sources_in_ztf, 'at_least_n_bands', n=2, branch=branch)

    # Check for big dmag in >1 bands
    sources_in_ztf = filters.filter(sources_in_ztf, 'no_nearby_source_filter', n_nearby_max=5, branch=branch)

    # Store pre-gaia filteration if requested
    if store_pre_gaia:
        sources_in_ztf.save(os.path.join(filter_result_dirpath, f'1_pre_gaia.ecsv'))

    # Check for proper motion
    sources_in_ztf = filters.filter(sources_in_ztf, 'proper_motion_filter', branch=branch)

    # Check for parallax
    sources_in_ztf = filters.filter(sources_in_ztf, 'parallax_filter', branch=branch)
    #---------------------------------------------------------------#

    # Save the sources and flowchart figure
    sources_in_ztf.save(os.path.join(filter_result_dirpath, f'1.ecsv'))
    d = create_filter_flowchart(filters.filter_stats)
    d.save(os.path.join(filter_result_dirpath, '1_flowchart.pdf'))

    # Save the filtered out tables
    filters.save_filtered_out(filter_result_dirpath, 1)

    ################################################################################
    ################################################################################
    ############### FILTERING FOR SOURCES DETECTED IN PAN-STARRS ONLY ###############
    ################################################################################
    ################################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '2_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[2]} graph...')
    in_ztf_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 1] for band, tab in tables.items()}
    in_pstarr_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 2] for band, tab in tables.items()}

    # Concat SNR < 5 non-detections
    in_ztf_tabs = {band: vstack([in_ztf_tabs[band], ztf_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}
    in_pstarr_tabs = {band: vstack([in_pstarr_tabs[band], pstarr_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}

    #---------------------------------------------------------------#
    # Initial stats
    init_counts = {band: len(tab) for band, tab in in_pstarr_tabs.items()}

    # Make sure that the Pan-STARRS magnitude is less than max_mag
    max_mag = 22.5
    in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'pstarr_mag_less_than', max_mag=max_mag)

    # Pan-STARRS not saturated
    in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'pstarr_not_saturated')

    # Pan-STARRS quality filter
    in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'pstarr_quality_filter')

    # Drop all sources with snr < 5
    in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'snr_filter', snr_min=5)

    # Drop sources with dec < -29.5
    in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'dec_greater_than', min_dec=min_dec)

    # Double check ZTF sources that are a little more than 1 arcsec from PanSTARRS sources
    min_sep, max_sep = 1.0, 3.0
    in_both_tabs = {}
    for band, tab in in_pstarr_tabs.items():
        in_both_tabs[band], in_pstarr_tabs[band] = associate_in_btwn_distance(in_pstarr_tabs[band], in_ztf_tabs[band], min_sep=min_sep, max_sep=max_sep)
        print(f'Associated {len(in_both_tabs[band])} / {len(in_pstarr_tabs[band]) + len(in_both_tabs[band])} more sources in {band} band between {min_sep} and {max_sep} from eachother.')

    ######################################################################
    ##### DEAL WITH THE SOURCES THAT ARE POTENTIALLY IN BOTH CATALOGS#####
    branch = 'in_both'

    # Drop all sources with bad SEP extraction flags
    in_both_tabs = filters.filter(in_both_tabs, 'sep_extraction_filter', branch=branch)

    # Drop all sources with snr < 5
    in_both_tabs = filters.filter(in_both_tabs, 'snr_filter', snr_min=5, branch=branch)

    # Axis ratio filter
    in_both_tabs = filters.filter(in_both_tabs, 'shape_filter', branch=branch)

    # Drop bad PSF fits
    in_both_tabs = filters.filter(in_both_tabs, 'psf_fit_filter', branch=branch)

    # Delta mag > n sigma
    in_both_tabs, _, _ = filters.filter(in_both_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds, branch=branch)

    # Converting to sources
    merged_coords_in_both = get_merged_tab_coords(in_both_tabs.values(), max_arcsec=3.0)
    sources_in_both = Sources(ras=merged_coords_in_both['ra'], decs=merged_coords_in_both['dec'], field_catalogs=in_both_tabs, verbose=0)

    # Check for big dmag in >1 bands
    sources_in_both = filters.filter(sources_in_both, 'at_least_n_bands', n=2, branch=branch)

    # Store pre-gaia filtration if requested
    if store_pre_gaia:
        sources_in_both.save(os.path.join(filter_result_dirpath, f'2_in_both_pre_gaia.ecsv'))

    # Check for proper motion
    sources_in_both = filters.filter(sources_in_both, 'proper_motion_filter', branch=branch)

    # Check for parallax
    sources_in_both = filters.filter(sources_in_both, 'parallax_filter', branch=branch)

    sources_in_both.save(os.path.join(filter_result_dirpath, f'2_wide_association.ecsv'))

    ######################################################################
    ##### DEAL WITH THE SOURCES THAT ARE NOT IN BOTH CATALOGS#####
    branch = 'in_pstarr'

    # Delta mag > n sigma
    in_pstarr_tabs, _, _ = filters.filter(in_pstarr_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, upper_lim='ZTF', bin_means=bin_means, bin_stds=bin_stds, branch=branch)

    #### TESTING ####
    # SAVE SO WE CAN SEE THE DATA
    for band, tab in in_pstarr_tabs.items():
        tab.write(os.path.join(filter_result_dirpath, f'2_{band}_test.ecsv'), overwrite=True)

    # # Converting to sources
    # sources_in_pstarr: Dict[str, Sources] = {
    #     band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=in_pstarr_tabs, verbose=0) for band, tab in in_pstarr_tabs.items()
    # }

    # # Check for big dmag in >1 bands
    # sources_in_pstarr = filters.filter(sources_in_pstarr, 'at_least_n_bands', n=2, branch=branch)
    # add_filt(f'Big dmag in\n>=2 bands', sources_in_pstarr, d, init_counts=init_counts)

    # # Store pre-gaia filteration if requested
    # if store_pre_gaia:
    #     for band, srcs in sources_in_pstarr.items():
    #         srcs.save(os.path.join(filter_result_dirpath, f'1_{band}_pre_gaia.ecsv'))

    # # Check for proper motion
    # sources_in_pstarr = filters.filter(sources_in_pstarr, 'proper_motion_filter', branch=branch)
    # add_filt(f'Proper Motion', sources_in_pstarr, d, init_counts=init_counts)

    # # Check for parallax
    # sources_in_pstarr = filters.filter(sources_in_pstarr, 'parallax_filter', branch=branch)
    # add_filt(f'Parallax', sources_in_pstarr, d, init_counts=init_counts)
    #---------------------------------------------------------------#

    # Save the sources and flowchart figure
    # sources_in_pstarr.save(os.path.join(filter_result_dirpath, f'1.ecsv'))
    d = create_filter_flowchart(filters.filter_stats)
    d.save(os.path.join(filter_result_dirpath, '2_flowchart.pdf'))

    # Save the filtered out tables
    filters.save_filtered_out(filter_result_dirpath, 2)


def _filter_field_wrapper(field):
    print(f'Filtering field {field}...')
    filter_field(
        field,
        overwrite=True,
        store_pre_gaia=False,
    )

def filter_fields():
    """Filter fields!"""
    # fields = os.listdir('/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results')
    # fields = [f.split('_')[0] for f in fields]
    # fields = np.unique(fields)

    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     executor.map(_filter_field_wrapper, fields)

    _filter_field_wrapper('000293')


if __name__ == '__main__':
    filter_fields()
