import os
import sys
import shutil
import pickle
import numpy as np
import schemdraw
import pandas as pd

from typing import Dict, List, Iterable, Tuple, Union, Optional
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from schemdraw.flow import Start, Arrow, Box, Decision
from schemdraw.elements import Element
from decimal import Decimal, ROUND_HALF_UP

sys.path.append('/Users/adamboesky/Research/long_transients')

from Source_Analysis.Sources import Sources, MANDATORY_SOURCE_COLUMNS
from Extracting.utils import get_snr_from_mag, get_data_path

BANDS = ['g', 'r', 'i']
CATALOG_KEY = {0: 'ZTF and Pan-STARRS', 1: 'ZTF', 2: 'Pan-STARRS', 3: 'Out of Coverage'}
PSTARR_UPPER_LIM = {'g': 23.3, 'r': 23.2, 'i': 23.1}


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
        }
        self.reset_filter_stats()
        self.filter_stat_fname = filter_stat_fname

    def reset_filter_stats(self):
        """Reset the filter statistics."""
        self.filter_stats = pd.DataFrame(data={
            'filter': [],
            'ng_before': [],
            'ng_after': [],
            'nr_before': [],
            'nr_after': [],
            'ni_before': [],
            'ni_after': [],
        })

    def save_filter_stats(self):
        """Save the filtration statistics to a csv file."""
        self.filter_stats.to_csv(self.filter_stat_fname, index=False)

    def filter(
            self,
            tabs: Dict[str, Union[Table, Sources]],
            filt_name: str,
            *args,
            branch: str = '',
            **kwargs,
        ) -> Dict[str, Table]:
        """Filter the tables using the specified filter."""
        # Filtration stats
        before_counts = {band: len(tab) for band, tab in tabs.items()}

        # Filter
        tabs = self.filters[filt_name](tabs, *args, **kwargs)
        if isinstance(tabs, tuple):
            other_returns = list(tabs[1:])
            tabs = tabs[0]
        else:
            other_returns = None

        # Update filtration stats
        after_counts = {band: len(tab) for band, tab in tabs.items()}
        self.filter_stats = pd.concat(
            (
                self.filter_stats,
                pd.DataFrame(
                    data={
                        'filter': filt_name,
                        'ng_before': [before_counts.get('g', 0)],
                        'ng_after': [after_counts.get('g', 0)],
                        'nr_before': [before_counts.get('r', 0)],
                        'nr_after': [after_counts.get('r', 0)],
                        'ni_before': [before_counts.get('i', 0)],
                        'ni_after': [after_counts.get('i', 0)],
                        'branch': [branch],
                    }
                )
            )
        )

        # Save filtration stats if filename is given
        if self.filter_stat_fname is not None:
            self.save_filter_stats()

        # Drop tables if they are empty
        bands = list(tabs.keys())
        for band in bands:
            if len(tabs[band]) == 0:
                tabs.pop(band)

        if other_returns is None:
            return tabs
        return tuple([tabs] + other_returns)

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
        for band in tabs.keys():
            tab = tabs[band]
            mask = np.logical_or(tab[f'ZTF_{band}PSFFlags'] == 0, np.isnan(tab[f'ZTF_{band}PSFFlags']))
            mask = np.logical_or(mask, tab[f'ZTF_{band}PSFFlags'] == 4)
            tabs[band] = tab[mask]

        return tabs

    def sep_extraction_filter(self, tabs: Dict[str, Table], copy: bool = True, *args, **kwargs) -> Table:
        # flags here: https://sextractor.readthedocs.io/en/latest/Flagging.html
        for band in tabs.keys():
            tab = tabs[band]
            okay_flags = [1, 2, 3]
            mask = np.logical_or(tab[f'ZTF_sepExtractionFlag'] == 0, np.isnan(tab[f'ZTF_sepExtractionFlag']))
            for f in okay_flags:
                mask = np.logical_or(mask, _is_flag(tab[f'ZTF_sepExtractionFlag'], f))
            tabs[band] = tab[mask]

        return tabs


    def snr_filter(self, tabs: Table, snr_min: float = 5, copy: bool = True, *args, **kwargs) -> Table:
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

    def shape_filter(self, tabs: Table, copy: bool = True, *args, **kwargs) -> Table:
        for band in tabs.keys():
            tab = tabs[band]
            good_shape_mask = (tab['ZTF_a'] / tab['ZTF_b']) < 2                                                 # a/b < 2.0
            good_shape_mask &= np.logical_or((tab['ZTF_a'] / tab['ZTF_b']) < 1.35, tab['ZTF_tnpix'] < 200)      # a/b < 1.35 for big sources
            good_shape_mask |= np.isnan(tab['ZTF_a'])
            tabs[band] = tab[good_shape_mask]

        return tabs

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

            tabs[band] = tab[outside_mask]

        return tabs, bin_means, bin_stds

    def at_least_n_bands(self, sources: Dict[str, Sources], n: int = 2, *args, **kwargs) -> Dict[str, Sources]:
        for band in sources.keys():

            # Masks for whether source is in a band
            in_g = ~np.isnan(sources[band].data['ZTF_g_ra'])
            in_r = ~np.isnan(sources[band].data['ZTF_r_ra'])
            in_i = ~np.isnan(sources[band].data['ZTF_i_ra'])

            # Annotate the filter info
            band_mask = np.hstack((in_g.reshape(-1,1), in_r.reshape(-1, 1), in_i.reshape(-1, 1)))
            in_bands = [np.array(['g', 'r', 'i'])[m] for m in band_mask]
            for i, src_in_bands in enumerate(in_bands):
                sources[band][i].filter_info['in_bands'] = list(src_in_bands)

            # Mask for sources with n bands >= n
            enough_bands_mask = np.sum(band_mask, axis=1) >= n
            sources[band] = sources[band][enough_bands_mask]

        return sources


    def parallax_filter(self, sources: Dict[str, Sources], *args, **kwargs) -> Dict[str, Sources]:
        for band in sources.keys():
            mask = np.ones(len(sources[band]), dtype=bool)
            for i, src in enumerate(sources[band]):
                if len(src.GAIA_info) > 0:
                    mask[i] = (src.GAIA_info[0]['parallax_over_error'] < 5.0) or (src.GAIA_info['parallax_over_error'].mask[0])

            sources[band] = sources[band][mask]

        return sources


    def proper_motion_filter(self, sources: Dict[str, Sources], *args, **kwargs) -> Dict[str, Sources]:
        for band in sources.keys():
            mask = np.ones(len(sources[band]), dtype=bool)
            for i, src in enumerate(sources[band]):
                if len(src.GAIA_info) > 0:
                    mask[i] = (src.GAIA_info[0]['pm'] / (src.GAIA_info[0]['pmra_error'] + src.GAIA_info[0]['pmdec_error']) < 5.0) or \
                        (src.GAIA_info['pm'].mask[0] or \
                        src.GAIA_info['pmra_error'].mask[0] or \
                        src.GAIA_info['pmdec_error'].mask[0]
                        )

            sources[band] = sources[band][mask]

        return sources

    def no_nearby_source_filter(self, sources: Dict[str, Sources], n_nearby_max: int = 5, *args, **kwargs) -> Dict[str, Sources]:
        for band in sources.keys():
            all_coords = sources[band].coords
            seps = all_coords.separation(all_coords[:, None])
            nearby_counts = np.sum(seps.arcsecond < 200, axis=1)
            mask = nearby_counts <= n_nearby_max
            sources[band] = sources[band][mask]

        return sources

    def pstarr_mag_less_than(self, tabs: Dict[str, Table], max_mag: float, *args, **kwargs) -> Table:
        for band in tabs.keys():
            tabs[band] = tabs[band][tabs[band][f'PSTARR_{band}PSFMag'] < max_mag]

        return tabs


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
    # Make the label
    stats = ''
    counts_norm = {band: 0.0 if band not in counts.keys() or init_counts[band] == 0 else counts[band] / init_counts[band] for band in ('g', 'r', 'i')}
    for band in init_counts.keys():
        ct = counts[band] if band in counts.keys() else 0
        ct_norm = counts_norm[band] if band in counts_norm.keys() else 0.0
        stats += f'{band}: {ct:,} ({format_two_sig_figs(ct_norm)})\n'

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
    if not np.all([
        np.all(combined_df['filter'].to_numpy() == df['filter'].to_numpy()) and
        np.all(combined_df['branch'].to_numpy() == df['branch'].to_numpy())
        for df in stats_dfs[1:]
    ]):
        raise ValueError('All given dataframes must have the same filters and branches.')

    # Combine the stats
    for stats_df in stats_dfs[1:]:
        combined_df = combined_df.groupby(['filter', 'branch']).sum() + stats_df.groupby(['filter', 'branch']).sum()

    return combined_df


def create_filter_flowchart(stats_df: pd.DataFrame, decision: Optional[Dict[str, str]] = None) -> schemdraw.Drawing:
    """Create a flowchart given a dataframe with the statistics of a filtration.

    You can easily save a flowchart by doing create_filter_flowchart(...).save('path/to/chart.pdf')
    """
    # Sort the dataframe sequentially
    stats_df = stats_df.sort_values(
        ['branch', 'ng_before', 'nr_before', 'ni_before'], ascending=[True, False, False, False]
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
            'text': 'Within 1-3\'\'',
            'yes': 'in_both',
            'no': no_branch,
        }

    with schemdraw.Drawing(show=False) as d:

        # Initial stats
        init_counts = stats_df.loc[(slice(None), ''), ['ng_before', 'nr_before', 'ni_before']].iloc[0]
        init_counts = init_counts.rename(index={'ng_before': 'g', 'nr_before': 'r', 'ni_before': 'i'}).astype(int)

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
        for (filter_name, branch), row in stats_df.iterrows():

            # Grab the counts from the row
            counts = row[['ng_after', 'nr_after', 'ni_after']].rename(
                index={
                    'ng_after': 'g',
                    'nr_after': 'r',
                    'ni_after': 'i',
                }
            ).astype(int).to_dict()

            # Deal with decisions in the graph
            if branch in decision.values() and not branched:
                decision_text = decision['text']
                yes_counts = stats_df.loc[(slice(None), decision['yes']), ['ng_before', 'nr_before', 'ni_before']].iloc[0].rename(
                    index={
                        'ng_before': 'g',
                        'nr_before': 'r',
                        'ni_before': 'i',
                    }
                ).astype(int).to_dict()
                no_counts = stats_df.loc[(slice(None), decision['no']), ['ng_before', 'nr_before', 'ni_before']].iloc[0].rename(
                    index={
                        'ng_before': 'g',
                        'nr_before': 'r',
                        'ni_before': 'i',
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
                'sep_extraction_filter': 'SEP Extraction Flags',
                'snr_filter': 'SNR > 5',
                'shape_filter': 'Axis Ratio',
                'psf_fit_filter': 'PSF Fit',
                'only_big_dmag': 'Delta Mag > 5 sigma',
                'at_least_n_bands': 'Big dmag in\n>=2 bands',
                'no_nearby_source_filter': 'Fewer than 5\nnearby sources',
                'proper_motion_filter': 'Proper Motion',
                'parallax_filter': 'Parallax',
                'pstarr_mag_less_than': 'Pan-STARRS Magnitude\nless than 22.5',
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
            tables[band] = ascii.read(os.path.join(get_data_path(), f'catalog_results/field_results/{field_name}_{band}.ecsv'), format='ecsv')
        except FileNotFoundError:
            print(f'Warning: Band {band} not available for filed {field_name}...')

    # Set values <=0 to the upper limit for ZTF
    for band in tables.keys():
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
    os.mkdir(filter_result_dirpath)


    #################################################################
    ############ Filtering for sources detected in both! ############
    #################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '0_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[0]} graph...')
    tabs = {band: tab.copy()[tab['Catalog_Flag'] == 0] for band, tab in tables.items()}

    with schemdraw.Drawing(show=False) as d:

        # Initial stats
        init_counts = {band: len(tab) for band, tab in tabs.items()}
        d, d0 = add_filt('Initial Counts', tabs, d, init_counts=init_counts, start=True)

        # Drop all sources with bad SEP extraction flags
        tabs = filters.filter(tabs, 'sep_extraction_filter')
        d, d1 = add_filt('SEP Extraction Flags', tabs, d, init_counts=init_counts, previous_element=d0)

        # Drop all sources with snr < 5
        tabs = filters.filter(tabs, 'snr_filter', snr_min=5)
        add_filt('SNR > 5', tabs, d, init_counts=init_counts, previous_element=d1)

        # Axis ratio filter
        tabs = filters.filter(tabs, 'shape_filter')
        add_filt('Axis Ratio', tabs, d, init_counts=init_counts)

        # Drop bad PSF fits
        tabs = filters.filter(tabs, 'psf_fit_filter')
        add_filt('PSF Fit', tabs, d, init_counts=init_counts)

        # Delta mag > n sigma
        dmag_sigma = 5
        tabs, _, _ = filters.filter(tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', tabs, d, init_counts=init_counts)

        # Converting to sources
        sources: Dict[str, Sources] = {
            band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=tabs, verbose=0) for band, tab in tabs.items()
        }

        # Check for big dmag in >1 bands
        sources = filters.filter(sources, 'at_least_n_bands', n=2)
        add_filt(f'Big dmag in >=2 bands', sources, d, init_counts=init_counts)

        # Store pre-gaia filteration if requested
        if store_pre_gaia:
            for band, srcs in sources.items():
                srcs.save(os.path.join(filter_result_dirpath, f'0_{band}_pre_gaia.ecsv'))

        # Check for proper motion
        sources = filters.filter(sources, 'proper_motion_filter')
        add_filt(f'Proper Motion', sources, d, init_counts=init_counts)

        # Check for parallax
        sources = filters.filter(sources, 'parallax_filter')
        add_filt(f'Parallax', sources, d, init_counts=init_counts)

        # Save the image
        d.save(os.path.join(filter_result_dirpath, '0_flowchart.pdf'))
        for band, srcs in sources.items():
            srcs.save(os.path.join(filter_result_dirpath, f'0_{band}.ecsv'))


    #################################################################
    ########## Filtering for sources detected in just ZTF! ##########
    #################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '1_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[1]} graph...')
    in_ztf_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 1] for band, tab in tables.items()}
    in_pstarr_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 2] for band, tab in tables.items()}

    with schemdraw.Drawing(show=False) as d:

        # Initial stats
        init_counts = {band: len(tab) for band, tab in in_ztf_tabs.items()}
        add_filt('Initial Counts', in_ztf_tabs, d, init_counts=init_counts, start=True)

        # Drop all sources with bad SEP extraction flags
        in_ztf_tabs = filters.filter(in_ztf_tabs, 'sep_extraction_filter')
        add_filt('SEP Extraction Flags', in_ztf_tabs, d, init_counts=init_counts)

        # Drop all sources with snr < 5
        in_ztf_tabs = filters.filter(in_ztf_tabs, 'snr_filter', snr_min=5)
        add_filt('SNR > 5', in_ztf_tabs, d, init_counts=init_counts)

        # Axis ratio filter
        in_ztf_tabs = filters.filter(in_ztf_tabs, 'shape_filter')
        add_filt('Axis Ratio', in_ztf_tabs, d, init_counts=init_counts)

        # Drop bad PSF fits
        in_ztf_tabs = filters.filter(in_ztf_tabs, 'psf_fit_filter')
        add_filt('PSF Fit', in_ztf_tabs, d, init_counts=init_counts)

        # Double check ZTF sources that are a little more than 1 arcsec from PanSTARRS sources
        min_sep, max_sep = 1.0, 3.0
        in_both_tabs = {}
        for band, tab in in_ztf_tabs.items():
            in_both_tabs[band], in_ztf_tabs[band] = associate_in_btwn_distance(in_ztf_tabs[band], in_pstarr_tabs[band], min_sep=min_sep, max_sep=max_sep)  # TODO: Deal with tables that aren't in the 1-3 range
            print(f'Associated {len(in_both_tabs[band])} / {len(in_ztf_tabs[band]) + len(in_both_tabs[band])} more sources in {band} band between {min_sep} and {max_sep} from eachother.')

        # Add decision for whether sources are within 1-3 arcsec
        d, in_both_element, just_ztf_element = add_decision('Within 1-3\'\'', in_both_tabs, in_ztf_tabs, d, init_counts=init_counts)

        ######################################################################
        ##### DEAL WITH THE SOURCES THAT ARE POTENTIALLY IN BOTH CATALOGS#####
        branch = 'in_both'        

        # Drop all sources with snr < 5
        in_both_tabs = filters.filter(in_both_tabs, 'snr_filter', snr_min=5, branch=branch)
        add_filt('SNR > 5', in_both_tabs, d, init_counts=init_counts, previous_element=in_both_element)
        
        # Delta mag > n sigma
        in_both_tabs, _, _ = filters.filter(in_both_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds, branch=branch)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', in_both_tabs, d, init_counts=init_counts)

        # Converting to sources
        sources_in_both: Dict[str, Sources] = {
            band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=in_both_tabs, verbose=0) for band, tab in in_both_tabs.items()
        }

        # Check for big dmag in >1 bands
        sources_in_both = filters.filter(sources_in_both, 'at_least_n_bands', n=2, branch=branch)
        add_filt(f'Big dmag in\n>=2 bands', sources_in_both, d, init_counts=init_counts)

        # Store pre-gaia filteration if requested
        if store_pre_gaia:
            for band, srcs in sources_in_both.items():
                srcs.save(os.path.join(filter_result_dirpath, f'1_in_both_{band}_pre_gaia.ecsv'))

        # Check for proper motion
        sources_in_both = filters.filter(sources_in_both, 'proper_motion_filter', branch=branch)
        add_filt(f'Proper Motion', sources_in_both, d, init_counts=init_counts)

        # Check for parallax
        sources_in_both = filters.filter(sources_in_both, 'parallax_filter', branch=branch)
        add_filt(f'Parallax', sources_in_both, d, init_counts=init_counts)

        for band, srcs in sources_in_both.items():
            srcs.save(os.path.join(filter_result_dirpath, f'1_wide_association_{band}.ecsv'))

        ######################################################################
        ##### DEAL WITH THE SOURCES THAT ARE NOT IN BOTH CATALOGS#####
        branch = 'in_ztf'

        # Delta mag > n sigma
        in_ztf_tabs, _, _ = filters.filter(in_ztf_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, upper_lim='PSTARR', bin_means=bin_means, bin_stds=bin_stds, branch=branch)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', in_ztf_tabs, d, init_counts=init_counts, previous_element=just_ztf_element)

        # Converting to sources
        sources_in_ztf: Dict[str, Sources] = {
            band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=in_ztf_tabs, verbose=0) for band, tab in in_ztf_tabs.items()
        }

        # Check for big dmag in >1 bands
        sources_in_ztf = filters.filter(sources_in_ztf, 'at_least_n_bands', n=2, branch=branch)
        add_filt(f'Big dmag in\n>=2 bands', sources_in_ztf, d, init_counts=init_counts)

        # Check for big dmag in >1 bands
        sources_in_ztf = filters.filter(sources_in_ztf, 'no_nearby_source_filter', n_nearby_max=5, branch=branch)
        add_filt(f'Fewer than 5\nnearby sources', sources_in_ztf, d, init_counts=init_counts)

        # Store pre-gaia filteration if requested
        if store_pre_gaia:
            for band, srcs in sources_in_ztf.items():
                srcs.save(os.path.join(filter_result_dirpath, f'1_{band}_pre_gaia.ecsv'))

        # Check for proper motion
        sources_in_ztf = filters.filter(sources_in_ztf, 'proper_motion_filter', branch=branch)
        add_filt(f'Proper Motion', sources_in_ztf, d, init_counts=init_counts)

        # Check for parallax
        sources_in_ztf = filters.filter(sources_in_ztf, 'parallax_filter', branch=branch)
        add_filt(f'Parallax', sources_in_ztf, d, init_counts=init_counts)

        # Save the image
        d.save(os.path.join(filter_result_dirpath, '1_flowchart.pdf'))
        for band, srcs in sources_in_ztf.items():
            srcs.save(os.path.join(filter_result_dirpath, f'1_{band}.ecsv'))


    #######################################################################
    ########## Filtering for sources detected in just PanSTARRS! ##########
    #######################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '2_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[2]} graph...')
    in_ztf_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 1] for band, tab in tables.items()}
    in_pstarr_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 2] for band, tab in tables.items()}

    with schemdraw.Drawing(show=False) as d:

        # Initial stats
        init_counts = {band: len(tab) for band, tab in in_pstarr_tabs.items()}
        add_filt('Initial Counts', in_pstarr_tabs, d, init_counts=init_counts, start=True)

        # Make sure that the Pan-STARRS magnitude is less than max_mag
        max_mag = 22.5
        in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'pstarr_mag_less_than', max_mag=max_mag)
        add_filt(f'Pan-STARRS Magnitude\nless than {max_mag}', in_pstarr_tabs, d, init_counts=init_counts)

        # Drop all sources with snr < 5
        in_pstarr_tabs = filters.filter(in_pstarr_tabs, 'snr_filter', snr_min=5)
        add_filt('SNR > 5', in_pstarr_tabs, d, init_counts=init_counts)

        # Double check ZTF sources that are a little more than 1 arcsec from PanSTARRS sources
        min_sep, max_sep = 1.0, 3.0
        in_both_tabs = {}
        for band, tab in in_pstarr_tabs.items():
            in_both_tabs[band], in_pstarr_tabs[band] = associate_in_btwn_distance(in_pstarr_tabs[band], in_ztf_tabs[band], min_sep=min_sep, max_sep=max_sep)  # TODO: Deal with tables that aren't in the 1-3 range
            print(f'Associated {len(in_both_tabs[band])} / {len(in_pstarr_tabs[band]) + len(in_both_tabs[band])} more sources in {band} band between {min_sep} and {max_sep} from eachother.')

        # Add decision for whether sources are within 1-3 arcsec
        d, in_both_element, just_pstarr_element = add_decision('Within 1-3\'\'', in_both_tabs, in_pstarr_tabs, d, init_counts=init_counts)

        ######################################################################
        ##### DEAL WITH THE SOURCES THAT ARE POTENTIALLY IN BOTH CATALOGS#####
        branch = 'in_both'

        # Drop all sources with bad SEP extraction flags
        in_both_tabs = filters.filter(in_both_tabs, 'sep_extraction_filter', branch=branch)
        add_filt('SEP Extraction Flags', in_both_tabs, d, init_counts=init_counts, previous_element=in_both_element)

        # Drop all sources with snr < 5
        in_both_tabs = filters.filter(in_both_tabs, 'snr_filter', snr_min=5, branch=branch)
        add_filt('SNR > 5', in_both_tabs, d, init_counts=init_counts)

        # Axis ratio filter
        in_both_tabs = filters.filter(in_both_tabs, 'shape_filter', branch=branch)
        add_filt('Axis Ratio', in_both_tabs, d, init_counts=init_counts)

        # Drop bad PSF fits
        in_both_tabs = filters.filter(in_both_tabs, 'psf_fit_filter', branch=branch)
        add_filt('PSF Fit', in_both_tabs, d, init_counts=init_counts)

        # Delta mag > n sigma
        in_both_tabs, _, _ = filters.filter(in_both_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, bin_means=bin_means, bin_stds=bin_stds, branch=branch)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', in_both_tabs, d, init_counts=init_counts)

        # Converting to sources
        sources_in_both: Dict[str, Sources] = {
            band: Sources(ras=tab['ra'], decs=tab['dec'], field_catalogs=in_both_tabs, verbose=0) for band, tab in in_both_tabs.items()
        }

        # Check for big dmag in >1 bands
        sources_in_both = filters.filter(sources_in_both, 'at_least_n_bands', n=2, branch=branch)
        add_filt(f'Big dmag in\n>=2 bands', sources_in_both, d, init_counts=init_counts)

        # Store pre-gaia filtration if requested
        if store_pre_gaia:
            for band, srcs in sources_in_both.items():
                srcs.save(os.path.join(filter_result_dirpath, f'2_in_both_{band}_pre_gaia.ecsv'))

        # Check for proper motion
        sources_in_both = filters.filter(sources_in_both, 'proper_motion_filter', branch=branch)
        add_filt(f'Proper Motion', sources_in_both, d, init_counts=init_counts)

        # Check for parallax
        sources_in_both = filters.filter(sources_in_both, 'parallax_filter', branch=branch)
        add_filt(f'Parallax', sources_in_both, d, init_counts=init_counts)

        for band, srcs in sources_in_both.items():
            srcs.save(os.path.join(filter_result_dirpath, f'2_wide_association_{band}.ecsv'))

        ######################################################################
        ##### DEAL WITH THE SOURCES THAT ARE NOT IN BOTH CATALOGS#####
        print('ADAM HERE!!!!!!!!!')
        branch = 'in_pstarr'

        # Delta mag > n sigma
        in_pstarr_tabs, _, _ = filters.filter(in_pstarr_tabs, 'only_big_dmag', mag_thresh=dmag_sigma, upper_lim='ZTF', bin_means=bin_means, bin_stds=bin_stds, branch=branch)
        add_filt(f'Delta Mag > {dmag_sigma} sigma', in_pstarr_tabs, d, init_counts=init_counts, previous_element=just_pstarr_element)

        #### TESTING ####
        # SAVE SO WE CAN SEE THE DATA
        for band, tab in in_pstarr_tabs.items():
            tab.write(os.path.join('/Users/adamboesky/Research/long_transients/Data/filter_testing', f'testing_stuff_{band}.ecsv'), overwrite=True)

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

        # Save the image
        d.save(os.path.join(filter_result_dirpath, '2_filtering_flowchart.pdf'))
        # for band, srcs in sources_in_pstarr.items():
        #     srcs.save(os.path.join(filter_result_dirpath, f'2_{band}.ecsv'))


def filter_fields():
    """Filter fields!"""
    for field in [
        '000292',
        '000581',
        '000445',
        '000578',
        '000276',
        '000446',
        '000582',
        '000294',
        '000444',
        '000293',
        '000246',
        '000245',
        '000228',
        '000447',
        '000579',
        '000295',
        '000580',
        '000202',
    ]:
        print(f'Filtering field {field}...')
        filter_field(
            field,
            overwrite=True,
            store_pre_gaia=False,
        )


if __name__ == '__main__':
    filter_fields()
