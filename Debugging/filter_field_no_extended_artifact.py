"""Re-run filter_fields.py's filter_field() for ONE field with the single
`extended_source_artifact_filter` call removed from the in_ztf (cat 1) chain.

Purpose: diff this script's `1.hdf5` / `1_wide_association.ecsv` output against
the real pipeline's existing output (`Data/filter_results_kde_sep_flag/<field>/1.hdf5`)
for the same field. Sources that appear in THIS output but not in the real one
are exactly the sources extended_source_artifact_filter is uniquely responsible
for removing (i.e. they survive every other filter in the chain unchanged).

This is a faithful copy of filter_field() (see Source_Analysis/filter_fields.py,
lines 1490-1875) with only that one filter call removed (marked below) and the
output directory changed so it never overwrites the real pipeline's results.
Everything else — filter order, helper functions, Gaia/KDE caching — is
imported from the original module so behavior stays identical.

Usage (run on the cluster, where the raw per-field catalogs and Gaia
credentials are available):
    python Debugging/filter_field_no_extended_artifact.py <field_name>
    python Debugging/filter_field_no_extended_artifact.py            # defaults to 000326
"""
import os
import sys
import shutil

import numpy as np
from astropy.table import Table, vstack

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis.Sources import Sources
from Source_Analysis.filter_fields import (
    Filters,
    CATALOG_KEY,
    EXTRACTED_CATALOG_DIR,
    remove_mask,
    query_gaia_for_field,
    load_or_build_kde_envelopes,
    get_merged_tab_coords,
    associate_in_btwn_distance,
    create_filter_flowchart,
)
from Extracting.utils import get_data_path

# Distinct output dir so this never clobbers the real pipeline's
# Data/filter_results_kde_sep_flag/<field>/ output.
NO_ARTIFACT_FILTER_RESULT_DIR = 'filter_results_kde_sep_flag_no_extended_artifact'


def filter_field_no_extended_artifact(field_name: str, overwrite: bool = True, store_pre_gaia: bool = False):
    """Copy of Source_Analysis.filter_fields.filter_field(), minus the
    extended_source_artifact_filter call in the in_ztf (cat 1) chain.
    """
    filter_result_dirpath = os.path.join(get_data_path(), f'{NO_ARTIFACT_FILTER_RESULT_DIR}/{field_name}')
    if os.path.exists(filter_result_dirpath) and not overwrite:
        print(f'Field {field_name} already exists. Use `overwrite=True` to overwrite it.')
        return

    # Load in the tables
    print('Loading tables...')
    tables = {}
    bands = ('g', 'r', 'i')
    for band in bands:
        try:
            print(f'{EXTRACTED_CATALOG_DIR}/{field_name}_{band}.hdf5')
            tables[band] = Table.read(os.path.join(get_data_path(), f'{EXTRACTED_CATALOG_DIR}/{field_name}_{band}.hdf5'), path='data')
        except FileNotFoundError:
            print(f'Warning: Band {band} not available for field {field_name}...')
    print('Finished loading tables...')

    # Set values <=0 to the upper limit for ZTF and add mag cols for Pan-STARRS
    for band in tables.keys():
        for b in ('g', 'r', 'i'):
            if f'PSTARR_{b}PSFMag' not in tables[band].colnames:
                tables[band][f'PSTARR_{b}PSFMag'] = -999 * np.ones(len(tables[band]))

        tab: Table = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]  # 4 means flux was negative
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)

    # Bulk Gaia query — one network round-trip for the whole field, cached to disk.
    all_ras = np.concatenate([np.asarray(t['ra']) for t in tables.values()])
    all_decs = np.concatenate([np.asarray(t['dec']) for t in tables.values()])
    field_gaia_table = query_gaia_for_field(field_name, all_ras, all_decs)
    print(f'Gaia query returned {len(field_gaia_table)} sources for field {field_name}.')

    # Delete and recreate field filter directory
    if os.path.exists(filter_result_dirpath):
        print(f'Overwriting {filter_result_dirpath}/')
        shutil.rmtree(filter_result_dirpath)
    os.makedirs(filter_result_dirpath, exist_ok=True)

    ################################################################################
    ############### FILTERING FOR SOURCES DETECTED IN BOTH CATALOGS ###############
    ################################################################################
    # Identical to the original — extended_source_artifact_filter is never
    # called in this branch, so nothing changes here. Kept because cat-1
    # (below) depends on ztf_tabs_low_snr / pstarr_tabs_low_snr / envelopes /
    # all_quality_source_tabs / field_gaia_table computed in this block.
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '0_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[0]} graph...')
    min_dec = -29.5
    tabs = {band: tab.copy()[tab['Catalog_Flag'] == 0] for band, tab in tables.items()}

    _qfilters = Filters()
    all_quality_source_tabs = {band: tab.copy() for band, tab in tables.items()}
    all_quality_source_tabs = _qfilters.filter(all_quality_source_tabs, 'sep_extraction_filter')
    all_quality_source_tabs, all_q_ztf_tabs_low_snr, all_q_pstarr_tabs_low_snr = _qfilters.filter(all_quality_source_tabs, 'snr_filter', snr_min=5, both_cat=True)

    for band in tabs.keys():
        all_q_ztf_tabs_low_snr[band]['Catalog_Flag'] = 1
        all_q_pstarr_tabs_low_snr[band]['Catalog_Flag'] = 2
        if band not in all_quality_source_tabs.keys():
            all_quality_source_tabs[band] = tabs[band][:0].copy()
    all_quality_source_tabs = {band: vstack([all_quality_source_tabs[band], all_q_ztf_tabs_low_snr[band], all_q_pstarr_tabs_low_snr[band]]) for band in tabs.keys()}
    all_quality_source_tabs = _qfilters.filter(all_quality_source_tabs, 'shape_filter')
    all_quality_source_tabs = _qfilters.filter(all_quality_source_tabs, 'pstarr_not_saturated')
    all_quality_source_tabs = _qfilters.filter(all_quality_source_tabs, 'psf_fit_filter')
    all_quality_source_tabs = _qfilters.filter(all_quality_source_tabs, 'dec_greater_than', min_dec=min_dec)

    in_both_quality_tabs = {band: tab[tab['Catalog_Flag'] == 0] for band, tab in all_quality_source_tabs.items()}
    envelopes = load_or_build_kde_envelopes(field_name, in_both_quality_tabs)
    sigma_boundary = 3.0

    tabs = filters.filter(tabs, 'sep_extraction_filter')
    tabs = filters.filter(tabs, 'shape_filter')
    tabs = filters.filter(tabs, 'pstarr_not_saturated')
    tabs = filters.filter(tabs, 'psf_fit_filter')
    tabs = filters.filter(tabs, 'dec_greater_than', min_dec=min_dec)
    tabs, ztf_tabs_low_snr, pstarr_tabs_low_snr = filters.filter(tabs, 'snr_filter', snr_min=5, both_cat=True)
    tabs, _ = filters.filter(tabs, 'only_big_dmag', sigma_boundary=sigma_boundary, envelopes=envelopes)

    merged_coords = get_merged_tab_coords(tabs)
    sources = Sources(ras=merged_coords['ra'], decs=merged_coords['dec'], field_catalogs=all_quality_source_tabs, verbose=0)
    sources = filters.filter(sources, 'at_least_n_big_dmag_bands', catalog='in_both', n=2, sigma_boundary=sigma_boundary, envelopes=envelopes)

    if store_pre_gaia:
        sources.save(os.path.join(filter_result_dirpath, f'0_pre_gaia.ecsv'))

    sources = filters.filter(sources, 'proper_motion_filter', gaia_table=field_gaia_table)
    sources = filters.filter(sources, 'parallax_filter', gaia_table=field_gaia_table)

    sources.save(os.path.join(filter_result_dirpath, f'0.hdf5'))
    d = create_filter_flowchart(filters.filter_stats)
    d.save(os.path.join(filter_result_dirpath, '0_flowchart.pdf'))
    filters.save_filtered_out(filter_result_dirpath, 0)

    ################################################################################
    ############### FILTERING FOR SOURCES DETECTED IN ZTF ONLY ###############
    ################################################################################
    filters = Filters(filter_stat_fname=os.path.join(filter_result_dirpath, '1_filter_stats.csv'))
    print(f'Building flowchart for {CATALOG_KEY[1]} graph...')
    in_ztf_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 1] for band, tab in tables.items()}
    in_pstarr_tabs = {band: tab.copy()[tab['Catalog_Flag'] == 2] for band, tab in tables.items()}

    in_ztf_tabs = {band: vstack([in_ztf_tabs[band], ztf_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}
    in_pstarr_tabs = {band: vstack([in_pstarr_tabs[band], pstarr_tabs_low_snr[band]]) for band in in_ztf_tabs.keys()}

    for band in in_ztf_tabs.keys():
        in_ztf_tabs[band]['Catalog_Flag'] = 1
        in_pstarr_tabs[band]['Catalog_Flag'] = 2

    # Drop all sources with bad SEP extraction flags
    in_ztf_tabs = filters.filter(in_ztf_tabs, 'sep_extraction_filter')

    # <<< REMOVED vs. the original: >>>
    # in_ztf_tabs = filters.filter(in_ztf_tabs, 'extended_source_artifact_filter')

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

    in_both_tabs = filters.filter(in_both_tabs, 'snr_filter', snr_min=5, branch=branch)

    for band in in_both_tabs.keys():
        in_both_tabs[band]['Catalog_Flag'] = 0
    in_both_tabs, _ = filters.filter(in_both_tabs, 'only_big_dmag', sigma_boundary=sigma_boundary, envelopes=envelopes, branch=branch)

    merged_coords_in_both = get_merged_tab_coords(in_both_tabs, max_arcsec=3.0)
    sources_in_both = Sources(ras=merged_coords_in_both['ra'], decs=merged_coords_in_both['dec'], field_catalogs=all_quality_source_tabs, verbose=0)

    sources_in_both = filters.filter(sources_in_both, 'at_least_n_big_dmag_bands', catalog='in_both', n=2, sigma_boundary=sigma_boundary, envelopes=envelopes, branch=branch)

    if store_pre_gaia:
        sources_in_both.save(os.path.join(filter_result_dirpath, f'1_in_both_pre_gaia.ecsv'))

    sources_in_both = filters.filter(sources_in_both, 'proper_motion_filter', branch=branch, gaia_table=field_gaia_table)
    sources_in_both = filters.filter(sources_in_both, 'parallax_filter', branch=branch, gaia_table=field_gaia_table)
    sources_in_both.save(os.path.join(filter_result_dirpath, f'1_wide_association.ecsv'))

    ######################################################################
    ##### DEAL WITH THE SOURCES THAT ARE NOT IN BOTH CATALOGS#####
    ######################################################################
    branch = 'in_ztf'

    in_ztf_tabs, _ = filters.filter(in_ztf_tabs, 'only_big_dmag', sigma_boundary=sigma_boundary, envelopes=envelopes, branch=branch)

    merged_coords_in_ztf = get_merged_tab_coords(in_ztf_tabs, max_arcsec=3.0)
    sources_in_ztf = Sources(ras=merged_coords_in_ztf['ra'], decs=merged_coords_in_ztf['dec'], field_catalogs=all_quality_source_tabs, verbose=0)

    sources_in_ztf = filters.filter(sources_in_ztf, 'at_least_n_big_dmag_bands', catalog='in_ztf', n=2, sigma_boundary=sigma_boundary, envelopes=envelopes, branch=branch)
    sources_in_ztf = filters.filter(sources_in_ztf, 'no_nearby_source_filter', n_nearby_max=5, branch=branch)

    if store_pre_gaia:
        sources_in_ztf.save(os.path.join(filter_result_dirpath, f'1_pre_gaia.ecsv'))

    sources_in_ztf = filters.filter(sources_in_ztf, 'proper_motion_filter', branch=branch, gaia_table=field_gaia_table)
    sources_in_ztf = filters.filter(sources_in_ztf, 'parallax_filter', branch=branch, gaia_table=field_gaia_table)

    sources_in_ztf.save(os.path.join(filter_result_dirpath, f'1.hdf5'))
    d = create_filter_flowchart(filters.filter_stats)
    d.save(os.path.join(filter_result_dirpath, '1_flowchart.pdf'))
    filters.save_filtered_out(filter_result_dirpath, 1)

    # Cat 2 (in_pstarr) never calls extended_source_artifact_filter, so it's
    # identical to the original either way — skipped here since only cat 1's
    # output is needed for the diff, and this field's raw data/Gaia access is
    # the expensive part to run.

    print(f'Done. Wrote {filter_result_dirpath}/1.hdf5 (compare against '
          f'{os.path.join(get_data_path(), "filter_results_kde_sep_flag", field_name, "1.hdf5")})')


if __name__ == '__main__':
    field = sys.argv[1] if len(sys.argv) > 1 else '000326'
    filter_field_no_extended_artifact(field, overwrite=True)
