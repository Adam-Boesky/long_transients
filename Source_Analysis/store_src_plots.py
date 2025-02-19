"""Script for saving plots of all the extracted sources."""
import os
import sys
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import get_data_path
from Sources import Source, Sources
from multiprocessing import Pool

# Plot formatting
plt.rc('text', usetex=True)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
mpl.rcParams['font.size'] = 12  # Adjust the font size as needed
mpl.rcParams['axes.formatter.use_mathtext'] = True

OVERWRITE = False


def save_src_plot(src: Source, out_fname: str, overwrite: bool, n_attempts: int = 3):
    for i_attempt in range(n_attempts):
        try:
            if os.path.exists(out_fname) and not overwrite:
                print(f'Skipping source... Already plotted and saved at {out_fname}')
            else:
                print(f'Plotting source {out_fname.split('/')[-1].split('.')[0]} at ({src.ra}, {src.dec})!')
                src.plot_everything()
                plt.savefig(out_fname, bbox_inches='tight')
        except:
            if i_attempt >= n_attempts - 1:
                raise
            else:
                print(f'Attempt {i_attempt + 1} / {n_attempts} to save {out_fname.split("/")[-1].split(".")[0]} failed.'
                      'Trying again...')


def store_source_plots():
    """Store the plots for each candidate resulting from filtering."""
    if not os.path.exists('/Volumes/T7/long_transients/candidates'):
        os.mkdir('/Volumes/T7/long_transients/candidates')

    # Load the stored files
    combined_g_tabs = {}
    combined_r_tabs = {}
    combined_i_tabs = {}
    combined_coords = {}
    combined_g_wide_tabs = {}
    combined_r_wide_tabs = {}
    combined_i_wide_tabs = {}
    combined_wide_coords = {}
    for cat_num in range(2):  # TODO make this range(3) and add pstarr only
        combined_g_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_g.ecsv'), format='ascii.ecsv')
        combined_r_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_r.ecsv'), format='ascii.ecsv')
        combined_i_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_i.ecsv'), format='ascii.ecsv')
        combined_coords[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_coords.ecsv'), format='ascii.ecsv')

        combined_g_wide_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_wide_g.ecsv'), format='ascii.ecsv')
        combined_r_wide_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_wide_r.ecsv'), format='ascii.ecsv')
        combined_i_wide_tabs[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_wide_i.ecsv'), format='ascii.ecsv')
        combined_wide_coords[cat_num] = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_wide_coords.ecsv'), format='ascii.ecsv')

    ### IN BOTH ###
    # plot_dir = os.path.join(get_data_path(), 'filter_results/candidates/in_both')
    plot_dir = '/Volumes/T7/long_transients/candidates/in_both'
    if os.path.exists(plot_dir) and OVERWRITE:
        shutil.rmtree(plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    srcs = Sources(
        ras=combined_coords[0]['ra'],
        decs=combined_coords[0]['dec'],
        field_catalogs={
            'g': combined_g_tabs[0],
            'r': combined_r_tabs[0],
            'i': combined_i_tabs[0],
        },
        max_arcsec=3,
        ztf_data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
    )
    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f'{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs]
        dec_strs = [f'{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs]
        candidate_names = [f'{i}_candidate_{ra_str}_{dec_str}.pdf' for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in
            zip(
                candidate_names,
                srcs,
            )
        ]
        pool.starmap(save_src_plot, args)


    ### IN JUST ZTF ###
    # plot_dir = os.path.join(get_data_path(), 'filter_results/candidates/in_ztf')
    plot_dir = '/Volumes/T7/long_transients/candidates/in_ztf'
    if os.path.exists(plot_dir) and OVERWRITE:
        shutil.rmtree(plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    srcs_ztf = Sources(
        ras=combined_coords[1]['ra'],
        decs=combined_coords[1]['dec'],
        field_catalogs={
            'g': combined_g_tabs[1],
            'r': combined_r_tabs[1],
            'i': combined_i_tabs[1],
        },
        max_arcsec=3,
        ztf_data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
    )
    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f'{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs_ztf]
        dec_strs = [f'{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs_ztf]
        candidate_names = [f'{i}_candidate_{ra_str}_{dec_str}.pdf' for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in
            zip(
                candidate_names,
                srcs_ztf,
            )
        ]
        pool.starmap(save_src_plot, args)

    # Wide associations in ZTF
    srcs_ztf_wide = Sources(
        ras=combined_wide_coords[1]['ra'],
        decs=combined_wide_coords[1]['dec'],
        field_catalogs={
            'g': combined_g_wide_tabs[1],
            'r': combined_r_wide_tabs[1],
            'i': combined_i_wide_tabs[1],
        },
        max_arcsec=3,
        ztf_data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
    )
    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f'{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs_ztf_wide]
        dec_strs = [f'{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}' for src in srcs_ztf_wide]
        candidate_names = [f'{i}_candidate_wide_{ra_str}_{dec_str}.pdf' for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in
            zip(
                candidate_names,
                srcs_ztf_wide,
            )
        ]
        pool.starmap(save_src_plot, args)

    # ### IN JUST PanSTARRS ###
    # plot_dir = '/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_pstarr'
    # if os.path.exists(plot_dir) and OVERWRITE:
    #     shutil.rmtree(plot_dir)
    # if not os.path.exists(plot_dir):
    #     os.mkdir(plot_dir)

    # srcs_pstarr = Sources(
    #     ras=combined_g_tabs[2]['ra'],
    #     decs=combined_g_tabs[2]['dec'],
    #     field_catalogs={
    #         'g': combined_g_tabs[2],
    #         'r': combined_r_tabs[2],
    #         'i': combined_i_tabs[2],
    #     },
    # )
    # for src in srcs_pstarr:
    #     out_fname = os.path.join(plot_dir, f'candidate_{i}.pdf')
    #     if os.path.exists(out_fname) and not OVERWRITE:
    #         print(f'Skipping source... Already plotting and saved at {out_fname}')
    #     else:
    #         print(f'Plotting source {i} / {len(srcs_pstarr)} at ({src.ra}, {src.dec})!')
    #         src.plot_everything()
    #         plt.savefig(out_fname, bbox_inches='tight')


if __name__ == '__main__':
    store_source_plots()
