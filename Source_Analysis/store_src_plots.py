"""Script for saving plots of all the extracted sources."""
import os
import sys
import shutil
import traceback
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from Extracting.utils import get_data_path
from Sources import Source, Sources
from Source_Analysis.coord_utils import coord_stem
from multiprocessing import Pool

# Plot formatting
plt.rc('text', usetex=True)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
mpl.rcParams['font.size'] = 12  # Adjust the font size as needed
mpl.rcParams['axes.formatter.use_mathtext'] = True

OVERWRITE = False
CANDIDATE_DIR = 'kde_analysis_pages_5_21_2026'
FILTER_RESULTS_DIRNAME = 'filter_results_kde'


PSTARR_UPPER_LIM = {'g': 23.3, 'r': 23.2, 'i': 23.1}


def mean_abs_dmag(src: Source) -> float:
    """Return the nanmean of absolute dmag (ZTF - PanSTARRS) across bands.

    Mirrors the only_big_dmag logic: substitutes PSTARR upper limits when
    Catalog_Flag==1 (ZTF-only) and ZTF mag limits when Catalog_Flag==2 (PanSTARRS-only).
    """
    data = src.data
    colnames = data.colnames
    dmags = []
    for band in ('g', 'r', 'i'):
        flag_col = f'{band}_Catalog_Flag' if f'{band}_Catalog_Flag' in colnames else 'Catalog_Flag'
        if flag_col not in colnames:
            continue
        flag = data[flag_col][0]

        ztf_col = f'ZTF_{band}PSFMag'
        ps_col = f'PSTARR_{band}PSFMag'
        lim_col = f'ZTF_{band}_mag_limit'

        ps = PSTARR_UPPER_LIM[band] if flag == 1 else (data[ps_col][0] if ps_col in colnames else float('nan'))
        ztf = data[lim_col][0] if (flag == 2 and lim_col in colnames) else (data[ztf_col][0] if ztf_col in colnames else float('nan'))

        if np.isfinite(ztf) and np.isfinite(ps):
            dmags.append(abs(float(ztf - ps)))
    return float(np.nanmean(dmags)) if dmags else float('nan')


def sort_by_dmag(srcs: Sources) -> Sources:
    """Return a new Sources sorted descending by mean absolute dmag."""
    return Sources(sources=sorted(srcs, key=mean_abs_dmag, reverse=True))


def write_dmag_sidecar(srcs: Sources, fnames: list[str], plot_dir: str, filename: str = 'dmag_order.csv') -> None:
    """Write a CSV recording the dmag-sorted order of sources."""
    rows = []
    for rank, (src, fname) in enumerate(zip(srcs, fnames), start=1):
        row = {'rank': rank, 'filename': fname, 'ra': src.ra, 'dec': src.dec}
        for band in ('g', 'r', 'i'):
            flag_col = f'{band}_Catalog_Flag' if f'{band}_Catalog_Flag' in src.data.colnames else 'Catalog_Flag'
            flag = src.data[flag_col][0] if flag_col in src.data.colnames else float('nan')
            ztf_col, ps_col, lim_col = f'ZTF_{band}PSFMag', f'PSTARR_{band}PSFMag', f'ZTF_{band}_mag_limit'
            ps = PSTARR_UPPER_LIM[band] if flag == 1 else (float(src.data[ps_col][0]) if ps_col in src.data.colnames else float('nan'))
            ztf = float(src.data[lim_col][0]) if (flag == 2 and lim_col in src.data.colnames) else (float(src.data[ztf_col][0]) if ztf_col in src.data.colnames else float('nan'))
            row[f'{band}_dmag'] = float(ztf - ps) if (np.isfinite(ztf) and np.isfinite(ps)) else float('nan')
        row['mean_abs_dmag'] = mean_abs_dmag(src)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(plot_dir, filename), index=False)


def src_fname(src: Source, prefix: str = '') -> str:
    return f"{prefix}{coord_stem(src.ra, src.dec)}.pdf"


def save_src_plot(src: Source, out_fname: str, overwrite: bool, n_attempts: int = 3):
    for i_attempt in range(n_attempts):
        try:
            if os.path.exists(out_fname) and not overwrite:
                print(f'Skipping source... Already plotted and saved at {out_fname}')
                break
            else:
                print(f"Plotting source {out_fname.split('/')[-1].split('.')[0]} at ({src.ra}, {src.dec})!")
                src.plot_everything()
                plt.savefig(out_fname, bbox_inches='tight')
                plt.close('all')
                break  # Success - exit the loop
        except Exception as e:
            plt.close('all')
            if i_attempt >= n_attempts - 1:
                print(f'Final attempt failed for {out_fname.split("/")[-1].split(".")[0]}: {str(e)}')
                print(f'Full traceback:')
                traceback.print_exc()
                raise
            else:
                print(f'Attempt {i_attempt + 1} / {n_attempts} to save {out_fname.split("/")[-1].split(".")[0]} failed: {str(e)}')
                print(f'Full traceback:')
                traceback.print_exc()
                print('Trying again...')
    print(f'Store source plot at {out_fname}')


def store_source_plots():
    """Store the plots for each candidate resulting from filtering."""
    if os.path.exists('/Volumes/T7/long_transients/'):
        path_to_data = '/Volumes/T7/long_transients/'
    else:
        path_to_data = get_data_path()
    if not os.path.exists(os.path.join(path_to_data, CANDIDATE_DIR)):
        os.mkdir(os.path.join(path_to_data, CANDIDATE_DIR))

    # Kwargs for Sources in all three catalogs
    src_kwargs = {
        'ztf_data_dir': os.path.join(path_to_data, 'ztf_data'),
    }


    ### IN BOTH CATALOGS ###
    plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_both')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    print('-'*100)
    print('IN BOTH CATALOG')
    print('-'*100)
    srcs = sort_by_dmag(Sources.from_file(
        os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/0.ecsv'),
        **src_kwargs,
    ))
    print(f'Finished loading {len(srcs)} sources')
    candidate_names = [src_fname(src) for src in srcs]
    write_dmag_sidecar(srcs, candidate_names, plot_dir)
    with Pool(processes=3) as pool:
        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in zip(candidate_names, srcs)
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred
    del srcs


    ### IN JUST ZTF ###
    print('-'*100)
    print('IN JUST ZTF')
    print('-'*100)
    srcs_ztf = sort_by_dmag(Sources.from_file(
        os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1.ecsv'),
        **src_kwargs,
    ))
    print(f'Finished loading {len(srcs_ztf)} sources')
    plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_ztf')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    candidate_names = [src_fname(src) for src in srcs_ztf]
    write_dmag_sidecar(srcs_ztf, candidate_names, plot_dir)
    with Pool(processes=3) as pool:
        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in zip(candidate_names, srcs_ztf)
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred
    del srcs_ztf

    # Wide associations in ZTF
    srcs_ztf_wide = sort_by_dmag(Sources.from_file(os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1_wide_association.ecsv')))
    candidate_names = [src_fname(src, prefix='wide_') for src in srcs_ztf_wide]
    write_dmag_sidecar(srcs_ztf_wide, candidate_names, plot_dir, filename='dmag_order_wide.csv')
    with Pool(processes=3) as pool:
        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 1)
            for cand_name, src in zip(candidate_names, srcs_ztf_wide)
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred
    del srcs_ztf_wide


    ### IN JUST PanSTARRS ###
    # srcs_pstarr = Sources.from_file(
    #     os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/2.ecsv'),
    #     **src_kwargs,
    # )
    plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_pstarr')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # with Pool(processes=3) as pool:
    #     # Construct the source name
    #     ra_strs = [f"{f"{src.ra:.4f}".replace('.', 'p').replace('-', 'n')}" for src in srcs_pstarr]
    #     dec_strs = [f"{f"{src.dec:.4f}".replace('.', 'p').replace('-', 'n')}" for src in srcs_pstarr]
    #     candidate_names = [f"{i}_candidate_{ra_str}_{dec_str}.pdf" for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

    #     args = [
    #         (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
    #         for cand_name, src in
    #         zip(
    #             candidate_names,
    #             srcs_pstarr,
    #         )
    #     ]
    #     results = pool.starmap_async(save_src_plot, args)
    #     results.get()  # This will raise any exceptions that occurred

    # Wide associations in ZTF
    srcs_pstarr_wide = sort_by_dmag(Sources.from_file(os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/2_wide_association.ecsv')))
    candidate_names = [src_fname(src, prefix='wide_') for src in srcs_pstarr_wide]
    write_dmag_sidecar(srcs_pstarr_wide, candidate_names, plot_dir)
    with Pool(processes=3) as pool:
        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 1)
            for cand_name, src in zip(candidate_names, srcs_pstarr_wide)
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred


if __name__ == '__main__':
    store_source_plots()
