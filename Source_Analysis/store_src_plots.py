"""Script for saving plots of all the extracted sources."""
import os
import sys
import shutil
import traceback
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
CANDIDATE_DIR = 'gemini_analysis_pages'
FILTER_RESULTS_DIRNAME = 'filter_results_gemini'


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
                break  # Success - exit the loop
        except Exception as e:
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


def store_source_plots():
    """Store the plots for each candidate resulting from filtering."""
    if os.path.exists('/Volumes/T7/long_transients/'):
        path_to_data = '/Volumes/T7/long_transients/'
    else:
        path_to_data = '/Users/adamboesky/Research/long_transients/Data'
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
    srcs = Sources.from_file(
        os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/0.ecsv'),
        **src_kwargs,
    )
    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f"{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs]
        dec_strs = [f"{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs]
        candidate_names = [f"{i}_candidate_{ra_str}_{dec_str}.pdf" for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in
            zip(
                candidate_names,
                srcs,
            )
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred


    ### IN JUST ZTF ###
    srcs_ztf = Sources.from_file(
        os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1.ecsv'),
        **src_kwargs,
    )
    plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_ztf')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f"{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs_ztf]
        dec_strs = [f"{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs_ztf]
        candidate_names = [f"{i}_candidate_{ra_str}_{dec_str}.pdf" for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
            for cand_name, src in
            zip(
                candidate_names,
                srcs_ztf,
            )
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred

    # Wide associations in ZTF
    srcs_ztf_wide = Sources.from_file(os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1_wide_association.ecsv'))
    with Pool(processes=3) as pool:
        # Construct the source name
        ra_strs = [f"{str(src.ra).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs_ztf_wide]
        dec_strs = [f"{str(src.dec).replace('.', 'p').replace('-', 'n')[:6]}" for src in srcs_ztf_wide]
        candidate_names = [f"{i}_candidate_wide_{ra_str}_{dec_str}.pdf" for i, (ra_str, dec_str) in enumerate(zip(ra_strs, dec_strs))]

        args = [
            (src, os.path.join(plot_dir, cand_name), OVERWRITE, 1)
            for cand_name, src in
            zip(
                candidate_names,
                srcs_ztf_wide,
            )
        ]
        results = pool.starmap_async(save_src_plot, args)
        results.get()  # This will raise any exceptions that occurred

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
