"""Script for saving plots of all the extracted sources."""
import os
import shutil
import matplotlib.pyplot as plt

from astropy.table import Table

from Sources import Sources

OVERWRITE = False


def store_source_plots():
    """Store the plots for each candidate resulting from filtering."""
    # Load the stored files
    combined_g_tabs = {}
    combined_r_tabs = {}
    combined_i_tabs = {}
    combined_g_wide_tabs = {}
    combined_r_wide_tabs = {}
    combined_i_wide_tabs = {}
    for cat_num in range(2):  # TODO make this range(3) and add pstarr only
        combined_g_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_g.ecsv', format='ascii.ecsv')
        combined_r_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_r.ecsv', format='ascii.ecsv')
        combined_i_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_i.ecsv', format='ascii.ecsv')

        combined_g_wide_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_wide_g.ecsv', format='ascii.ecsv')
        combined_r_wide_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_wide_r.ecsv', format='ascii.ecsv')
        combined_i_wide_tabs[cat_num] = Table.read(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat_num}_wide_i.ecsv', format='ascii.ecsv')

    ### IN BOTH ###
    plot_dir = '/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_both'
    if os.path.exists(plot_dir) and OVERWRITE:
        shutil.rmtree(plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    srcs = Sources(
        ras=combined_g_tabs[0]['ra'],
        decs=combined_g_tabs[0]['dec'],
        field_catalogs={
            'g': combined_g_tabs[0],
            'r': combined_r_tabs[0],
            'i': combined_i_tabs[0],
        },
        max_arcsec=3,
    )
    for i, src in enumerate(srcs):
        out_fname = os.path.join(plot_dir, f'candidate_{i}.pdf')
        if os.path.exists(out_fname) and not OVERWRITE:
            print(f'Skipping source... Already plotting and saved at {out_fname}')
        else:
            print(f'Plotting source {i} / {len(srcs)} at ({src.ra}, {src.dec})!')
            src.plot_everything()
            plt.savefig(out_fname, bbox_inches='tight')


    ### IN JUST ZTF ###
    plot_dir = '/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_ztf'
    if os.path.exists(plot_dir) and OVERWRITE:
        shutil.rmtree(plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    srcs_ztf = Sources(
        ras=combined_g_tabs[1]['ra'],
        decs=combined_g_tabs[1]['dec'],
        field_catalogs={
            'g': combined_g_tabs[1],
            'r': combined_r_tabs[1],
            'i': combined_i_tabs[1],
        },
    )
    for src in srcs_ztf:
        out_fname = os.path.join(plot_dir, f'candidate_{i}.pdf')
        if os.path.exists(out_fname) and not OVERWRITE:
            print(f'Skipping source... Already plotting and saved at {out_fname}')
        else:
            print(f'Plotting source {i} / {len(srcs_ztf)} at ({src.ra}, {src.dec})!')
            src.plot_everything()
            plt.savefig(out_fname, bbox_inches='tight')

    # Wide associations in ZTF
    srcs_ztf_wide = Sources(
        ras=combined_i_wide_tabs[1]['ra'],
        decs=combined_i_wide_tabs[1]['dec'],
        field_catalogs={
            'g': combined_g_wide_tabs[1],
            'r': combined_r_wide_tabs[1],
            'i': combined_i_wide_tabs[1],
        },
        max_arcsec=3,
    )
    for src in srcs_ztf_wide:
        out_fname = os.path.join(plot_dir, f'candidate_{i}_wide.pdf')
        if os.path.exists(out_fname) and not OVERWRITE:
            print(f'Skipping source... Already plotting and saved at {out_fname}')
        else:
            print(f'Plotting source {i} / {len(srcs_ztf_wide)} at ({src.ra}, {src.dec})!')
            src.plot_everything()
            plt.savefig(out_fname, bbox_inches='tight')

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
