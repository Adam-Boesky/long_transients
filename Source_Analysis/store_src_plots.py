"""Script for saving plots of all the extracted sources."""
import os

from source_db.constants import PSTARR_UPPER_LIM
import sys
import shutil
import traceback
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from Extracting.utils import get_data_path, init_worker_sdss_lock, load_ecsv
from Sources import Source, Sources
from Source_Analysis.coord_utils import coord_stem
from multiprocessing import Pool, Lock, Value

# Plot formatting
plt.rc('text', usetex=True)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
mpl.rcParams['font.size'] = 12  # Adjust the font size as needed
mpl.rcParams['axes.formatter.use_mathtext'] = True

OVERWRITE = False
CANDIDATE_DIR = 'analysis_pages_gemini_2027a_v2'
# The run the exciting-source list was drawn from, and the one the local
# database is built from -- see the header of EXCITING_SRCS_FILE.
FILTER_RESULTS_DIRNAME = 'filter_results_9_21_2026_full_catalog_fixed_pm'

# ── the Gemini 2027A target list ──────────────────────────────────────────
#
# Only the hand-picked exciting sources that are observable in semester A.
EXCITING_SRCS_FILE = Path(__file__).resolve().parents[1] / 'exciting_srcs_radec_v2.txt'

# RA runs 04h -> 01h *through 0h*, i.e. everything except 01h-04h, which sets
# too early to be useful in a February-July semester. Written as the pair it
# wraps between rather than a min/max, because a plain `lo < ra < hi` cannot
# express a range that crosses 0h.
RA_HOURS_FROM, RA_HOURS_TO = 4.0, 1.0
DEC_RANGE = (-37.0, 90.0)

# Match radius when locating an exciting source in the candidate table. The
# coordinates were written out of that same table, so this only has to absorb
# the text file's 7-decimal rounding.
EXCITING_MATCH_ARCSEC = 1.0

# Serve each source from the local source database instead of the live
# services. The catalog row, the light curves and the SIMBAD / AGN-DB
# crossmatches all come from `Data/db`, which is ~8 network services and a
# 301 GB field-catalog directory replaced by a few parquet reads.
#
# Two things it does NOT cover, so a run is faster but not offline:
#   cutouts   the release stores stamp addresses and rendered PNGs, not FITS
#             pixels, and `plot_cutout` needs an array plus a WCS
#   spectra   the arrays are local but `plot_spectrum` reads HDUList internals
# Both fall back to their usual path.
#
# The store is per-source and declines what it does not have, so a candidate
# missing from the release falls back rather than failing -- see the release
# check in `make_store`, which is where that stops being invisible.
LOCAL_DB = True




def exciting_coords() -> SkyCoord:
    """The exciting sources inside the Gemini 2027A window.

    The RA cut wraps through 0h, so it is an OR rather than a range: keep
    anything at or after 04h, plus anything before 01h.
    """
    data = np.loadtxt(EXCITING_SRCS_FILE)
    ra, dec = data[:, 0], data[:, 1]
    hours = ra / 15.0
    in_ra = (hours > RA_HOURS_FROM) | (hours < RA_HOURS_TO)
    in_dec = (dec > DEC_RANGE[0]) & (dec < DEC_RANGE[1])
    keep = in_ra & in_dec
    print(f'Exciting sources: {keep.sum()} of {len(data)} inside '
          f'RA {RA_HOURS_FROM:g}h->{RA_HOURS_TO:g}h (through 0h), '
          f'Dec {DEC_RANGE[0]:g}..{DEC_RANGE[1]:g}')
    return SkyCoord(ra[keep], dec[keep], unit='deg')


def restrict_to_exciting(table: Table, wanted: SkyCoord) -> tuple[Table, np.ndarray]:
    """Subset a candidate table to the exciting sources, in the file's order.

    Matched on sky rather than by row index: the text file carries only
    coordinates, and rounding them to 7 decimals moves a position by well
    under a milliarcsecond. Also returns which of `wanted` matched, so the
    caller can report sources found in no table rather than drop them
    silently.
    """
    have = SkyCoord(np.asarray(table['ra'], dtype=float),
                    np.asarray(table['dec'], dtype=float), unit='deg')
    idx, sep, _ = wanted.match_to_catalog_sky(have)
    ok = sep.arcsec <= EXCITING_MATCH_ARCSEC
    print(f'  matched {int(ok.sum())} into the candidate table')
    return table[idx[ok]], ok


def make_store():
    """A `LocalStore` when `LOCAL_DB` is set, else None.

    Also reports whether the release actually matches the candidates being
    plotted. The two are independent: `FILTER_RESULTS_DIRNAME` chooses which
    filter run supplies the candidate list, while the database is built from
    whichever run `release.json` records. When they differ, sources that the
    release does not contain resolve to nothing and quietly fall back to the
    live services -- which is slow rather than wrong, but is worth knowing
    before a run of thousands of sources rather than after.
    """
    if not LOCAL_DB:
        return None

    import json
    from Source_Analysis.local_store import LocalStore
    from source_db.paths import DEFAULT_OUT_DIR

    store = LocalStore()
    release_path = Path(DEFAULT_OUT_DIR) / 'release.json'
    release = json.loads(release_path.read_text()) if release_path.exists() else {}
    built_from = Path(release.get('combined_dir', '')).parent.name

    print(f'LOCAL_DB: serving from release {store.release_id} '
          f'({release.get("n_sources", "?")} sources, built from {built_from or "?"})')
    if built_from and built_from != FILTER_RESULTS_DIRNAME:
        print(f'  WARNING: candidates come from {FILTER_RESULTS_DIRNAME}, which is '
              f'not what the release was built from. Sources absent from the '
              f'release will fall back to live queries.')
    print('  cutouts and spectra still use the live path.')
    return store


def sdss_throttled_pool(processes: int = 3) -> Pool:
    """Make a Pool whose workers share a lock + timestamp so SDSS queries across all
    of them stay under SDSS's rate limit (see throttled_sdss_query in Extracting.utils)."""
    return Pool(processes=processes, initializer=init_worker_sdss_lock, initargs=(Lock(), Value('d', 0.0)))


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


ZTFORCE_DIR = Path(__file__).resolve().parents[1] / 'Data' / 'followup' / 'ztforce_photometry'


def prefetch_ztforce(srcs: Sources) -> None:
    """Batch-fetch ztforce light curves for all sources so pool workers hit the disk cache."""
    from ztforce import run_forced_photometry_batch, build_config
    coords = [SkyCoord(src.ra, src.dec, unit='deg') for src in srcs]
    print(f'Pre-fetching ztforce light curves for {len(coords)} sources...')
    run_forced_photometry_batch(
        targets=coords,
        bands=('g', 'r', 'i'),
        data_dir=ZTFORCE_DIR,
        config=build_config(),
    )
    print('ztforce prefetch complete.')


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
        'store': make_store(),      # None unless LOCAL_DB
    }
    DEFAULT_LC_CATALOGS = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom']


    # ### IN BOTH CATALOGS ###
    # plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_both')
    # if not os.path.exists(plot_dir):
    #     os.mkdir(plot_dir)
    # print('-'*100)
    # print('IN BOTH CATALOG')
    # print('-'*100)
    # srcs = sort_by_dmag(Sources.from_file(
    #     os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/0.ecsv'),
    #     lc_catalogs=DEFAULT_LC_CATALOGS , # TODO: add + ['zubercal'],
    #     **src_kwargs,
    # ))
    # print(f'Finished loading {len(srcs)} sources')
    # candidate_names = [src_fname(src) for src in srcs]
    # write_dmag_sidecar(srcs, candidate_names, plot_dir)
    # with sdss_throttled_pool(processes=3) as pool:
    #     args = [
    #         (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
    #         for cand_name, src in zip(candidate_names, srcs)
    #     ]
    #     # chunksize=1 so tasks dispatch in dmag-rank order. The default chunksize
    #     # (len(pool)*4) hands each worker one contiguous block, which would start
    #     # them thousands of ranks apart. Dispatch overhead is negligible next to a
    #     # multi-second plot.
    #     results = pool.starmap_async(save_src_plot, args, chunksize=1)
    #     results.get()  # This will raise any exceptions that occurred
    # del srcs


    ### THE EXCITING SOURCES, IN WHICHEVER TIER HOLDS THEM ###
    # `.hdf5`, not `.ecsv`: this filter run only wrote the HDF5, and it is
    # what the exciting-source list was drawn from. `load_ecsv` reads both.
    wanted = exciting_coords()
    found = np.zeros(len(wanted), dtype=bool)
    for tier, subdir in [('0', 'in_both'), ('1', 'in_ztf')]:
        print('-'*100)
        print(subdir.upper())
        print('-'*100)
        candidates_path = os.path.join(
            path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/{tier}.hdf5')
        table, ok = restrict_to_exciting(load_ecsv(candidates_path), wanted)
        found |= ok
        if len(table) == 0:
            continue
        srcs = sort_by_dmag(Sources.from_table(
            table,
            lc_catalogs=DEFAULT_LC_CATALOGS, # TODO: add + ['ztforce'],
            filtering_dirpath=os.path.dirname(os.path.dirname(candidates_path)),
            **src_kwargs,
        ))
        print(f'Finished loading {len(srcs)} sources')
        plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, subdir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        candidate_names = [src_fname(src) for src in srcs]
        write_dmag_sidecar(srcs, candidate_names, plot_dir)
        # prefetch_ztforce(srcs) # TODO: add back!
        with sdss_throttled_pool(processes=3) as pool:
            args = [
                (src, os.path.join(plot_dir, cand_name), OVERWRITE, 3)
                for cand_name, src in zip(candidate_names, srcs)
            ]
            # chunksize=1 so tasks dispatch in dmag-rank order. The default chunksize
            # (len(pool)*4) hands each worker one contiguous block, which would start
            # them thousands of ranks apart. Dispatch overhead is negligible next to a
            # multi-second plot.
            results = pool.starmap_async(save_src_plot, args, chunksize=1)
            results.get()  # This will raise any exceptions that occurred
        del srcs

    if not found.all():
        print(f'WARNING: {int((~found).sum())} exciting source(s) found in no candidate table:')
        for c in wanted[~found]:
            print(f'    missing: {c.ra.deg:.7f} {c.dec.deg:.7f}')

    # Wide associations are not part of the Gemini 2027A list -- every
    # exciting source is in 1.hdf5 -- so this whole block is off.
    # Wide associations in ZTF
    # srcs_ztf_wide = sort_by_dmag(Sources.from_file(
    # os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1_wide_association.ecsv'),
    # lc_catalogs=DEFAULT_LC_CATALOGS + ['ztforce'],
    # ))
    # candidate_names = [src_fname(src, prefix='wide_') for src in srcs_ztf_wide]
    # write_dmag_sidecar(srcs_ztf_wide, candidate_names, plot_dir, filename='dmag_order_wide.csv')
    # prefetch_ztforce(srcs_ztf_wide)
    # with sdss_throttled_pool(processes=3) as pool:
    # args = [
    # (src, os.path.join(plot_dir, cand_name), OVERWRITE, 1)
    # for cand_name, src in zip(candidate_names, srcs_ztf_wide)
    # ]
    # chunksize=1 so tasks dispatch in dmag-rank order. The default chunksize
    # (len(pool)*4) hands each worker one contiguous block, which would start
    # them thousands of ranks apart. Dispatch overhead is negligible next to a
    # multi-second plot.
    # results = pool.starmap_async(save_src_plot, args, chunksize=1)
    # results.get()  # This will raise any exceptions that occurred
    # del srcs_ztf_wide


    ### IN JUST PanSTARRS ###
    # srcs_pstarr = Sources.from_file(
    #     os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/2.ecsv'),
    #     **src_kwargs,
    # )
    # plot_dir = os.path.join(path_to_data, CANDIDATE_DIR, 'in_pstarr')
    # if not os.path.exists(plot_dir):
    #     os.mkdir(plot_dir)

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

    # Off for the same reason as the ZTF wide block above: the Gemini 2027A
    # list lives entirely in 1.hdf5, and 2_wide_association.ecsv was never
    # written by this filter run -- leaving it live crashed the script after
    # all 76 pages had already been plotted.
    # Wide associations in PanSTARRS
    # srcs_pstarr_wide = sort_by_dmag(Sources.from_file(os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/2_wide_association.ecsv')))
    # candidate_names = [src_fname(src, prefix='wide_') for src in srcs_pstarr_wide]
    # write_dmag_sidecar(srcs_pstarr_wide, candidate_names, plot_dir)
    # with sdss_throttled_pool(processes=3) as pool:
    #     args = [
    #         (src, os.path.join(plot_dir, cand_name), OVERWRITE, 1)
    #         for cand_name, src in zip(candidate_names, srcs_pstarr_wide)
    #     ]
    #     # chunksize=1 so tasks dispatch in dmag-rank order. The default chunksize
    #     # (len(pool)*4) hands each worker one contiguous block, which would start
    #     # them thousands of ranks apart. Dispatch overhead is negligible next to a
    #     # multi-second plot.
    #     results = pool.starmap_async(save_src_plot, args, chunksize=1)
    #     results.get()  # This will raise any exceptions that occurred


if __name__ == '__main__':
    store_source_plots()
