"""Save per-source ZTF forced-photometry lightcurve PDFs.

For every source in the local sync index (source_map.csv), generates a
lightcurve plot and saves it as a PDF in
  Data/followup/ztf_forced_photometry/plots/

Files are named using the same coordinate-encoding convention used elsewhere
in Data/followup/:
  {ra}p{dec}.pdf  (decimal point → 'p', minus sign → 'n')
"""
import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from ztf_fp_query.sync import get_data_dir, load_map
from ztf_fp_query.forced_photometry import ForcedPhotometry
from Source_Analysis.coord_utils import coord_stem

OVERWRITE = True
PLOTS_DIR = os.path.join(get_data_dir(), 'plots')


def fp_plot_fname(ra: float, dec: float) -> str:
    return f"{coord_stem(ra, dec)}.pdf"


def save_fp_plot(ra: float, dec: float, out_fname: str, overwrite: bool = OVERWRITE) -> None:
    if os.path.exists(out_fname) and not overwrite:
        print(f'Skipping ({ra}, {dec}) — already saved at {out_fname}')
        return
    try:
        fp = ForcedPhotometry(ra, dec)
        fig = fp.plot_rolling_stack_summary()
        plt.savefig(out_fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_fname}')
    except Exception:
        print(f'Failed for ({ra}, {dec}):')
        traceback.print_exc()


def store_fp_plots(overwrite: bool = OVERWRITE) -> None:
    """Generate and save a lightcurve PDF for every synced source."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    map_df = load_map()
    if map_df.empty:
        print('No synced sources found. Run ztf_fp_query.sync.sync() first.')
        return

    print(f'Saving FP plots for {len(map_df)} source(s) → {PLOTS_DIR}')
    for _, row in map_df.iterrows():
        ra, dec = float(row['ra']), float(row['dec'])
        out_fname = os.path.join(PLOTS_DIR, fp_plot_fname(ra, dec))
        save_fp_plot(ra, dec, out_fname, overwrite=overwrite)


if __name__ == '__main__':
    store_fp_plots()
