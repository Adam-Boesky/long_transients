"""Save each target's ZTF forced-photometry lightcurve PDF.

For every follow-up target with ZFPS results, plots the most recent request to
  Data/followup/<slug>/zfps/zfps.pdf
"""
import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis import followup
from ztf_fp_query.forced_photometry import ForcedPhotometry

OVERWRITE = True


def save_fp_plot(fpath: str, out_fname: str, overwrite: bool = OVERWRITE) -> None:
    if os.path.exists(out_fname) and not overwrite:
        print(f'Skipping {fpath} — already saved at {out_fname}')
        return
    try:
        fp = ForcedPhotometry.from_file(fpath)
        fig = fp.plot_rolling_stack_summary()
        plt.savefig(out_fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_fname}')
    except Exception:
        print(f'Failed for {fpath}:')
        traceback.print_exc()


def store_fp_plots(overwrite: bool = OVERWRITE) -> None:
    """Generate and save a lightcurve PDF for every target with ZFPS results."""
    targets = [t for t in followup.iter_targets() if t.zfps_files]
    if not targets:
        print('No ZFPS results found. Run ztf_fp_query.sync.sync() first.')
        return

    print(f'Saving FP plots for {len(targets)} target(s)')
    for t in targets:
        # Request ids increase with time, so the last file is the latest request.
        latest = t.zfps_files[-1]
        save_fp_plot(str(latest), str(t.path / followup.ZFPS_DIR / followup.ZFPS_PLOT), overwrite=overwrite)


if __name__ == '__main__':
    store_fp_plots()
