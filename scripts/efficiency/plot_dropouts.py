"""Save per-filter dropout cutout grids for one injection run.

Called by run.py for seed=0 only.  One PDF per filter step per magnitude,
saved under <out_dir>/mag_<XX.X>/<slug>.pdf.
"""

import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.visualization import simple_norm

CUTOUT_HW = 25
N_COLS    = 8

STEP_COLORS = {
    'not_extracted':          '#AAAAAA',
    'flagged_by_cross_match': '#BB5500',
    'matched_to_panstarrs':   '#4477AA',
    'failed_sep_flags':       '#EE6677',
    'failed_ext_artifact':    '#FF4444',
    'failed_snr':             '#FF8800',
    'failed_shape':           '#CCBB44',
    'failed_psf_fit':         '#9933CC',
    'failed_dec':             '#00BBBB',
    'failed_delta_mag':       '#AA3377',
    'recovered':              '#228833',
}

# (slug, human label) in pipeline order
STEP_META = [
    ('not_extracted',          'Not extracted'),
    ('flagged_by_cross_match', 'Flagged by cross-match (near NaN / no PanSTARRS coverage)'),
    ('matched_to_panstarrs',   'Matched to PanSTARRS'),
    ('failed_sep_flags',       'Failed SEP flags'),
    ('failed_ext_artifact',    'Failed extended source artifact (SEP flag=1 & PSF-Kron > 1.5)'),
    ('failed_snr',             'Failed SNR filter'),
    ('failed_shape',           'Failed shape filter'),
    ('failed_psf_fit',         'Failed PSF fit (bad PSF flags)'),
    ('failed_dec',             'Failed declination cut'),
    ('failed_delta_mag',       'Failed delta-mag'),
    ('recovered',              'Recovered'),
]


def plot_dropout_cutouts(
    mag: float,
    seed: int,
    coords: np.ndarray,
    orig_img: np.ndarray,
    inj_img: np.ndarray,
    dropout_index_sets: dict,
    out_dir: str,
    extra_info: dict | None = None,
    pixel_scale_arcsec: float = 1.01,
) -> None:
    """Save one PDF per filter step showing all dropped/recovered cutouts.

    Each source gets two side-by-side panels: original image | injected image.

    Args:
        mag: Injected magnitude.
        seed: Random seed used for this run (for the suptitle).
        coords: (N, 2) array of (row, col) injection positions.
        orig_img: 2-D original (pre-injection) image array (g-band).
        inj_img: 2-D injected image array (g-band).
        dropout_index_sets: mapping slug -> set of coord indices.
        out_dir: Root output directory; mag sub-folder is created inside.
        extra_info: optional per-slug annotation data:
            'failed_sep_flags'     -> {src_i: int flag value}
            'matched_to_panstarrs' -> {src_i: (row, col) PanSTARRS pixel pos}
    """
    if extra_info is None:
        extra_info = {}
    mag_dir = os.path.join(out_dir, f'mag_{mag:.1f}')
    os.makedirs(mag_dir, exist_ok=True)

    for slug, label in STEP_META:
        idxs = dropout_index_sets.get(slug, set())
        if len(idxs) == 0:
            continue

        color   = STEP_COLORS[slug]
        samples = sorted(idxs)
        n_src   = len(samples)
        # Each source occupies 2 columns (orig | inj); N_COLS sets sources per row
        n_src_cols = min(N_COLS, n_src)
        n_rows     = math.ceil(n_src / n_src_cols)
        n_cols     = n_src_cols * 2

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.2 * n_cols, 2.6 * n_rows),
            squeeze=False,
        )

        for idx, src_i in enumerate(samples):
            r, c   = coords[src_i]
            r0     = max(0, r - CUTOUT_HW)
            c0     = max(0, c - CUTOUT_HW)
            rs     = slice(r0, min(orig_img.shape[0], r + CUTOUT_HW + 1))
            cs     = slice(c0, min(orig_img.shape[1], c + CUTOUT_HW + 1))
            cy     = r - r0   # injection centre within the cutout
            cx     = c - c0

            row_i      = idx // n_src_cols
            src_col    = idx  % n_src_cols

            orig_cut = orig_img[rs, cs]
            if orig_cut.size == 0 or not np.any(np.isfinite(orig_cut)):
                for panel in range(2):
                    axes[row_i, src_col * 2 + panel].set_visible(False)
                continue
            norm = simple_norm(orig_cut, 'sqrt', percent=99.5)

            for panel, img in enumerate([orig_img, inj_img]):
                cut = img[rs, cs]
                ax  = axes[row_i, src_col * 2 + panel]
                ax.imshow(cut, norm=norm, origin='lower', cmap='gray_r')

                # Red overlay for NaN-masked pixels (cross-match dropout step only)
                if slug == 'flagged_by_cross_match':
                    nan_mask = extra_info.get('nan_mask')
                    if nan_mask is not None:
                        mask_cut = nan_mask[rs, cs]
                        if mask_cut.shape == cut.shape:
                            red = np.zeros((*mask_cut.shape, 4), dtype=float)
                            red[mask_cut, 0] = 1.0
                            red[mask_cut, 3] = 0.55
                            ax.imshow(red, origin='lower', interpolation='nearest')

                ax.axhline(cy, color=color, lw=1.0, alpha=0.8)
                ax.axvline(cx, color=color, lw=1.0, alpha=0.8)

                # PanSTARRS match marker (both panels)
                if slug == 'matched_to_panstarrs':
                    ps_pos = extra_info.get('matched_to_panstarrs', {}).get(src_i)
                    if ps_pos is not None:
                        ps_cy = ps_pos[0] - r0
                        ps_cx = ps_pos[1] - c0
                        ax.plot(ps_cx, ps_cy, '+', color='cyan',
                                ms=10, mew=1.5, zorder=5)
                        radius_px = 1.0 / pixel_scale_arcsec
                        circle = mpatches.Circle(
                            (ps_cx, ps_cy), radius=radius_px,
                            edgecolor='cyan', facecolor='none',
                            lw=1.2, zorder=5,
                        )
                        ax.add_patch(circle)

                ax.set_xticks([]); ax.set_yticks([])
                # Column headers on first row
                if row_i == 0:
                    ax.set_title('orig' if panel == 0 else 'inj', fontsize=7, pad=2)
                # Source index label on left panel only
                if panel == 0:
                    ax.set_ylabel(f'src {src_i}', fontsize=6, labelpad=2)

                # SEP flag value annotated on the injected panel
                if slug == 'failed_sep_flags' and panel == 1:
                    flag_val = extra_info.get('failed_sep_flags', {}).get(src_i)
                    if flag_val is not None:
                        ax.text(0.03, 0.97, f'flag={flag_val}',
                                transform=ax.transAxes, fontsize=6,
                                va='top', ha='left', color='white',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          fc='black', alpha=0.6))

                # qfit value annotated on the injected panel
                if slug == 'failed_psf_fit' and panel == 1:
                    qfit_val = extra_info.get('failed_psf_fit', {}).get(src_i)
                    if qfit_val is not None:
                        ax.text(0.03, 0.97, f'qfit={qfit_val:.3f}',
                                transform=ax.transAxes, fontsize=6,
                                va='top', ha='left', color='white',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          fc='black', alpha=0.6))

        # Hide unused panels in the last row
        for idx in range(n_src, n_rows * n_src_cols):
            row_i   = idx // n_src_cols
            src_col = idx  % n_src_cols
            for panel in range(2):
                axes[row_i, src_col * 2 + panel].set_visible(False)

        fig.suptitle(
            f'{label}  —  {n_src} sources  |  left=original  right=injected\n'
            f'g-band, mag={mag:.1f}, seed={seed}',
            fontsize=11, color=color, y=1.01,
        )
        plt.tight_layout()
        out_path = os.path.join(mag_dir, f'{slug}.pdf')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
