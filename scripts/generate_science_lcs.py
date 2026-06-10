"""Generate forced-PSF photometry lightcurves on ZTF science images for all sources
in Data/followup/ztf_forced_photometry/source_map.csv using ztforce.

Output layout:
  Data/followup/ztforce_photometry/
    lightcurves/   — per-source per-band ECSV files (ztforce cache)
    plots/         — one PDF per source with single-epoch + rolling-stack panels
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SOURCE_MAP = REPO_ROOT / "Data" / "followup" / "ztf_forced_photometry" / "source_map.csv"
OUT_DIR = REPO_ROOT / "Data" / "followup" / "ztforce_photometry"
PLOT_DIR = OUT_DIR / "plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── ztforce ────────────────────────────────────────────────────────────────────
from ztforce import run_forced_photometry_batch, build_config
from ztforce.lightcurve import Lightcurve
from Source_Analysis.coord_utils import coord_stem

BAND_COLORS = {"g": "forestgreen", "r": "lightcoral", "i": "darkorchid"}
BANDS = ("g", "r", "i")

_WINDOW_DAYS = (30, 182, 365, 730)
_WINDOW_IMGS = (10, 30, 60, 100)


def source_stem(ra: float, dec: float) -> str:
    return coord_stem(ra, dec)


def _plot_single_epoch(lcs: dict[str, Lightcurve], bands: list[str], ax: plt.Axes) -> None:
    """Plot per-epoch detections and upper limits for all bands onto ax."""
    for band in bands:
        df = lcs[band].df
        color = BAND_COLORS[band]

        det = df[df["detection"]]
        if not det.empty:
            ax.errorbar(
                det["obsjd"], det["mag"], yerr=det["mag_err"],
                fmt="o", color=color, markersize=4, linewidth=0.8, capsize=2,
                label=band,
            )

        ul = df[~df["detection"] & np.isfinite(df["upper_limit"])]
        if not ul.empty:
            ax.scatter(ul["obsjd"], ul["upper_limit"], marker="v", color=color, alpha=0.35, s=16)

    ax.invert_yaxis()
    ax.set_ylabel("Magnitude (AB)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_rolled(rolled: pd.DataFrame, bands: list[str], ax: plt.Axes, fmt: str) -> None:
    """Plot a rolling-stack DataFrame onto ax."""
    if rolled.empty or "band" not in rolled.columns:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=9)
        ax.invert_yaxis()
        return
    for band in bands:
        grp = rolled[rolled["band"] == band].sort_values("obsjd_center")
        if grp.empty:
            continue
        color = BAND_COLORS[band]
        valid = np.isfinite(grp["mag_stack"])
        if valid.any():
            ax.errorbar(
                grp["obsjd_center"][valid], grp["mag_stack"][valid],
                yerr=grp["mag_err_stack"][valid],
                fmt=fmt, color=color, markersize=4, linewidth=0.8, capsize=2,
                label=band,
            )
    ax.invert_yaxis()
    ax.set_ylabel("Magnitude (AB)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def _date_ticks(ax: plt.Axes) -> None:
    """Replace JD x-ticks with human-readable date strings."""
    ticks = ax.get_xticks()
    finite = ticks[np.isfinite(ticks)]
    if len(finite) == 0:
        return
    dates = Time(finite, format="jd").strftime("%m-%d-%Y")
    ax.set_xticks(finite, dates, rotation=45, ha="right")
    ax.set_xlabel("Date")


def plot_rolling_stack_summary(
    lcs: dict[str, Lightcurve],
    ra: float,
    dec: float,
    out_path: Path,
) -> None:
    """Single-epoch on top + 4×2 grid of day/image rolling stacks, matching
    the layout produced by ForcedPhotometry.plot_rolling_stack_summary()."""
    bands = [b for b in BANDS if b in lcs and len(lcs[b]) > 0]
    if not bands:
        print(f"  [skip] no data for ({ra:.5f}, {dec:.5f})", flush=True)
        return

    # Merge all bands into a single Lightcurve-like object for convenience.
    # We'll query each Lightcurve individually instead.
    n_rows = 1 + len(_WINDOW_DAYS)  # 5 rows total
    fig = plt.figure(figsize=(20, 5 * n_rows))
    ax_single = plt.subplot2grid((n_rows, 2), (0, 0), colspan=2)

    # Panel config: (window, unit, fmt, grid_position)
    panel_cfg = (
        [(w, "days",   "s", (r, 0)) for r, w in enumerate(_WINDOW_DAYS, start=1)] +
        [(w, "images", "^", (r, 1)) for r, w in enumerate(_WINDOW_IMGS,  start=1)]
    )
    panel_axes = {
        pos: plt.subplot2grid((n_rows, 2), pos, sharex=ax_single)
        for _, _, _, pos in panel_cfg
    }

    _plot_single_epoch(lcs, bands, ax_single)
    ax_single.set_title("Single-epoch")
    plt.setp(ax_single.get_xticklabels(), visible=False)
    ax_single.set_xlabel("")

    for window, unit, fmt, pos in panel_cfg:
        ax = panel_axes[pos]
        rolled_parts = []
        for b in bands:
            r = lcs[b].rolling_stack(window=window, window_unit=unit)
            rolled_parts.append(r)
        rolled = pd.concat(rolled_parts, ignore_index=True) if rolled_parts else pd.DataFrame()
        _plot_rolled(rolled, bands, ax, fmt)
        ax.set_title(f"{window}-{'day' if unit == 'days' else 'image'} rolling stack")
        if pos[1] == 1:
            ax.set_ylabel("")
        if pos[0] < n_rows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")

    # Date-string ticks on bottom row only
    for pos, ax in panel_axes.items():
        if pos[0] == n_rows - 1:
            _date_ticks(ax)

    fig.suptitle(
        f"ZTF science forced photometry  |  RA={ra:.5f}  Dec={dec:.5f}",
        fontsize=11, y=1.002,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path.name}", flush=True)


def main() -> None:
    src = pd.read_csv(SOURCE_MAP)
    print(f"Loaded {len(src)} sources from {SOURCE_MAP}", flush=True)

    targets = [SkyCoord(ra=row.ra, dec=row.dec, unit="deg") for _, row in src.iterrows()]
    config = build_config()

    print("Running ztforce batch (this will take a while)…", flush=True)
    all_lcs = run_forced_photometry_batch(
        targets=targets,
        bands=BANDS,
        data_dir=OUT_DIR,
        config=config,
        n_workers=3,
        download_workers=12,
        show_progress=True,
    )

    print("\nSaving plots…", flush=True)
    for i, (row, lcs) in enumerate(zip(src.itertuples(), all_lcs)):
        ra, dec = float(row.ra), float(row.dec)
        stem = source_stem(ra, dec)
        print(f"[{i+1}/{len(src)}] ({ra:.5f}, {dec:.5f})", flush=True)
        plot_rolling_stack_summary(lcs, ra, dec, PLOT_DIR / f"{stem}.pdf")

    print(f"\nDone. Plots in {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    main()
