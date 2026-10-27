"""Regenerate every follow-up spectrum plot from its target's `target.yaml`.

Each `spectra` entry's reduced `file` is plotted at the target's `redshift` to its `plot`
path; see `Source_Analysis.followup`.

Usage
-----
    python scripts/regen_spectra.py            # every target
    python scripts/regen_spectra.py <slug> ... # just these
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Source_Analysis import followup


def _bin(wavelength, flux, binsize):
    n = len(wavelength) // binsize
    wl = np.mean(wavelength[: n * binsize].reshape(-1, binsize), axis=1)
    fl = np.mean(flux[: n * binsize].reshape(-1, binsize), axis=1)
    return wl, fl


def plot_spectrum(txt_path, title, redshift, out_path):
    plt.rc("text", usetex=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cmr10"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.formatter.use_mathtext"] = True

    data = np.loadtxt(txt_path)
    wavelength, flux = data[:, 0], data[:, 1]

    telluric = (wavelength > 7580) & (wavelength < 7680)
    mask = ~telluric
    if mask.sum() > 70:
        w_bin, f_bin = _bin(wavelength[mask][50:-20], flux[mask][50:-20], 10)
    else:
        w_bin, f_bin = _bin(wavelength, flux, 10)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wavelength / (1 + redshift), flux * 1e17,
            color="k", linewidth=0.5, alpha=0.25)
    ax.plot(w_bin / (1 + redshift), f_bin * 1e17,
            color="k", linewidth=1)
    xlim = ax.get_xlim()

    ref_lines = [
        (2799.117, "red",    "Mg II"),
        (1549.48,  "green",  "C IV"),
        (1908.734, "blue",   "C III]"),
        (3726,     "purple", "O II"),
        (5007,     "orange", "O III"),
        (4861,     "black",  r"H$\beta$"),
        (6562.8,   "black",  r"H$\alpha$"),
        (6548,     "pink",   "N II"),
        (6583,     "pink",   None),
    ]
    for wl, color, label in ref_lines:
        kw = dict(color=color, linewidth=0.5, zorder=-1)
        if label:
            kw["label"] = label
        ax.axvline(x=wl, **kw)

    leg = ax.legend(loc="upper right", fontsize=12, handlelength=1.5, ncols=2)
    for h in leg.legend_handles:
        h.set_linewidth(1.0)

    ax.set_xlabel(
        fr"Rest Wavelength [\AA] ($\rm{{z}} = {redshift}$)", fontsize=18)
    ax.set_ylabel(
        r"Flux \ [$10^{-17}$erg/cm$^2$/s/\AA]", fontsize=18)
    ax.set_xticks(np.arange(1500, 10000, 500))
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlim(xlim)
    ax.set_title(title.replace("_", r"\_"))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(slugs=None):
    targets = [followup.get_target(sl) for sl in slugs] if slugs else followup.iter_targets()
    for t in targets:
        for spec in t.spectra:
            if t.redshift is None:
                print(f"  {t.slug}: no redshift in target.yaml, skipping {spec['file'].name}")
                continue
            plot_spectrum(spec["file"], t.slug, t.redshift, spec["plot"])
            print(f"  {t.slug}/{spec['plot'].relative_to(t.path)}")


if __name__ == "__main__":
    main(sys.argv[1:])
