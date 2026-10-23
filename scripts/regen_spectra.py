"""Regenerate all spectrum PDFs with canonical coord_stem filenames.

Coordinates sourced from:
  - FITS files: CATRA/CATDEC header keywords (precise)
  - LDSS3 ASCII files: raw observation filename (2 d.p., negative sign authoritative)
  - FLEET files: filtered .txt filename (no coords in header)
  - Manual: 21_166p83_19p366 (no raw file in notebook list)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Source_Analysis.coord_utils import coord_stem

SPECTRA_DIR = "Data/followup/spectra"
PLOTS_DIR   = "Data/followup/spectra/plots"
REDSHIFTS_CSV = "Data/followup/spectra/redshifts.csv"

# (ra, dec, redshift) for each processed .txt file.
# RA/Dec are authoritative values; coord_stem() encodes them at 4 d.p.
SOURCES = {
    "121p24_17p408.txt":     (121.2458,  17.4083, 2.21),   # FITS CATRA/CATDEC
    "126p27_26p706.txt":     (126.2748,  26.7060, 1.77),   # FITS CATRA/CATDEC
    "85_210p82_19p294.txt":  (210.8221,  19.2950, 0.57),   # FITS CATRA/CATDEC
    "38_222p85_21p040.txt":  (222.8579,  21.0405, 2.15),   # FITS CATRA/CATDEC
    "112_227p82_19p622.txt": (227.8208,  19.6224, 1.68),   # FITS CATRA/CATDEC
    "31_221p68_17p961.txt":  (221.6800,  17.9610, 1.70),   # FLEET (no coord header)
    "49_353p25_n21p86.txt":  (353.2500, -21.8600, 2.242),  # LDSS3 raw filename
    "55_4p5025_n17p62.txt":  (  4.5025, -17.6200, 1.367),  # LDSS3 raw filename
    "21_166p83_19p366.txt":  (166.8300,  19.3660, 1.21),   # from .txt filename
    "14_355p63_27p83.txt":   (355.6300, -27.8300, 1.60),   # LDSS3 raw: AT14355p63n27p83
    "6_352p55_25p80.txt":    (352.5500, -25.8000, 0.57),   # LDSS3 raw: AT6352p55n25p80
}


def _bin(wavelength, flux, binsize):
    n = len(wavelength) // binsize
    wl = np.mean(wavelength[: n * binsize].reshape(-1, binsize), axis=1)
    fl = np.mean(flux[: n * binsize].reshape(-1, binsize), axis=1)
    return wl, fl


def plot_spectrum(txt_name, ra, dec, redshift, out_path):
    plt.rc("text", usetex=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cmr10"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.formatter.use_mathtext"] = True

    data = np.loadtxt(os.path.join(SPECTRA_DIR, txt_name))
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

    stem = coord_stem(ra, dec)
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
    ax.set_title(stem)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rewrite_redshifts_csv():
    rows = ["filename_stem,redshift"]
    for txt_name, (ra, dec, z) in SOURCES.items():
        stem = coord_stem(ra, dec)
        rows.append(f"{stem},{z}")
    with open(REDSHIFTS_CSV, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Rewrote {REDSHIFTS_CSV}")


if __name__ == "__main__":
    # Remove all existing PDFs in plots/
    removed = 0
    for fname in os.listdir(PLOTS_DIR):
        if fname.endswith(".pdf"):
            os.remove(os.path.join(PLOTS_DIR, fname))
            removed += 1
    print(f"Removed {removed} old PDF(s)")

    # Regenerate
    for txt_name, (ra, dec, z) in SOURCES.items():
        stem = coord_stem(ra, dec)
        out = os.path.join(PLOTS_DIR, f"{stem}_spectrum.pdf")
        print(f"  {txt_name}  →  {stem}_spectrum.pdf")
        plot_and_save = plot_spectrum(txt_name, ra, dec, z, out)

    # Rewrite redshifts.csv with canonical stems
    rewrite_redshifts_csv()

    print("Done. New plots:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".pdf"):
            print(f"  {f}")
