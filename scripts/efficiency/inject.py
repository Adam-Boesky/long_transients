"""Multi-band PSF injection into ZTF reference images.

Injects fake point sources at the same pixel coordinates into g, r, and i
reference images so that downstream efficiency tests can follow a source
through the full multi-band filter pipeline.
"""

import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom

import sys
sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import img_ab_mag_to_flux

BANDS = ['g', 'r', 'i']
_IMG_TEMPLATE = '{img_dir}/ztf_{field}_z{band}_c{ccdid}_q{quadrant}_refimg.fits'

# EPSFBuilder default oversampling factor — the saved .npy arrays are at this
# resolution relative to the native image pixel scale.
EPSF_OVERSAMPLING = 4


def inject_multiband(
    field: int,
    ccdid: int,
    quadrant: int,
    outdir: str,
    img_dir: str,
    n_injected: int = 100,
    injection_mags: np.ndarray = np.arange(16, 25, 0.5),
) -> tuple:
    """Inject fake PSF sources into all available bands at identical pixel coords.

    For each magnitude in injection_mags, a FITS file is written per band with
    n_injected fake sources added at the same (row, col) positions.  Original
    images are saved as orig_{band}.fits so the caller can diff against them.

    Args:
        field: ZTF field number (e.g. 582).
        ccdid: CCD number (e.g. 2).
        quadrant: Quadrant number 1-4.
        outdir: Directory that holds EPSFs, WCSs, nan_masks, and PSTARR.ecsv
            for this quadrant.  Injected FITS files are written here too.
        img_dir: Directory containing the raw ZTF reference images.
        n_injected: Number of sources to inject per magnitude bin.
        injection_mags: Array of AB magnitudes to inject at.

    Returns:
        coords: np.ndarray of shape (n_injected, 2) — each row is
            (row_index, col_index) in numpy / image convention.
        mag_to_band_to_fpath: nested dict {float(mag): {band: filepath}}.
    """
    band_data = {}
    for band in BANDS:
        img_fpath = _IMG_TEMPLATE.format(
            img_dir=img_dir,
            field=str(field).zfill(6),
            band=band,
            ccdid=str(ccdid).zfill(2),
            quadrant=quadrant,
        )
        epsf_fpath = os.path.join(outdir, f'EPSFs/ZTF_{band}_EPSF.npy')
        if not (os.path.exists(img_fpath) and os.path.exists(epsf_fpath)):
            continue
        hdul = fits.open(img_fpath)
        img = hdul[0].data.byteswap().newbyteorder()
        # Downsample from oversampled EPSF space to native image pixel scale,
        # then normalise so the PSF sums to 1 (flux-conserving injection).
        epsf_oversampled = np.load(epsf_fpath)
        epsf = zoom(epsf_oversampled, 1.0 / EPSF_OVERSAMPLING, order=3)
        epsf /= epsf.sum()
        psf_hw = epsf.shape[0] // 2
        band_data[band] = {'img': img, 'epsf': epsf, 'header': hdul[0].header, 'hw': psf_hw}

    if not band_data:
        raise RuntimeError(
            f'No reference images / EPSFs found for field {field} ccd{ccdid} q{quadrant}'
        )

    # Draw injection coordinates safe for all band PSF half-widths
    margin = max(d['hw'] for d in band_data.values()) + 2
    ref_img = next(iter(band_data.values()))['img']
    xs = np.random.randint(low=margin, high=ref_img.shape[0] - margin, size=n_injected)
    ys = np.random.randint(low=margin, high=ref_img.shape[1] - margin, size=n_injected)
    coords = np.vstack((xs, ys)).T  # shape (n_injected, 2): (row, col)

    np.save(os.path.join(outdir, 'coords.npy'), coords)

    # Save original images so callers can use them as the zero-injection baseline
    for band, data in band_data.items():
        fits.writeto(
            os.path.join(outdir, f'orig_{band}.fits'),
            data=data['img'], header=data['header'], overwrite=True,
        )

    mag_to_band_to_fpath = {}
    for mag in injection_mags:
        mag_key = float(mag)
        mag_to_band_to_fpath[mag_key] = {}
        tag = str(mag).replace('.', '_')

        for band, data in band_data.items():
            img, epsf, header, hw = data['img'], data['epsf'], data['header'], data['hw']
            flux = img_ab_mag_to_flux(mag, zero_point=header['MAGZP'])
            img_inj = img.copy()
            for x, y in zip(xs, ys):
                img_inj[x - hw: x + hw + 1, y - hw: y + hw + 1] += epsf * flux

            fpath = os.path.join(outdir, f'mag{tag}_{band}.fits')
            fits.writeto(fpath, data=img_inj, header=header, overwrite=True)
            mag_to_band_to_fpath[mag_key][band] = fpath

    return coords, mag_to_band_to_fpath
