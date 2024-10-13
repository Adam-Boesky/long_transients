import os
import pathlib
from typing import Optional, Tuple, Union

import numpy as np


def get_credentials(fname: str) -> Union[Tuple[str, str], str]:
    """Retrieves credentials from a specified file in my ~/vault/directory."""
    key_location = os.path.join(pathlib.Path.home(), f'vault/{fname}')
    return np.genfromtxt(key_location, dtype = 'str')


def img_flux_to_ab_mag(flux: np.ndarray, zero_point: float, fluxerr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert image flux to AB magnitude.

    Parameters:
    flux (np.ndarray): The flux values of the image.
    zero_point (float): The zero point for the magnitude calculation.
    fluxerr (Optional[np.ndarray]): The flux error values of the image, by default None.

    Returns:
        1. The AB magnitudes corresponding to the input fluxes.
        2. If `fluxerr` is provided, the AB magnitude error.
    """
    mag = -2.5 * np.log10(flux) + zero_point
    if fluxerr is not None:
        magerr = 2.5 * fluxerr / (np.log(10) * flux)
        return mag, magerr
    return mag


def img_ab_mag_to_flux(mag: np.ndarray, zero_point: np.ndarray, magerr: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of img_flux_to_ab_mag."""
    flux = 10 ** ((mag - zero_point) / -2.5)
    if magerr is not None:
        fluxerr = (magerr * (np.log(10) * flux)) / 2.5
        return flux, fluxerr
    return flux

def get_snr_from_mag(mag: np.ndarray, magerr: np.ndarray, zp: float) -> np.ndarray:
        flux, fluxerr = img_ab_mag_to_flux(mag, zero_point=zp, magerr=magerr)
        return flux / fluxerr

def on_cluster() -> bool:
    """Check if the code is running on the cluster."""
    return os.path.exists('/n/home04')

def get_data_path() -> str:
    """Get the path to the data directory."""
    if on_cluster():
        return '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients'
    return '/Users/adamboesky/Research/long_transients/Data'
