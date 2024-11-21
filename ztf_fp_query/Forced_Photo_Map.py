import os
import re
import sys
import linecache
import numpy as np
import pandas as pd
import astropy.units as u

from astropy.coordinates import SkyCoord, match_coordinates_sky
from typing import Tuple, Union, Iterable, List

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import get_data_path


class Forced_Photo_Map():
    """Class corresponding to the forced photometry map."""
    def __init__(self):

        # Get the existing map if it exists, else make a new one
        self.data_path = os.path.join(get_data_path(), 'ztf_forced_photometry')
        self.map_path = os.path.join(self.data_path, 'source_map.csv')
        if os.path.exists(self.map_path):
            map = pd.read_csv(self.map_path)
        else:
            map = pd.DataFrame(data={'ra': [], 'dec': [], 'fname': []})

        self.map: pd.DataFrame = map

    def update(self):
        """Update the file."""
        self.map.to_csv(self.map_path, index=False)

    def _get_ra_dec(self, fname: str) -> Tuple[float, float]:
        """Get the ra and dec of a given file"""
        # Known line numbers for R.A. and Dec.
        ra_line_number = 4      # Line number for R.A. (1-based indexing)
        dec_line_number = 5     # Line number for Dec. (1-based indexing)

        # Define regex patterns
        ra_pattern = r"Requested input R\.A\. = ([\d.]+) degrees"
        dec_pattern = r"Requested input Dec\. = ([\d.]+) degrees"

        # Get the specific lines
        fpath = os.path.join(self.data_path, fname)
        ra_line = linecache.getline(fpath, ra_line_number)
        dec_line = linecache.getline(fpath, dec_line_number)

        # Extract R.A. and Dec. using regex
        ra_match = re.search(ra_pattern, ra_line)
        dec_match = re.search(dec_pattern, dec_line)

        # Convert matches to float
        ra = float(ra_match.group(1)) if ra_match else None
        dec = float(dec_match.group(1)) if dec_match else None

        return ra, dec

    def add_light_curve(self, lc_fname: str, update: bool = True):
        """Add a lightcurve to the map given its filename."""
        # Get the light curve ra and dec, and add to map
        ra, dec = self._get_ra_dec(lc_fname)
        new_row = pd.DataFrame([{'ra': ra, 'dec': dec, 'fname': lc_fname}])
        self.map = pd.concat([self.map, new_row], ignore_index=True)

        # Update the CSV file
        if update:
            self.update()

    def add_all_new_light_curves(self):
        """Add all new lightcurves to the map."""
        # Iterate through all files in the data path
        for fname in os.listdir(self.data_path):

            # Add file to map if its not already there and file ends with '.txt'
            if not fname in self.map['fname'].to_numpy() and fname.endswith('.txt'):
                self.add_light_curve(fname, update=False)

        # Update the CSV file after adding all new light curves
        self.update()

    def get_lightcurve_fname(
            self,
            ras: Union[Iterable[float], float],
            decs: Union[Iterable[float], float],
            max_dist_arcsec: float = 0.5
        ) -> List[str]:
        """Get the lightcurve fname for a given set of ras and decs."""

        # Cast ra and dec to iterables if not already
        if not isinstance(ras, Iterable): ras = [ras]
        if not isinstance(decs, Iterable): decs = [decs]

        # Get coordinates for the files
        file_coords = SkyCoord(self.map['ra'], self.map['dec'], unit='deg')

        # Get fnames for all given coordinates
        fnames = []
        for ra, dec in zip(ras, decs):

            # Make a coordinate
            coord = SkyCoord(ra, dec, unit='deg')

            # Add the closest coordinate under the threshold to the list
            _, sep2d, _ = match_coordinates_sky(coord, file_coords)
            if np.any(sep2d.arcsecond < max_dist_arcsec):
                fnames.append(self.map['fname'][np.argmin(sep2d)])

        return fnames

    def contains(self, ras: Iterable[float], decs: Iterable[float], tol_deg: float = 1E-6) -> bool:
        # Cast to iterable if not alreaday
        if not isinstance(ras, Iterable): ras = [ras]
        if not isinstance(decs, Iterable): decs = [decs]

        # Check if close
        isin = []
        for ra, dec in zip(ras, decs):

            # Check if ra and dec are within tolerance
            ra_close = np.isclose(ra, self.map['ra'], atol=tol_deg)
            dec_close = np.isclose(dec, self.map['dec'], atol=tol_deg)
            isin.append(np.any(ra_close & dec_close))

        return np.array(isin)
