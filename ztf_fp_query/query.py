import os
import re
import time
import json
import requests
import numpy as np
import pandas as pd

from requests import Response
from bs4 import BeautifulSoup

from typing import Iterable, Optional, Tuple, List

try:
    from Forced_Photo_Map import Forced_Photo_Map
except:
    from .Forced_Photo_Map import Forced_Photo_Map


class ZTFFP_Service():
    def __init__(self, email: Optional[str] = None, pword: Optional[str] = None):
        # Get credentials
        if email is None: email = os.getenv("ztf_email_address", None)
        if pword is None: pword = os.getenv("ztf_user_password", None)
        self._ztffp_email_address = email
        self._ztffp_user_password = pword
        self.fp_map = Forced_Photo_Map()

        # Make sure credentials exist
        if None in [self._ztffp_email_address, self._ztffp_user_password]:
            raise ValueError('Must set ztf_email_address and ztf_user_password environment variables.')

    def _submit_post(self, ra_list: list, dec_list: list):
        ra = json.dumps(ra_list)
        dec = json.dumps(dec_list)
        payload = {
            'ra': ra,
            'dec': dec, #'jdstart': jdstart, 'jdend': jdend,
            'email': self._ztffp_email_address,
            'userpass': self._ztffp_user_password
        }

        # Submit job
        r = requests.post(
            'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit',
            auth=('ztffps', 'dontgocrazy!'),
            data=payload
        )

        # Check status
        if r.status_code != 200:
            raise ValueError(f'Submission failed with status code {r.status_code}. Message:\n{r.content}.')

        return r.status_code

    def submit(self, ras: Iterable[float], decs: Iterable[float]) -> int:

        # Cast to lists if not already
        if isinstance(ras, float):
            ras = [ras]
        if isinstance(decs, float):
            decs = [decs]

        # Round stuff
        ras = [float('%.7f'%(ra)) for ra in ras]
        decs = [float('%.7f'%(dec)) for dec in decs]

        # Submit batch
        self._submit_post(ras, decs)

    def download_batch(
            self,
            outdir: str = '/Users/adamboesky/Research/long_transients/Data/ztf_forced_photometry',
        ) -> List[str]:
        # Get the wget commands from the monitor
        wget_commands = self.monitor()

        # Download files
        fpaths = []
        for wget_cmd in wget_commands:

            # Parse the filename from the wget command
            match = re.search(r'-O\s+(\S+)', wget_cmd)
            if not match:
                print("Failed to parse filename from wget command.")
                continue
            filename = match.group(1)
            filepath = os.path.join(outdir, filename)
            fpaths.append(filepath)

            # If it's already downloaded, just skip the iteration
            if os.path.exists(filepath):
                print(f'{filepath} already downloaded... skipping!')
                continue

            # Parse the URL from the wget command
            match = re.search(r'"(https://.+?)"', wget_cmd)
            if not match:
                print("Failed to parse URL from wget command.")
                continue
            url = match.group(1)

            # Perform the download
            response = requests.get(url, auth=('ztffps', 'dontgocrazy!'))
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {filepath}")
            else:
                print(f"Failed to download {url}. Status code: {response.status_code}, Reason: {response.reason}")

        # Update the map based on the new downloads
        self.fp_map.add_all_new_light_curves()

        return fpaths

    def convert_recent_jobs_to_df(self, html_content: str)-> pd.DataFrame:
        # Get table from BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')

        # Extract data
        headers = [header.text.strip() for header in table.find_all('th')]
        rows = []
        for row in table.find_all('tr')[1:]:  # Skip the header row
            cells = [cell.text.strip() for cell in row.find_all('td')]
            rows.append(cells)

        return pd.DataFrame(rows, columns=headers)
    
    def _query_database(self, option: str) -> Response:
        if option.lower() not in ['all recent jobs', 'pending jobs']:
            raise ValueError(f'option must be either \'All recent jobs\' or \'Pending jobs\' but is {option}')

        # Job info
        settings = {
            'email': self._ztffp_email_address,
            'userpass': self._ztffp_user_password,
            'option': option,
            'action': 'Query Database'
        }

        # Check recent jobs
        r = requests.get(
            'https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi',
            auth = ('ztffps', 'dontgocrazy!'),
            params=settings
        )

        return r

    def query_pending_jobs(self) -> Tuple[int, Optional[pd.DataFrame]]:
        # Submit query
        r = self._query_database('Pending jobs')

        if r.status_code == 200:
            return r.status_code, self.convert_recent_jobs_to_df(r.text)
        return r.status_code, None

    def query_recent_jobs(self) -> Tuple[int, Optional[pd.DataFrame]]:
        # Submit query
        r = self._query_database('All recent jobs')

        if r.status_code == 200:
            return r.status_code, self.convert_recent_jobs_to_df(r.text)
        return r.status_code, None

    def monitor(self, sleep_time: int = 10):
        # wget string info
        wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O '
        wget_url = 'https://ztfweb.ipac.caltech.edu'
        wget_suffix = '"'

        # Iteratively check jobs
        while True:

            # Check recent jobs
            status_code, recent_df = self.query_recent_jobs()
            if status_code == 200:
                print("Script executed normally and queried the ZTF Batch Forced Photometry database.")

                # Load the recent job table
                lightcurves = recent_df['lightcurve'].to_numpy()

                # Make the wget strings
                wget_strs = []
                if len(lightcurves) != 0:
                    for lc in lightcurves:
                        p = re.match(r'.+/(.+)', lc)
                        fileonly = p.group(1)
                        wget_strs.append(wget_prefix + " " + fileonly + " \"" + wget_url + lc + wget_suffix)
                    return wget_strs

            else:
                print("Status_code=", status_code, "; Jobs either queued or abnormal execution.")
                time.sleep(sleep_time)

    def recently_queried(self, ras: Iterable[float], decs: Iterable[float], tol_deg: float = 1E-6) -> bool:
        # Cast to iterable if not alreaday
        if not isinstance(ras, Iterable): ras = [ras]
        if not isinstance(decs, Iterable): decs = [decs]

        # Get the recent queries
        _, recent_df = self.query_recent_jobs()

        # Check if close
        isin = []
        for ra, dec in zip(ras, decs):

            # Check if ra and dec are within tolerance)
            ra_close = np.isclose(ra, recent_df['ra'].to_numpy(dtype=float), atol=tol_deg)
            dec_close = np.isclose(dec, recent_df['dec'].to_numpy(dtype=float), atol=tol_deg)
            isin.append(np.any(ra_close & dec_close))

        return np.array(isin)


    def currently_pending(self, ras: Iterable[float], decs: Iterable[float], tol_deg: float = 1E-6) -> bool:
        # Cast to iterable if not alreaday
        if not isinstance(ras, Iterable): ras = [ras]
        if not isinstance(decs, Iterable): decs = [decs]

        # Get the recent queries
        status_code, recent_df = self.query_pending_jobs()
        if status_code != 200:
            return np.zeros(len(ras), dtype=bool)

        # Check if close
        isin = []
        for ra, dec in zip(ras, decs):

            # Check if ra and dec are within tolerance
            ra_close = np.isclose(ra, recent_df['ra'], atol=tol_deg)
            dec_close = np.isclose(dec, recent_df['dec'], atol=tol_deg)
            isin.append(np.any(ra_close & dec_close))

        return np.array(isin)
