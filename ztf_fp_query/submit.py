import json
import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup
from typing import Iterable

from Extracting.utils import get_credentials
from ztf_fp_query.constants import SUBMIT_URL, QUERY_URL, HTTP_AUTH, ZTF_LOGIN_FNAME


def _parse_jobs_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    headers = [th.text.strip() for th in table.find_all('th')]
    rows = [[td.text.strip() for td in row.find_all('td')] for row in table.find_all('tr')[1:]]
    return pd.DataFrame(rows, columns=headers)


_JOBS_COLUMNS = ['reqId', 'ra', 'dec', 'startJD', 'endJD', 'created', 'started', 'ended', 'machine', 'exitcode', 'lightcurve']


def _query_database(option: str) -> pd.DataFrame:
    email, pword = get_credentials(ZTF_LOGIN_FNAME)
    r = requests.get(
        QUERY_URL,
        auth=HTTP_AUTH,
        params={'email': email, 'userpass': pword, 'option': option, 'action': 'Query Database'},
    )
    # The service returns 400 with a "Zero records returned" body when there are no results
    if r.status_code == 400 and 'Zero records returned' in r.text:
        return pd.DataFrame(columns=_JOBS_COLUMNS)
    r.raise_for_status()
    return _parse_jobs_html(r.text)


def submit(ras: Iterable[float], decs: Iterable[float]) -> None:
    """Submit a batch forced-photometry request to the ZTF service."""
    email, pword = get_credentials(ZTF_LOGIN_FNAME)

    if isinstance(ras, float): ras = [ras]
    if isinstance(decs, float): decs = [decs]
    ras = [float('%.7f' % ra) for ra in ras]
    decs = [float('%.7f' % dec) for dec in decs]

    r = requests.post(
        SUBMIT_URL,
        auth=HTTP_AUTH,
        data={'ra': json.dumps(ras), 'dec': json.dumps(decs), 'email': email, 'userpass': pword},
    )
    r.raise_for_status()
    print(r.text.strip())
    return r.text.strip()


def query_recent_jobs() -> pd.DataFrame:
    """Return a DataFrame of all recent jobs for the authenticated user."""
    return _query_database('All recent jobs')


def query_pending_jobs() -> pd.DataFrame:
    """Return a DataFrame of pending jobs for the authenticated user (empty if none)."""
    return _query_database('Pending jobs')


def recently_queried(
    ras: Iterable[float],
    decs: Iterable[float],
    tol_deg: float = 1e-6,
) -> np.ndarray:
    """Boolean array: which (ra, dec) pairs appear in recent jobs."""
    if not isinstance(ras, Iterable): ras = [ras]
    if not isinstance(decs, Iterable): decs = [decs]
    recent_df = query_recent_jobs()
    return np.array([
        np.any(
            np.isclose(ra, recent_df['ra'].astype(float), atol=tol_deg) &
            np.isclose(dec, recent_df['dec'].astype(float), atol=tol_deg)
        )
        for ra, dec in zip(ras, decs)
    ])


def currently_pending(
    ras: Iterable[float],
    decs: Iterable[float],
    tol_deg: float = 1e-6,
) -> np.ndarray:
    """Boolean array: which (ra, dec) pairs are still pending."""
    if not isinstance(ras, Iterable): ras = [ras]
    if not isinstance(decs, Iterable): decs = [decs]
    pending_df = query_pending_jobs()
    return np.array([
        np.any(
            np.isclose(ra, pending_df['ra'].astype(float), atol=tol_deg) &
            np.isclose(dec, pending_df['dec'].astype(float), atol=tol_deg)
        )
        for ra, dec in zip(ras, decs)
    ])
