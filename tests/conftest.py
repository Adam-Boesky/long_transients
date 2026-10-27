"""Fixtures for the serving-tier and API tests.

The databases are built here from synthetic data rather than pointed at
`Data/db/`, for three reasons: the tests then run anywhere without a 17-minute
build, they exercise the real build code, and they cannot be broken by a
half-finished fetch.

Scoping follows the read/write split of the tiers themselves:

    science_db      session -- static, read-only, expensive to build once
    annotations_db  function -- mutable, so every test gets a clean one

Sharing the writable database would make tests order-dependent, which is the
classic way an API suite starts passing locally and failing in CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The app's lifespan would otherwise start the ztforce worker against the real
# databases; tests drive the queue directly.
os.environ.setdefault('LT_ZTFORCE_WORKER', '0')
# ...and never read the developer's .env: tests must not depend on local settings.
os.environ.setdefault('LT_ENV_FILE', '')

from api import catalog_filters as api_filters  # noqa: E402
from api import db as api_db  # noqa: E402
from api import ztforce_store  # noqa: E402
from api.main import app  # noqa: E402
from source_db.lightcurves import _base  # noqa: E402
from source_db.serving import build_serving_db  # noqa: E402
from source_db.spectra import _base as spec_base  # noqa: E402

# Deterministic uuids spread across bucket prefixes, so the bucketing in
# `_base.BucketedWriter` is actually exercised rather than all landing in one
# file. The last one deliberately has no light curves.
UUIDS = [
    '00a1b2c3-0000-4000-8000-000000000001',
    '3fd4e5f6-0000-4000-8000-000000000002',
    '7a889900-0000-4000-8000-000000000003',
    'c1d2e3f4-0000-4000-8000-000000000004',
    'ff000000-0000-4000-8000-000000000005',   # no epochs
]
NO_LC_UUID = UUIDS[-1]
LC_UUIDS = UUIDS[:-1]

# Spectra are sparser than light curves (~34% of real sources), so only some
# of these carry one -- the fixture mirrors that rather than giving every
# source a spectrum.
SPEC_UUIDS = [UUIDS[0], UUIDS[1]]
# UUIDS[0]'s SDSS match; the serving build only publishes a source's current one.
SDSS_SPECOBJID = '1389259429339162624'
NO_SPEC_UUID = UUIDS[2]

# (survey, band, n_epochs) per source with data.
_PLAN = {
    UUIDS[0]: [('ztf', 'g', 5), ('ztf', 'r', 4), ('panstarrs', 'i', 3)],
    UUIDS[1]: [('ztf', 'g', 6), ('wise', 'w1', 2)],
    UUIDS[2]: [('sdss', 'u', 1), ('sdss', 'g', 1)],
    UUIDS[3]: [('custom', 'r', 2)],
}


def _lightcurve_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for uuid, spec in _PLAN.items():
        for survey, band, n in spec:
            for i in range(n):
                row = {
                    'source_uuid': uuid, 'survey': survey, 'band': band,
                    'mjd': 58000.0 + i, 'mag': 20.0 + rng.normal(0, 0.1),
                    'magerr': 0.05, 'snr': np.nan,
                    'catalog_id': f'{survey}-{uuid[:4]}',
                }
                if survey == 'ztf':
                    row['catflags'] = 0 if i else 6141   # one flagged epoch
                    row['clrcoeff'] = 0.1
                if survey == 'panstarrs':
                    row['infoFlag'] = 8
                    row['infoFlag2'] = 0
                if survey == 'sdss':
                    row['psf_mag'] = 20.5
                    row['psf_magerr'] = 0.06
                if survey == 'custom':
                    row['is_upper_limit'] = bool(i)
                    row['telescope'] = 'FLWO'
                    row['instrument'] = 'KeplerCam'
                rows.append(row)
    return pd.DataFrame(rows)


def _spectra_frames() -> dict[str, pd.DataFrame]:
    """One SDSS spectrum and one DESI spectrum, on realistic grids."""
    rng = np.random.default_rng(1)
    sdss_wave = np.linspace(3600.0, 9200.0, 40)
    desi_wave = np.arange(3600.0, 3600.0 + 0.8 * 50, 0.8)
    sdss = pd.DataFrame([spec_base.spectrum_row(
        source_uuid=UUIDS[0], survey='sdss', spec_id=SDSS_SPECOBJID,
        wavelength=sdss_wave, flux=rng.normal(10, 1, sdss_wave.size),
        model=rng.normal(10, 0.2, sdss_wave.size),
        spec_class='QSO', redshift=1.23)])
    desi = pd.DataFrame([
        spec_base.spectrum_row(
            source_uuid=UUIDS[0], survey='desi', spec_id='39627908108519236',
            wavelength=desi_wave, flux=rng.normal(8, 1, desi_wave.size),
            model=rng.normal(8, 0.2, desi_wave.size),
            spec_class='QSO', redshift=1.24),
        spec_base.spectrum_row(
            source_uuid=UUIDS[1], survey='desi', spec_id='39627884012243433',
            wavelength=desi_wave, flux=rng.normal(5, 1, desi_wave.size),
            model=None, spec_class='GALAXY', redshift=0.44),
    ])
    return {'sdss_spectra': sdss, 'desi_spectra': desi}


@pytest.fixture(scope='session')
def built_db(tmp_path_factory, monkeypatch_session) -> Path:
    """A real science.sqlite, built by the real build code from synthetic input."""
    out = tmp_path_factory.mktemp('db')

    pd.DataFrame({
        'source_uuid': UUIDS,
        'coord_slug': [f'slug{i}' for i in range(len(UUIDS))],
        'ra': [10.0, 20.0, 30.0, 40.0, 50.0],
        'dec': [-5.0, 0.0, 5.0, 10.0, 15.0],
        'PSTARR_PanSTARR_ID': [1, 2, 3, 4, 5],
        'some_float': [1.5, 2.5, 3.5, 4.5, 5.5],
        'sdss_matched': [True, False, False, False, False],
        'sdss_specobjid': [SDSS_SPECOBJID, None, None, None, None],
    }).to_parquet(out / 'sources.parquet', index=False)

    with _base.BucketedWriter(out) as w:
        w.write(_lightcurve_frame())

    # The serving build reads spectra from RAW_DIR, so point that at the
    # temporary tree for the duration of the session.
    raw = out / 'raw'
    for catalog, df in _spectra_frames().items():
        spec_base.write_raw(catalog, '0000', df, raw_dir=raw)
    monkeypatch_session.setattr(build_serving_db, 'RAW_DIR', raw)
    # The real lookup re-lists SDSS spectra from the local catalog.
    monkeypatch_session.setattr(build_serving_db, 'current_sdss_keys',
                                lambda out_dir: {f'{UUIDS[0]}|{SDSS_SPECOBJID}'})

    return build_serving_db.build(out_dir=out)


@pytest.fixture(scope='session')
def monkeypatch_session():
    """`monkeypatch` is function-scoped; `built_db` is not."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope='session')
def science_con(built_db):
    con = api_db.connect_science(built_db)
    yield con
    con.close()


@pytest.fixture
def annotations_con(tmp_path):
    """A fresh annotations database per test -- no cross-test pollution."""
    con = api_db.connect_annotations(tmp_path / 'annotations.sqlite')
    yield con
    con.close()


@pytest.fixture
def ztforce_con(tmp_path):
    """A fresh ztforce queue per test."""
    con = ztforce_store.connect(tmp_path / 'ztforce.sqlite')
    yield con
    con.close()


@pytest.fixture(scope='session')
def filter_engine(built_db, science_con):
    """The catalog filter engine, over the same synthetic build.

    The list endpoints filter and paginate in this engine, so it has to
    describe the same sources as `science.sqlite` -- left at its default it
    would read the real 198k-row catalog and agree with the fixture about
    nothing.
    """
    engine = api_filters.startup(science_con, built_db.parent / 'sources.parquet')
    yield engine
    api_filters.shutdown()


@pytest.fixture
def client(science_con, annotations_con, filter_engine, ztforce_con):
    """TestClient with all three tiers overridden.

    `dependency_overrides` is the supported seam for this; it avoids having to
    set environment variables before `api.config` is imported.
    """
    from fastapi.testclient import TestClient

    app.dependency_overrides[api_db.get_science] = lambda: science_con
    app.dependency_overrides[api_db.get_annotations] = lambda: annotations_con
    app.dependency_overrides[api_filters.get_filters] = lambda: filter_engine
    app.dependency_overrides[ztforce_store.get_ztforce_store] = lambda: ztforce_con
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
