# Plan: Conditional Light Curve in `Source.plot_lc()`

## Overview

Add `'ztforce'` and `'zubercal'` as **non-default catalog options** to `Light_Curve`. Because `plot_lc()` already plots every `*_mag` column it finds in `self.light_curve.lc`, no changes to `Sources.py` or `plot_lc()` are needed at all. The conditionality lives at the call site in `store_src_plots.py`, which already knows catalog membership from the filename.

- **`in_ztf` sources** → include `'ztforce'` in `lc_catalogs`; handler fetches the **365-day rolling stack** and stores one row per stack window
- **`in_both` sources** → include `'zubercal'` in `lc_catalogs`; handler fetches single-epoch Zubercal photometry via LSDB

---

## 1. Background

### How `Light_Curve` works

`Light_Curve.get_catalog_lc(catalog)` is a big `if/elif` dispatch — each branch fetches, processes, and normalizes data into a common column schema. `get_lc()` calls all catalogs in parallel and vstacks the results. `plot_lc()` then plots any column whose name ends in `_mag`. Adding a new catalog requires only:
- A new `elif` branch in `get_catalog_lc()`
- New entries in `LC_MARKER_INFO` and `ALL_BAND_DF` so the right marker/color is used

### ZTF Science Forced Photometry (`ztforce`) — for `in_ztf`

`ztforce` is already used by `scripts/generate_science_lcs.py`. Key API:

```python
from ztforce import run_forced_photometry_batch, build_config
from ztforce.lightcurve import Lightcurve

config = build_config()
all_lcs = run_forced_photometry_batch(
    targets=[SkyCoord(ra=ra, dec=dec, unit='deg')],
    bands=('g', 'r', 'i'),
    data_dir=OUT_DIR,   # caches on disk — fast on repeat calls
    config=config,
)
lcs: dict[str, Lightcurve] = all_lcs[0]
rolled = lcs['g'].rolling_stack(window=365, window_unit='days')
# rolled columns: obsjd_center, band, mag_stack, mag_err_stack
```

The cache directory `Data/followup/ztforce_photometry/` is already on disk from `generate_science_lcs.py`. Sources already processed will return instantly.

**Output schema for `Light_Curve`:** one row per stack window center per band, with columns `mjd`, `ztforce_g_mag`, `ztforce_g_magerr`, `ztforce_r_mag`, etc.

> **Open question — time unit:** Confirm whether `rolling_stack()` returns `obsjd_center` in JD or MJD before implementing. If JD, subtract 2,400,000.5 to convert to MJD.

### Zubercal — for `in_both`

**What it is:** Zubercal (ZTF-ubercal) is IPAC's recalibration of ZTF PSF photometry to ~1% accuracy, cross-matched to PanSTARRS DR1 sources within 1.2 arcsec.

**Access — LSDB cone search:**

```python
import lsdb

ZUBERCAL_HATS = "https://data.lsdb.io/hats/ztf_dr16/zubercal/"
cat = lsdb.open_catalog(ZUBERCAL_HATS, columns=["mjd", "band", "mag", "magerr"])
df = cat.cone_search(ra, dec, radius_deg).compute()
df["magerr"] = df["magerr"] / 10_000.0   # stored as 1e-4 mag units
```

**Columns:** `mjd`, `band` (g/r/i), `mag`, `magerr` (÷10,000).

**Output schema for `Light_Curve`:** one row per epoch per band, with columns `mjd`, `zubercal_g_mag`, `zubercal_g_magerr`, etc.

**Dependency status (checked 2026-06-10):**
- `lsdb` is **not installed** in the `long_transients` conda env → `conda run -n long_transients pip install lsdb`
- `astroquery.ipac.irsa` is installed but Zubercal is **not** queryable via IRSA TAP — LSDB is the only programmatic access path

---

## Resolved Open Questions

**1. ztforce time unit — confirmed JD.**
`obsjd` is read directly from the ZTF FITS header keyword `OBSJD`, which is Julian Date. `rolling_stack()` returns `obsjd_center` also in JD. Subtract `2_400_000.5` when storing to the Light_Curve `mjd` column (which is MJD throughout).

**2. lsdb installation — working, with caveats.**
Full dependency chain installed (numba via conda-forge, then dask-expr, nested-pandas, hats, deprecated, etc.). One persistent issue: `lsdb.open_catalog()` tries to download a **50 MB `point_map.fits`** (the MOC coverage map) on every cold open, and the default aiohttp timeout kills it before the download completes. Fix: patch `hats.io.file_io.file_io.read_fits_image` to return `None` for `point_map.fits` paths, skipping the download. This is a 3-line addition at the top of `get_catalog_lc('zubercal')`. Without the MOC, cone searches still work correctly (the MOC is only used for query planning optimisations).

**3. Zubercal HATS URL and columns — confirmed.**
- Correct URL: `https://data.lsdb.io/hats/ztf_dr16/zubercal/` (DR16 is current; DR20 path does not exist)
- Confirmed columns: `mjd`, `mag`, `magerr`, `band`, `objra`, `objdec`, `objectid`
- **`magerr` confirmed in units of 1e-4 mag** (observed values ~1000–3000 → 0.1–0.3 mag after dividing by 10,000)
- **`mjd` column is already MJD** (observed values ~58000–60000, consistent with ZTF 2018-present) — no conversion needed
- Cone search takes ~15–20 s on first call per partition (parquet downloaded from S3); subsequent calls are faster once fsspec caches the files

---

## 2. Changes Required

### A. `Source_Analysis/Light_Curve.py`

**1. `LC_MARKER_INFO`** — add two entries:
```python
LC_MARKER_INFO = {
    ...
    'ztforce': 's',       # square markers
    'zubercal': 'D',      # diamond markers
}
```

**2. `ALL_BAND_DF`** — add per-band entries for each new catalog:
```python
ALL_BAND_DF = pd.DataFrame(
    data={
        ...
        'ztforce_g_mag':   ['g', 'ztforce'],
        'ztforce_r_mag':   ['r', 'ztforce'],
        'ztforce_i_mag':   ['i', 'ztforce'],
        'zubercal_g_mag':  ['g', 'zubercal'],
        'zubercal_r_mag':  ['r', 'zubercal'],
        'zubercal_i_mag':  ['i', 'zubercal'],
    },
    index=['band', 'survey'],
)
```

**3. `get_catalog_lc()`** — two new `elif` branches:

```python
elif catalog == 'ztforce':
    from ztforce import run_forced_photometry_batch, build_config
    from astropy.coordinates import SkyCoord as SC
    OUT_DIR = Path(__file__).resolve().parents[1] / 'Data' / 'followup' / 'ztforce_photometry'
    config = build_config()
    all_lcs = run_forced_photometry_batch(
        targets=[SC(ra=self.ra, dec=self.dec, unit='deg')],
        bands=('g', 'r', 'i'),
        data_dir=OUT_DIR,
        config=config,
    )
    lcs = all_lcs[0]
    rows = []
    for band, lc in lcs.items():
        rolled = lc.rolling_stack(window=365, window_unit='days')
        for _, row in rolled.iterrows():
            rows.append({
                'mjd': row['obsjd_center'] - 2_400_000.5,  # JD → MJD (confirm unit first)
                f'ztforce_{band}_mag': row['mag_stack'],
                f'ztforce_{band}_magerr': row['mag_err_stack'],
            })
    lightcurve_tab = Table.from_pandas(pd.DataFrame(rows)) if rows else None

elif catalog == 'zubercal':
    import lsdb
    import hats.io.file_io.file_io as _hats_fio
    # The HATS point_map.fits is 50 MB and times out on cold open; skip it.
    # The MOC is only used for query planning — cone searches work without it.
    _orig_fits = _hats_fio.read_fits_image
    _hats_fio.read_fits_image = lambda p, *a, **kw: (
        None if 'point_map' in str(p) else _orig_fits(p, *a, **kw)
    )
    try:
        ZUBERCAL_HATS = "https://data.lsdb.io/hats/ztf_dr16/zubercal/"
        cat = lsdb.open_catalog(
            ZUBERCAL_HATS,
            columns=["mjd", "band", "mag", "magerr", "objectid"],
        )
        df = cat.cone_search(self.ra, self.dec, self.query_rad_arcsec / 3600.0).compute()
    finally:
        _hats_fio.read_fits_image = _orig_fits  # always restore

    if df.empty:
        lightcurve_tab = None
    else:
        df = df.copy()
        df["magerr"] = df["magerr"] / 10_000.0   # stored as 1e-4 mag
        rows = []
        for _, row in df.iterrows():
            b = row['band']
            rows.append({
                'mjd': row['mjd'],   # already MJD — no conversion needed
                f'zubercal_{b}_mag': row['mag'],
                f'zubercal_{b}_magerr': row['magerr'],
            })
        lightcurve_tab = Table.from_pandas(pd.DataFrame(rows))
```

Both branches need `column_rename_map = []` and should skip the generic IRSA query path. The handlers return `None` on failure, which `get_lc()` already handles gracefully.

Also add `from pathlib import Path` at the top of `Light_Curve.py` if not already present.

---

### B. `Source_Analysis/store_src_plots.py`

Pass the appropriate `lc_catalogs` at each `Sources.from_file()` call. The default catalog list is already defined in `Light_Curve.__init__`; we extend it per catalog type:

```python
DEFAULT_LC_CATALOGS = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom']

# in_both — currently line 138
srcs = sort_by_dmag(Sources.from_file(
    os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/0.ecsv'),
    lc_catalogs=DEFAULT_LC_CATALOGS + ['zubercal'],
    **src_kwargs,
))

# in_ztf — currently line 159
srcs_ztf = sort_by_dmag(Sources.from_file(
    os.path.join(path_to_data, f'{FILTER_RESULTS_DIRNAME}/combined/1.ecsv'),
    lc_catalogs=DEFAULT_LC_CATALOGS + ['ztforce'],
    **src_kwargs,
))
```

Wide-association sources (`1_wide_association.ecsv`) are in_ztf, so they also get `+ ['ztforce']`.

---

## 3. Files Changed

| File | What changes |
|------|-------------|
| `Source_Analysis/Light_Curve.py` | Two new `elif` branches in `get_catalog_lc()`; two entries in `LC_MARKER_INFO`; six entries in `ALL_BAND_DF`; `from pathlib import Path` |
| `Source_Analysis/store_src_plots.py` | `lc_catalogs=` kwarg added to each `Sources.from_file()` call (~4 lines) |
| `Sources.py` | **No changes** |
| `plot_lc()` | **No changes** |

---

## 4. Validation Plan

1. **ztforce handler returns expected shape:**
   ```python
   lc = Light_Curve(ra, dec, catalogs=['ztforce'])
   tab = lc.get_catalog_lc('ztforce')
   assert 'ztforce_g_mag' in tab.colnames
   assert 'mjd' in tab.colnames
   ```

2. **Zubercal handler returns expected shape** (requires lsdb):
   ```python
   lc = Light_Curve(ra, dec, catalogs=['zubercal'])
   tab = lc.get_catalog_lc('zubercal')
   assert 'zubercal_g_mag' in tab.colnames
   ```

3. **`plot_lc()` renders the new data** — call `src.plot_lc()` for one in_ztf and one in_both source loaded via `store_src_plots.py`; confirm new markers appear in the legend.

4. **Graceful fallback** — source with no ztforce cache or no Zubercal match renders without error (empty table returned, skipped by `get_lc()`).

5. **`plot_everything()` unchanged** — confirm layout is not affected.

---

## 5. Open Questions

All three pre-implementation questions are resolved — see **Resolved Open Questions** above. No blockers remain before coding.
