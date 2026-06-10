import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from typing import Dict, List, Optional, Union
from matplotlib.axes import Axes
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time

from ztf_fp_query.sync import get_data_dir, load_map

# Recommended SNT/SNU thresholds from https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
# SNT: signal-to-noise threshold below which an epoch is treated as a non-detection
# SNU: the multiple of flux_unc used to compute the upper-limit magnitude
_SNT = 3
_SNU = 5

_RA_PATTERN = re.compile(r'Requested input R\.A\. = ([\d.]+) degrees')
_DEC_PATTERN = re.compile(r'Requested input Dec\. = (-?[\d.]+) degrees')

BAND_COLORS: Dict[str, str] = {'ZTF_g': 'forestgreen', 'ZTF_r': 'lightcoral', 'ZTF_i': 'darkorchid'}


class ForcedPhotometry:
    """Container for a ZTF forced-photometry lightcurve from the IPAC service.

    Parses a ZFPS ASCII output file and exposes the data as a DataFrame with
    derived magnitude columns (mag, magerr, snr, upperlim) already added.

    Usage:
        # Load by sky position (requires a prior sync())
        fp = ForcedPhotometry(ra=210.08983, dec=-6.88676)

        # Or load directly from a file
        fp = ForcedPhotometry.from_file('path/to/batchfp_req0004838848_lc.txt')
    """

    def __init__(self, ra: float, dec: float, max_sep_arcsec: float = 0.5, data_dir: Optional[str] = None):
        self.ra = ra
        self.dec = dec

        data_dir = data_dir or get_data_dir()
        map_df = load_map()
        if map_df.empty:
            raise FileNotFoundError('No forced photometry files in the local index. Run sync() first.')

        map_coords = SkyCoord(map_df['ra'].values, map_df['dec'].values, unit='deg')
        target = SkyCoord(ra, dec, unit='deg')
        idx, sep2d, _ = match_coordinates_sky(target, map_coords)
        if sep2d.arcsecond > max_sep_arcsec:
            raise FileNotFoundError(
                f'No forced photometry within {max_sep_arcsec}" of ({ra}, {dec}). '
                'Submit a request and run sync() first.'
            )

        self._fpath = os.path.join(data_dir, map_df.iloc[idx]['fname'])
        self._df = self._parse_file(self._fpath)

    @classmethod
    def from_file(cls, fpath: str) -> 'ForcedPhotometry':
        """Load directly from a ZFPS ASCII file path."""
        obj = object.__new__(cls)
        obj._fpath = fpath
        obj.ra, obj.dec = cls._read_ra_dec(fpath)
        obj._df = cls._parse_file(fpath)
        return obj

    @staticmethod
    def _read_ra_dec(fpath: str) -> tuple[float, float]:
        ra = dec = None
        with open(fpath) as f:
            for line in f:
                if ra is None:
                    m = _RA_PATTERN.search(line)
                    if m: ra = float(m.group(1))
                if dec is None:
                    m = _DEC_PATTERN.search(line)
                    if m: dec = float(m.group(1))
                if ra is not None and dec is not None:
                    break
        return ra, dec

    @staticmethod
    def _parse_file(fpath: str) -> pd.DataFrame:
        """Parse a ZFPS ASCII lightcurve file and return a DataFrame with derived columns."""
        with open(fpath) as f:
            lines = f.readlines()

        # Non-comment, non-empty lines: first is the column header, rest are data
        non_comment = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        col_names = [c.strip() for c in non_comment[0].split(',')]
        df = pd.read_csv(
            StringIO(''.join(non_comment[1:])),
            sep=r'\s+',
            names=col_names,
            na_values=['null'],
            header=None,
        )

        # Derive mag, magerr, snr, upperlim
        # Non-detections: flux/flux_unc < SNT. Upper-limit mag = zpdiff - 2.5*log10(SNU * flux_unc).
        # Detections: mag = zpdiff - 2.5*log10(flux), magerr = (2.5/ln10) * flux_unc/flux.
        flux = df['forcediffimflux'].values
        flux_unc = df['forcediffimfluxunc'].values
        zpdiff = df['zpdiff'].values

        snr = flux / flux_unc
        upperlim = snr < _SNT

        mag = np.full(len(df), np.nan)
        magerr = np.full(len(df), np.nan)

        det = ~upperlim & (flux > 0)
        mag[det] = zpdiff[det] - 2.5 * np.log10(flux[det])
        magerr[det] = 1.0857 * flux_unc[det] / flux[det]

        ul = upperlim & (flux_unc > 0)
        mag[ul] = zpdiff[ul] - 2.5 * np.log10(_SNU * flux_unc[ul])

        df['snr'] = snr
        df['upperlim'] = upperlim.astype(int)
        df['mag'] = mag
        df['magerr'] = magerr

        return df

    @property
    def df(self) -> pd.DataFrame:
        """Full parsed DataFrame including derived columns (mag, magerr, snr, upperlim)."""
        return self._df

    @property
    def bands(self) -> List[str]:
        """Unique filters present in this lightcurve, in g/r/i order."""
        _order = ['ZTF_g', 'ZTF_r', 'ZTF_i']
        present = set(self._df['filter'].unique())
        return [b for b in _order if b in present]

    def get_band(self, band: str) -> pd.DataFrame:
        """Return rows for a single filter (e.g. 'ZTF_g', 'ZTF_r', 'ZTF_i')."""
        return self._df[self._df['filter'] == band].copy()

    def stack(
        self,
        jd_min: Optional[float] = None,
        jd_max: Optional[float] = None,
        bands: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Inverse-variance weighted mean flux per band over a time window.

        Computes the optimal stacked flux estimator (Zackay & Ofek 2015,
        arXiv:1512.06872):

            f_stack = Σ(f_i / σ_i²) / Σ(1 / σ_i²)
            σ_stack = 1 / sqrt(Σ(1 / σ_i²))

        Equivalent to image coaddition but operating directly on calibrated
        forcediffimflux values. Achieves √N SNR improvement in the
        photon-noise-dominated regime (Masci et al. 2023, arXiv:2305.16279).

        Only meaningful over windows where the source is non-variable (e.g.
        a quiescent baseline or pre-event non-detection window). Stacking
        a varying source biases the result.

        Args:
            jd_min: start of stacking window in JD (default: all epochs).
            jd_max: end of stacking window in JD (default: all epochs).
            bands: filters to stack (default: all present).

        Returns:
            DataFrame indexed by band with columns:
            flux, flux_err, snr, mag, mag_err, upperlim, n_epochs.
        """
        df = self._df.copy()
        if jd_min is not None:
            df = df[df['jd'] >= jd_min]
        if jd_max is not None:
            df = df[df['jd'] <= jd_max]

        rows = []
        for band in (bands or self.bands):
            lc = df[df['filter'] == band].dropna(subset=['forcediffimflux', 'forcediffimfluxunc'])
            if lc.empty:
                continue

            flux = lc['forcediffimflux'].values
            flux_unc = lc['forcediffimfluxunc'].values
            zpdiff = lc['zpdiff'].values

            weights = 1.0 / flux_unc ** 2
            f_stack = np.sum(flux * weights) / np.sum(weights)
            sigma_stack = 1.0 / np.sqrt(np.sum(weights))
            snr_stack = f_stack / sigma_stack
            zp_wmean = np.sum(zpdiff * weights) / np.sum(weights)

            is_upperlim = snr_stack < _SNT
            if not is_upperlim and f_stack > 0:
                mag_stack = zp_wmean - 2.5 * np.log10(f_stack)
                magerr_stack = 1.0857 * sigma_stack / f_stack
            else:
                mag_stack = zp_wmean - 2.5 * np.log10(_SNU * sigma_stack)
                magerr_stack = np.nan

            rows.append({
                'band': band,
                'flux': f_stack,
                'flux_err': sigma_stack,
                'snr': snr_stack,
                'mag': mag_stack,
                'mag_err': magerr_stack,
                'upperlim': int(is_upperlim),
                'n_epochs': len(lc),
            })

        return pd.DataFrame(rows).set_index('band')

    def rolling_stack(
        self,
        window: float,
        bands: Optional[List[str]] = None,
        window_unit: str = 'days',
        step: Optional[float] = None,
    ) -> pd.DataFrame:
        """Rolling inverse-variance weighted stack across the lightcurve.

        Slides a window across time (or image count) and computes a stacked
        flux measurement at each position, following the moving-average approach
        described in Masci et al. 2023 (arXiv:2305.16279, Section 6.6).

        Each stacked point uses the same estimator as stack():
            f_stack = Σ(f_i / σ_i²) / Σ(1 / σ_i²),  σ_stack = 1/√(Σ 1/σ_i²)

        Args:
            window: width of the rolling window in days or number of images,
                depending on window_unit.
            bands: filters to stack (default: all present).
            window_unit: 'days' (physically motivated, non-uniform sensitivity)
                or 'images' (uniform sensitivity, non-uniform time coverage).
            step: step size between window centres in the same unit as window.
                Defaults to window/2 (50% overlap).

        Returns:
            DataFrame with columns: jd_center, band, flux, flux_err, snr, mag,
            mag_err, upperlim, n_epochs.
        """
        if window_unit not in ('days', 'images'):
            raise ValueError("window_unit must be 'days' or 'images'")
        if step is None:
            step = window / 2

        df = self._df.dropna(subset=['forcediffimflux', 'forcediffimfluxunc']).copy()
        rows = []

        for band in (bands or self.bands):
            lc = df[df['filter'] == band].sort_values('jd').reset_index(drop=True)
            if lc.empty:
                continue

            if window_unit == 'days':
                jd_min, jd_max = lc['jd'].min(), lc['jd'].max()
                centers = np.arange(jd_min + window / 2, jd_max - window / 2 + step, step)
                for center in centers:
                    w = lc[(lc['jd'] >= center - window / 2) & (lc['jd'] < center + window / 2)]
                    if w.empty:
                        continue
                    rows.append(self._stack_window(w, band))

            else:  # images
                n = len(lc)
                half = int(window // 2)
                indices = np.arange(half, n - half, max(1, int(step)))
                for i in indices:
                    w = lc.iloc[max(0, i - half): i + half + 1]
                    center = lc.iloc[i]['jd']
                    rows.append(self._stack_window(w, band))

        return pd.DataFrame(rows)

    @staticmethod
    def _stack_window(lc: pd.DataFrame, band: str) -> dict:
        """Compute the inverse-variance weighted stack for a single window."""
        flux = lc['forcediffimflux'].values
        flux_unc = lc['forcediffimfluxunc'].values
        zpdiff = lc['zpdiff'].values
        jd = lc['jd'].values

        weights = 1.0 / flux_unc ** 2
        f_stack = np.sum(flux * weights) / np.sum(weights)
        sigma_stack = 1.0 / np.sqrt(np.sum(weights))
        snr_stack = f_stack / sigma_stack
        zp_wmean = np.sum(zpdiff * weights) / np.sum(weights)

        # Use the weighted-mean JD of the actual data rather than the grid centre,
        # so that each stacked point is plotted at the true temporal barycentre of
        # its constituent epochs.
        jd_wmean = np.sum(jd * weights) / np.sum(weights)

        is_upperlim = snr_stack < _SNT
        if not is_upperlim and f_stack > 0:
            mag_stack = zp_wmean - 2.5 * np.log10(f_stack)
            magerr_stack = 1.0857 * sigma_stack / f_stack
        else:
            mag_stack = zp_wmean - 2.5 * np.log10(_SNU * sigma_stack)
            magerr_stack = np.nan

        return {
            'jd_center': jd_wmean,
            'band': band,
            'flux': f_stack,
            'flux_err': sigma_stack,
            'snr': snr_stack,
            'mag': mag_stack,
            'mag_err': magerr_stack,
            'upperlim': int(is_upperlim),
            'n_epochs': len(lc),
        }

    def plot(
        self,
        bands: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        y_units: str = 'mag',
        colors: Dict[str, str] = BAND_COLORS,
        include_upper_lim: bool = True,
        time_offset: Union[str, float] = 'first',
        time_as_str: bool = True,
        xlab_kwargs: dict = None,
        **kwargs,
    ) -> Axes:
        """Plot the forced-photometry lightcurve.

        Args:
            bands: which filters to plot (default: all present).
            ax: existing Axes to plot onto.
            y_units: 'mag' or 'flux'.
            colors: mapping of filter name to matplotlib color.
            include_upper_lim: whether to draw upper limits as downward triangles.
            time_offset: 'first' to subtract the earliest JD, or a float JD value.
        """
        if y_units not in ('mag', 'flux'):
            raise ValueError(f"y_units must be 'mag' or 'flux', got '{y_units}'")

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        kwargs.setdefault('markersize', 5)
        kwargs.setdefault('capsize', 2)
        kwargs.setdefault('fmt', 'o')

        # Upper-limit scatter kwargs: strip errorbar-only keys, switch to downward triangle
        scatter_kwargs = {k: v for k, v in kwargs.items()
                         if k not in ('capsize', 'fmt')}
        scatter_kwargs['marker'] = 'v'
        scatter_kwargs['s'] = scatter_kwargs.pop('markersize', 5) * 4

        if xlab_kwargs is None:
            xlab_kwargs = {'rotation': 45, 'ha': 'right'}

        if time_as_str:
            time_offset = 0.0
        elif time_offset == 'first':
            time_offset = np.nanmin(self._df['jd'])

        y_key = 'mag' if y_units == 'mag' else 'forcediffimflux'
        yerr_key = 'magerr' if y_units == 'mag' else 'forcediffimfluxunc'

        for band in (bands or self.bands):
            lc = self.get_band(band)
            if lc.empty:
                continue

            not_ul = ~lc['upperlim'].astype(bool)
            color = colors.get(band)

            ax.errorbar(
                x=lc['jd'][not_ul] - time_offset,
                y=lc[y_key][not_ul],
                yerr=lc[yerr_key][not_ul],
                color=color,
                label=band.replace('ZTF_', ''),
                **kwargs,
            )

            if include_upper_lim and (~not_ul).any():
                ax.scatter(
                    x=lc['jd'][~not_ul] - time_offset,
                    y=lc[y_key][~not_ul],
                    color=color,
                    **scatter_kwargs,
                )

        if y_units == 'mag':
            ax.invert_yaxis()

        if time_as_str:
            ticks_as_time = Time(ax.get_xticks(), format='jd')
            ax.set_xticks(
                ticks_as_time.jd,
                ticks_as_time.strftime('%m-%d-%Y'),
                **xlab_kwargs,
            )
            ax.set_xlabel('Date')
        else:
            ax.set_xlabel(f'JD - {time_offset:.2f} [days]')

        ax.set_ylabel('Magnitude' if y_units == 'mag' else 'Flux [DN]')
        ax.legend()

        return ax

    def plot_rolled(
        self,
        rolled: pd.DataFrame,
        bands: List[str],
        ax: Axes,
        time_offset: float,
        colors: Dict[str, str] = BAND_COLORS,
        include_upper_lim: bool = True,
        fmt: str = 's',
        markersize: int = 5,
        capsize: int = 2,
    ) -> None:
        """Plot a rolling-stack DataFrame (output of rolling_stack()) onto ax."""
        for band in bands:
            grp = rolled[rolled['band'] == band].sort_values('jd_center')
            if grp.empty:
                continue
            not_ul = grp['upperlim'] == 0
            color = colors.get(band)

            ax.errorbar(
                x=grp['jd_center'][not_ul] - time_offset,
                y=grp['mag'][not_ul],
                yerr=grp['mag_err'][not_ul],
                fmt=fmt, markersize=markersize, capsize=capsize,
                color=color, label=band.replace('ZTF_', ''),
            )

            if include_upper_lim and (~not_ul).any():
                ax.scatter(
                    x=grp['jd_center'][~not_ul] - time_offset,
                    y=grp['mag'][~not_ul],
                    marker='v', s=markersize * 4,
                    color=color, alpha=0.4,
                )

        ax.invert_yaxis()
        ax.set_ylabel('Magnitude')
        ax.legend()

    def plot_rolling_stack_summary(
        self,
        window_days: tuple = (30, 182, 365, 730),
        window_imgs: tuple = (10, 30, 60, 100),
        bands: Optional[List[str]] = None,
        colors: Dict[str, str] = BAND_COLORS,
        include_upper_lim: bool = True,
        time_offset: Union[str, float] = 'first',
        time_as_str: bool = True,
    ) -> plt.Figure:
        """Summary: single-epoch on top, 3x2 grid of rolling stacks below.

        Left column is day-windowed, right column is image-windowed; rows are
        the short and long window sizes from window_days and window_imgs.

        Args:
            window_days: (short, long) day-based rolling window widths.
            window_imgs: (short, long) image-count rolling window widths.
            bands: filters to plot (default: all present).
            colors: mapping of filter name to matplotlib color.
            include_upper_lim: whether to draw upper limits as downward triangles.
            time_offset: 'first' to subtract the earliest JD, or a float JD value.

        Returns:
            The Figure containing all panels.
        """
        bands = bands or self.bands
        if time_as_str:
            time_offset = 0.0
        elif time_offset == 'first':
            time_offset = np.nanmin(self._df['jd'])

        n_rows = 1 + len(window_days)
        panel_cfg = (
            [(w, 'days',   's', (r, 0)) for r, w in enumerate(window_days,   start=1)] +
            [(w, 'images', '^', (r, 1)) for r, w in enumerate(window_imgs, start=1)]
        )

        fig = plt.figure(figsize=(20, 5 * n_rows))
        ax_single = plt.subplot2grid((n_rows, 2), (0, 0), colspan=2)
        panel_axes = {
            pos: plt.subplot2grid((n_rows, 2), pos, sharex=ax_single)
            for _, _, _, pos in panel_cfg
        }

        self.plot(bands=bands, ax=ax_single, colors=colors,
                  include_upper_lim=include_upper_lim, time_offset=time_offset,
                  time_as_str=False)
        ax_single.set_title('Single-epoch')
        ax_single.set_xlabel('')
        plt.setp(ax_single.get_xticklabels(), visible=False)

        for window, unit, fmt, pos in panel_cfg:
            rolled = self.rolling_stack(window=window, bands=bands, window_unit=unit)
            ax = panel_axes[pos]
            self.plot_rolled(rolled, bands=bands, ax=ax, time_offset=time_offset,
                             colors=colors, include_upper_lim=include_upper_lim, fmt=fmt)
            ax.set_title(f'{window}-{unit[:-1]} rolling stack')
            if pos[1] == 1:
                ax.set_ylabel('')
            if pos[0] < n_rows - 1:
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), visible=False)

        if time_as_str:
            bottom_row = n_rows - 1
            for pos, ax in panel_axes.items():
                if pos[0] == bottom_row:
                    ticks_as_time = Time(ax.get_xticks(), format='jd')
                    ax.set_xticks(
                        ticks_as_time.jd,
                        ticks_as_time.strftime('%m-%d-%Y'),
                        rotation=45,
                        ha='right',
                    )
                    ax.set_xlabel('Date')
        else:
            for pos, ax in panel_axes.items():
                if pos[0] == n_rows - 1:
                    ax.set_xlabel(f'JD - {time_offset:.2f} [days]')

        plt.tight_layout()
        return fig

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f'ForcedPhotometry(ra={self.ra}, dec={self.dec}, '
            f'n_epochs={len(self._df)}, bands={self.bands})'
        )
