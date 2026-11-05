"""Cross-match a coordinate list against objects that already have studies or our own spectra.

    python tools/check_known_objects.py gemini_2027A.txt [--radius 5] [--csv out.csv]

The input is the website's coordinate-list format (`api.coord_lists.parse`): one
position per line, `ra dec` or `name ra dec`, decimal or sexagesimal, commas or
whitespace, `#` comments. Positions are matched against

  ant_ent    Data/ant_ent_objects.csv   published ANTs/ENTs (incl. Hinkle+25 Sci. Adv.)
  mir_tde    Data/mir_tde_objects.csv   Masterson+24 / Nair+26 WISE-selected TDEs
  ants_nest  ANTs-Nest/                 ant_properties.csv plus any spectrum FITS header
  followup   Data/followup/             our targets; the note says how many spectra we have

Every match within the radius is listed, so a position known under several
papers shows each of them.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.coord_lists import parse  # noqa: E402
from Extracting.utils import get_data_path  # noqa: E402
from Source_Analysis import followup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(get_data_path())
ANTS_NEST = ROOT / 'ANTs-Nest'
DEFAULT_RADIUS_ARCSEC = 5.0


def read_positions(path: Path) -> pd.DataFrame:
    """name, ra, dec per position; the name is the leading token when the line has one."""
    lines = [l.split('#', 1)[0].strip() for l in path.read_text().splitlines()]
    rows = []
    for line in lines:
        if not line:
            continue
        coords, bad, _, _ = parse(line)
        if not coords:  # header or unparsable line
            continue
        tokens = line.replace(',', ' ').replace('\t', ' ').split()
        name = tokens[0] if _is_name(tokens[0]) else f'{coords[0][0]:.5f}_{coords[0][1]:+.5f}'
        rows.append({'name': name, 'ra': coords[0][0], 'dec': coords[0][1]})
    _, bad, _, _ = parse(path.read_text())
    for b in bad:
        print(f'  ! not read as a position: {b!r}', file=sys.stderr)
    return pd.DataFrame(rows)


def _is_name(token: str) -> bool:
    try:
        float(token)
        return False
    except ValueError:
        return ':' not in token


def _literature(path: Path, source: str) -> pd.DataFrame:
    d = pd.read_csv(path, comment='#')
    return pd.DataFrame({'source': source, 'object': d['object'], 'ra': d['ra_deg'], 'dec': d['dec_deg'],
                         'note': d['paper_key']})


def _ants_nest() -> pd.DataFrame:
    d = pd.read_csv(ANTS_NEST / 'ant_properties.csv').dropna(subset=['RA', 'Dec'])
    spectra_ids = {os.path.basename(p.rstrip('/')) for p in glob.glob(str(ANTS_NEST / 'spectra' / '*/'))}
    rows = [{'source': 'ants_nest', 'object': a['IAU name'] if a['IAU name'] != '-' else a['ZTF ID'],
             'ra': a['RA'], 'dec': a['Dec'],
             'note': a['ZTF ID'] + (' (spectrum)' if a['ZTF ID'] in spectra_ids else '')}
            for _, a in d.iterrows()]
    # Spectra for objects missing from ant_properties.csv: take the position from a FITS header.
    for ztf_id in sorted(spectra_ids - set(d['ZTF ID'])):
        pos = _fits_position(ANTS_NEST / 'spectra' / ztf_id)
        if pos is None:
            print(f'  ! ANTs-Nest/spectra/{ztf_id}: no coordinates found, skipped', file=sys.stderr)
            continue
        rows.append({'source': 'ants_nest', 'object': pos[0], 'ra': pos[1], 'dec': pos[2],
                     'note': f'{ztf_id} (spectrum; position from FITS header)'})
    return pd.DataFrame(rows)


def _fits_position(d: Path):
    from astropy.io import fits
    for f in sorted(d.glob('*.fits')):
        h = fits.getheader(f)
        for ra_key, dec_key in (('CAT-RA', 'CAT-DEC'), ('RA', 'DEC')):
            if h.get(ra_key) is not None and h.get(dec_key) is not None:
                ra, dec = str(h[ra_key]), str(h[dec_key])
                c = (SkyCoord(ra, dec, unit=(u.hourangle, u.deg)) if ':' in ra
                     else SkyCoord(float(ra), float(dec), unit='deg'))
                return h.get('OBJECT', d.name), c.ra.deg, c.dec.deg
    return None


def _followup() -> pd.DataFrame:
    t = followup.load_targets()
    if t.empty:
        return pd.DataFrame(columns=['source', 'object', 'ra', 'dec', 'note'])
    return pd.DataFrame({'source': 'followup', 'object': t['slug'], 'ra': t['ra'], 'dec': t['dec'],
                         'note': t['n_spectra'].map(lambda n: f'{n} spectra' if n else 'target, no spectrum yet')})


def known_objects() -> pd.DataFrame:
    return pd.concat([
        _literature(DATA / 'ant_ent_objects.csv', 'ant_ent'),
        _literature(DATA / 'mir_tde_objects.csv', 'mir_tde'),
        _ants_nest(),
        _followup(),
    ], ignore_index=True)


def crossmatch(positions: pd.DataFrame, known: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    """One row per (position, known object) pair within the radius."""
    a = SkyCoord(positions['ra'].values * u.deg, positions['dec'].values * u.deg)
    b = SkyCoord(known['ra'].values * u.deg, known['dec'].values * u.deg)
    i_a, i_b, sep, _ = b.search_around_sky(a, radius_arcsec * u.arcsec)
    m = pd.DataFrame({'name': positions['name'].values[i_a], 'ra': positions['ra'].values[i_a],
                      'dec': positions['dec'].values[i_a], 'source': known['source'].values[i_b],
                      'object': known['object'].values[i_b], 'note': known['note'].values[i_b],
                      'sep_arcsec': sep.arcsec})
    return m.sort_values(['name', 'sep_arcsec']).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('coords', type=Path, help='text file of positions (website coordinate-list format)')
    p.add_argument('--radius', type=float, default=DEFAULT_RADIUS_ARCSEC, help='match radius [arcsec]')
    p.add_argument('--csv', type=Path, help='also write the matches to this CSV')
    args = p.parse_args()

    positions = read_positions(args.coords)
    known = known_objects()
    print(f'{len(positions)} positions from {args.coords}; '
          f'{len(known)} known objects ({", ".join(f"{s}: {n}" for s, n in known["source"].value_counts().items())}); '
          f'radius {args.radius:g}"\n')

    matches = crossmatch(positions, known, args.radius)
    by_name = {n: g for n, g in matches.groupby('name', sort=False)}
    for _, r in positions.iterrows():
        g = by_name.get(r['name'])
        head = f'{r["name"]:<24s} {r["ra"]:10.5f} {r["dec"]:+10.5f}'
        if g is None:
            print(f'{head}  no match')
            continue
        print(f'{head}  MATCHED')
        for _, m in g.iterrows():
            print(f'{"":46s}  {m["source"]:<10s} {m["object"]:<24s} {m["sep_arcsec"]:5.2f}"  {m["note"]}')

    n_known = len(by_name)
    print(f'\n{len(positions) - n_known} with no match, {n_known} matched within {args.radius:g}"')
    if args.csv:
        matches.to_csv(args.csv, index=False, float_format='%.6f')
        print(f'matches written to {args.csv}')


if __name__ == '__main__':
    main()
