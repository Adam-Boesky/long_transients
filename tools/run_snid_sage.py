"""Run SNID SAGE on every follow-up spectrum and record the verdict in target.yaml.

    python tools/run_snid_sage.py            # every target
    python tools/run_snid_sage.py <slug> ... # just these
    python tools/run_snid_sage.py --force    # rerun even where a result of this version exists

Each `spectra` entry's reduced `file` is cross-correlated against the SNID SAGE
template bank at the target's `redshift` (a free search over 0 < z < 1 when the
target has none). The point is "does any SN, SLSN, TDE or other transient
template match, or only AGN/galaxy ones" -- the bank cannot separate an ANT
from an AGN, and it has no rest-UV templates, so targets above z ~ 0.8 overlap
the bank too little to say anything (see `lap`).

Outputs, under `<slug>/spectra/snid_sage/<stem>/`:
    <stem>.output                 SNID SAGE's own summary and match table
    <stem>_flux_spectrum.png      spectrum with the best template overplotted
    <stem>_flattened_spectrum.png the continuum-removed view the correlation uses
and a compact `snid_sage` dict on the spectrum's entry in target.yaml.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Extracting.utils import get_data_path  # noqa: E402
from Source_Analysis import followup  # noqa: E402

# The template bank (~460 MB) lives with the data, not in the home directory.
os.environ.setdefault('SNID_SAGE_TEMPLATE_DIR', str(Path(get_data_path()) / 'snid_sage_templates'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from snid_sage import __version__ as SNID_SAGE_VERSION  # noqa: E402
from snid_sage.shared.templates_manager import get_templates_dir  # noqa: E402
from snid_sage.shared.utils.math_utils import get_best_metric_value  # noqa: E402
from snid_sage.snid.io import write_result  # noqa: E402
from snid_sage.snid.plotting import plot_flat_comparison, plot_flux_comparison  # noqa: E402
from snid_sage.snid.snid import run_snid  # noqa: E402

OUT_DIR = 'snid_sage'
ZMAX_FREE = 1.0


def top_match(result):
    """The match the GUI shows: winning cluster, else filtered, else best; highest metric first."""
    cl = getattr(result, 'clustering_results', None) or {}
    matches = (cl.get('best_cluster') or {}).get('matches') if cl.get('success') else None
    matches = matches or result.filtered_matches or result.best_matches or []
    return max(matches, key=get_best_metric_value) if matches else None


def summary(result, forced_z, out_dir: Path) -> dict:
    head = {'version': SNID_SAGE_VERSION, 'run_date': date.today().isoformat(), 'forced_z': forced_z,
            'success': bool(result.success), 'output_dir': str(out_dir)}
    m = top_match(result)
    if not result.success or m is None:
        # No template overlapped enough to correlate (lap < lapmin); typical above z ~ 0.8.
        return {**head, 'type': None, 'note': 'no template match above the overlap floor'}
    return {
        **head,
        'type': result.consensus_type,
        'subtype': result.best_subtype,
        'template': m['name'],
        'template_type': m['type'],
        'template_subtype': m['subtype'],
        'z': round(float(result.redshift), 5),
        'z_err': round(float(result.redshift_error), 5),
        'hsigma_lap_ccc': round(float(get_best_metric_value(m)), 3),
        'lap': round(float(m['lap']), 3),
        'type_fractions': {k: round(float(v), 3) for k, v in (result.type_fractions or {}).items()},
        'n_matches': len(result.filtered_matches or []),
    }


def run_one(target: followup.Target, spec: dict) -> dict:
    stem = spec['file'].stem
    out_dir = target.path / followup.SPECTRA_DIR / OUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    z = target.redshift
    result, _ = run_snid(
        str(spec['file']), str(get_templates_dir()),
        forced_redshift=z, zmax=ZMAX_FREE,
        aband_remove=True, skyclip=True, wavelength_masks=list(followup.TELLURIC_BANDS),
        show_plots=False, verbose=False,
    )
    write_result(result, str(out_dir / f'{stem}.output'))
    m = top_match(result)
    if m is not None:
        for plotter, name in ((plot_flux_comparison, 'flux_spectrum'), (plot_flat_comparison, 'flattened_spectrum')):
            fig = plotter(m, result)
            fig.savefig(out_dir / f'{stem}_{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    return summary(result, z is not None, out_dir.relative_to(target.path))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('slugs', nargs='*')
    p.add_argument('--force', action='store_true', help='rerun spectra already done with this version')
    args = p.parse_args(argv)
    logging.getLogger('snid_sage').setLevel(logging.WARNING)

    targets = [followup.get_target(s) for s in args.slugs] if args.slugs else followup.iter_targets()
    for t in targets:
        entries = t.meta.get('spectra') or []
        for entry, spec in zip(entries, t.spectra):
            done = (entry.get('snid_sage') or {}).get('version') == SNID_SAGE_VERSION
            if done and not args.force:
                print(f'  {t.slug}/{spec["file"].name}: already run with {SNID_SAGE_VERSION}')
                continue
            s = run_one(t, spec)
            entry['snid_sage'] = s
            followup.write_target(t)
            if s['type'] is None:
                print(f'  {t.slug}/{spec["file"].name}: no match')
                continue
            z = f'z={s["z"]:.3f}{"" if s["forced_z"] else " (free)"}'
            print(f'  {t.slug}/{spec["file"].name}: {s["type"]}/{s["subtype"]} via {s["template"]} '
                  f'{z} hsigma={s["hsigma_lap_ccc"]} lap={s["lap"]} fractions={s["type_fractions"]}')


if __name__ == '__main__':
    main(sys.argv[1:])
