"""Copy figures into the Overleaf paper and push them, so they appear in Overleaf at once.

    python tools/push_figures.py Figures/a.pdf ...     # add or update these in the paper
    python tools/push_figures.py                       # refresh every figure the paper already has
    python tools/push_figures.py --committed [REV]     # the paper's figures that commit REV (default HEAD) changed

The paper's own `figures/` directory is the list of paper figures: push a figure
once by name and every later commit here that changes a file of that name
updates it, through the post-commit hook (`tools/install_hooks.sh`), which runs
the `--committed` form. The Overleaf commit names the long_transients commit it
came from.

The paper is a git clone of the Overleaf project (its git bridge), kept beside
this repo at `../long_transients_paper` or `$LT_PAPER_DIR`; see the clone
command in `paper_dir()`. Figures land in `figures/` under their own file
names, and only files whose content changed are committed, so re-running is
free. The clone is pulled first, so the check is against what the paper has now
and the push cannot be rejected; nothing but the named files is touched, so
text being edited in Overleaf is never disturbed. Pushes authenticate with the
Overleaf git token in the macOS keychain (host git.overleaf.com, user git);
`~/vault/overleaf_token.txt` is its backup.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLEAF_PROJECT = '6aec8308f1f30d5bc98431ba'
FIGURES_SUBDIR = 'figures'
FIGURES_DIR = ROOT / 'Figures'


def paper_dir(required: bool = True) -> Path | None:
    d = Path(os.environ.get('LT_PAPER_DIR', ROOT.parent / 'long_transients_paper'))
    if (d / '.git').exists():
        return d
    if required:
        sys.exit(f'{d} is not the paper clone; make it with\n'
                 f'  git clone https://git.overleaf.com/{OVERLEAF_PROJECT} {d}')
    return None


def paper_figures(d: Path) -> dict[str, Path]:
    """The figures the paper has, by file name, mapped to their source in Figures/ (pulls the clone first)."""
    git(d, 'pull', '--rebase', '--quiet')
    names = sorted(p.name for p in (d / FIGURES_SUBDIR).glob('*') if p.is_file())
    return {n: FIGURES_DIR / n for n in names}


def git(d: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(d), *args], check=True, text=True, capture_output=True).stdout


def committed_files(rev: str, d: Path) -> tuple[str, list[Path]]:
    """(short sha, the paper's figures that commit `rev` changed)."""
    sha = git(ROOT, 'rev-parse', '--short', rev).strip()
    changed = git(ROOT, 'diff-tree', '--no-commit-id', '--name-only', '-r', rev).split()
    have = paper_figures(d)
    return sha, [ROOT / p for p in sorted(changed) if Path(p).name in have and (ROOT / p).is_file()]


def push(files: list[Path], message: str | None = None, do_push: bool = True) -> list[str]:
    """Copy `files` into the paper; commit and push those whose content changed. Returns their names."""
    d = paper_dir()
    git(d, 'pull', '--rebase', '--quiet')  # Overleaf keeps one line of history, so stay linear
    dest = d / FIGURES_SUBDIR
    dest.mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)
        git(d, 'add', str(Path(FIGURES_SUBDIR) / f.name))
    changed = [line[3:].split('/', 1)[1] for line in git(d, 'status', '--porcelain', FIGURES_SUBDIR).splitlines()]
    if not changed:
        return []
    git(d, 'commit', '--quiet', '-m', message or f'Update figures: {", ".join(changed)}')
    if do_push:
        git(d, 'push', '--quiet')
    return changed


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='*', type=Path, help='figures to add or update (default: all the paper has)')
    p.add_argument('--committed', nargs='?', const='HEAD', metavar='REV',
                   help="push the paper's figures that commit REV changed (default HEAD); quiet no-op otherwise")
    p.add_argument('-m', '--message', help='commit message (default: lists the files)')
    p.add_argument('--no-push', action='store_true', help='commit locally without pushing to Overleaf')
    args = p.parse_args(argv)

    if args.committed:
        d = paper_dir(required=False)
        if d is None:  # e.g. the cluster clone: nothing to do
            return
        sha, files = committed_files(args.committed, d)
        if not files:
            return
        message = args.message or f'Update figures: {", ".join(f.name for f in files)} (long_transients {sha})'
    else:
        files = args.files or list(paper_figures(paper_dir()).values())
        message = args.message
    missing = [f for f in files if not f.is_file()]
    if missing:
        sys.exit(f'not found: {", ".join(map(str, missing))}')

    changed = push(files, message, not args.no_push)
    if not changed:
        print('figures unchanged; nothing to push')
    elif args.no_push:
        print(f'committed {", ".join(changed)} (not pushed)')
    else:
        print(f'pushed {", ".join(changed)} to Overleaf ({FIGURES_SUBDIR}/)')


if __name__ == '__main__':
    main(sys.argv[1:])
