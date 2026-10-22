"""Streamlit app for flipping through analysis-page PDFs and assigning labels.

Not run directly - launched via the `classify_pages` CLI wrapper, which calls:
    streamlit run classify_pages_app.py -- <analysis_page_dirpath>

Labels are persisted to `<analysis_page_dirpath>/labels.csv` (columns:
filename, ra, dec, label, timestamp), written after every assignment so
progress survives a crash or ctrl-c.

Within each catalog subdirectory (e.g. in_both, in_ztf), pages are ordered by
decreasing dmag per that subdirectory's `dmag_order*.csv` sidecar(s), which
`store_src_plots.write_dmag_sidecar` already writes alongside the PDFs.
"""
import sys
import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

sys.path.append('/Users/adamboesky/Research/long_transients')
from Source_Analysis.coord_utils import parse_coord_stem

st.set_page_config(page_title='classify_pages', layout='wide')

LABELS_FNAME = 'labels.csv'


def get_dirpath() -> Path:
    args = sys.argv[1:]
    if not args:
        st.error('Usage: classify_pages <analysis_page_dirpath>')
        st.stop()
    dirpath = Path(args[0]).expanduser().resolve()
    if not dirpath.is_dir():
        st.error(f'Not a directory: {dirpath}')
        st.stop()
    return dirpath


@st.cache_data(show_spinner=False)
def load_dmag_order(catalog_dirpath_str: str, mtimes: tuple[float, ...]) -> dict[str, float]:
    """Merge every `dmag_order*.csv` in a catalog directory into {filename: mean_abs_dmag}.

    `mtimes` (of the matched CSVs) is part of the cache key purely to
    invalidate the cache if a sidecar is regenerated.
    """
    catalog_dirpath = Path(catalog_dirpath_str)
    dmag_by_fname: dict[str, float] = {}
    for csv_path in sorted(catalog_dirpath.glob('dmag_order*.csv')):
        sidecar = pd.read_csv(csv_path, usecols=['filename', 'mean_abs_dmag'])
        dmag_by_fname.update(zip(sidecar['filename'], sidecar['mean_abs_dmag']))
    return dmag_by_fname


def discover_catalogs(dirpath: Path, all_files: list[Path]) -> dict[str, list[Path]]:
    """Group PDFs by their catalog subdirectory (e.g. in_both, in_ztf), each
    ordered by decreasing dmag per that catalog's `dmag_order*.csv` sidecar(s)
    (files with no dmag entry sort last, alphabetically).
    """
    by_catalog: dict[str, list[Path]] = {}
    for f in all_files:
        parts = f.relative_to(dirpath).parts
        catalog = parts[0] if len(parts) > 1 else '(root)'
        by_catalog.setdefault(catalog, []).append(f)

    catalogs: dict[str, list[Path]] = {}
    for catalog, files in by_catalog.items():
        catalog_dirpath = dirpath if catalog == '(root)' else dirpath / catalog
        csv_paths = sorted(catalog_dirpath.glob('dmag_order*.csv'))
        dmag_by_fname = load_dmag_order(
            str(catalog_dirpath), tuple(p.stat().st_mtime for p in csv_paths),
        )
        catalogs[catalog] = sorted(
            files,
            key=lambda f: (-dmag_by_fname[f.name] if f.name in dmag_by_fname else float('inf'), f.name),
        )
    return catalogs


def load_table(dirpath: Path, files: list[Path]) -> pd.DataFrame:
    labels_path = dirpath / LABELS_FNAME
    if labels_path.exists():
        df = pd.read_csv(labels_path, dtype={'label': str}, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=['filename', 'ra', 'dec', 'label', 'timestamp'])

    # Add a row for any PDF on disk that isn't in the table yet (e.g. new pages
    # since the last labeling session).
    known = set(df['filename'])
    new_rows = []
    for f in files:
        rel = str(f.relative_to(dirpath))
        if rel not in known:
            coord = parse_coord_stem(f.stem)
            ra, dec = coord if coord else (None, None)
            new_rows.append({'filename': rel, 'ra': ra, 'dec': dec, 'label': '', 'timestamp': ''})
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)

    return df


def save_table(df: pd.DataFrame, dirpath: Path) -> None:
    df.to_csv(dirpath / LABELS_FNAME, index=False)


def set_label(df: pd.DataFrame, dirpath: Path, rel: str, label: str) -> pd.DataFrame:
    df.loc[df['filename'] == rel, 'label'] = label
    df.loc[df['filename'] == rel, 'timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    save_table(df, dirpath)
    return df


@st.cache_data(show_spinner=False)
def render_pdf_page(path_str: str, mtime: float, dpi: int = 150) -> bytes:
    """Rasterize a PDF's first page to PNG bytes.

    Browsers refuse to run Chrome's built-in PDF viewer inside Streamlit's
    (sandboxed) iframe element, so PDFs are rendered server-side instead of
    embedded directly. `mtime` is part of the cache key purely to invalidate
    the cache if a page is regenerated on disk.
    """
    doc = fitz.open(path_str)
    pix = doc[0].get_pixmap(dpi=dpi)
    return pix.tobytes('png')


def show_pdf(path: Path) -> None:
    png_bytes = render_pdf_page(str(path), path.stat().st_mtime)
    st.image(png_bytes, width='stretch')


def main():
    dirpath = get_dirpath()
    all_files = sorted(dirpath.rglob('*.pdf'))
    if not all_files:
        st.error(f'No PDFs found under {dirpath}')
        st.stop()

    if st.session_state.get('dirpath') != str(dirpath):
        st.session_state.df = load_table(dirpath, all_files)
        save_table(st.session_state.df, dirpath)
        st.session_state.dirpath = str(dirpath)
        st.session_state.idx_by_catalog = {}

    df = st.session_state.df

    # Sidebar: catalog toggle (e.g. in_both vs in_ztf), so it's always clear
    # which catalog subset is being sorted through.
    st.sidebar.title('classify_pages')
    st.sidebar.write(f'**{dirpath.name}**')

    catalogs = discover_catalogs(dirpath, all_files)
    catalog_names = sorted(catalogs)
    if len(catalog_names) > 1:
        catalog = st.sidebar.segmented_control(
            'Catalog', catalog_names, default=catalog_names[0], required=True, key='catalog',
        )
    else:
        catalog = catalog_names[0]

    files = catalogs[catalog]
    st.session_state.idx_by_catalog.setdefault(catalog, 0)
    idx = max(0, min(st.session_state.idx_by_catalog[catalog], len(files) - 1))
    st.session_state.idx_by_catalog[catalog] = idx
    n = len(files)

    rel = str(files[idx].relative_to(dirpath))
    row = df.loc[df['filename'] == rel].iloc[0]
    catalog_rels = {str(f.relative_to(dirpath)) for f in files}
    n_labeled = int((df.loc[df['filename'].isin(catalog_rels), 'label'] != '').sum())

    st.sidebar.progress(n_labeled / n)
    st.sidebar.write(f'{n_labeled} / {n} labeled in **{catalog}**')
    st.sidebar.caption('Ordered by decreasing Δmag where available')

    existing_labels = df.loc[df['label'] != '', 'label'].value_counts().index.tolist()

    st.sidebar.subheader('Assign label')
    for i, lab in enumerate(existing_labels):
        shortcut = str(i + 1) if i < 9 else None
        clicked = st.sidebar.button(
            f'{lab}' + (f'  ({shortcut})' if shortcut else ''),
            width='stretch',
            key=f'lab_{lab}',
            shortcut=shortcut,
        )
        if clicked:
            st.session_state.df = set_label(df, dirpath, rel, lab)
            st.session_state.idx_by_catalog[catalog] = min(idx + 1, n - 1)
            st.rerun()

    new_label = st.sidebar.text_input('New label')
    if st.sidebar.button('Add + assign', width='stretch') and new_label.strip():
        st.session_state.df = set_label(df, dirpath, rel, new_label.strip())
        st.session_state.idx_by_catalog[catalog] = min(idx + 1, n - 1)
        st.rerun()

    st.sidebar.divider()
    nav_cols = st.sidebar.columns(2)
    prev_clicked = nav_cols[0].button('< Prev', width='stretch', shortcut='Left')
    next_clicked = nav_cols[1].button('Next >', width='stretch', shortcut='Right')
    if prev_clicked and idx > 0:
        st.session_state.idx_by_catalog[catalog] = idx - 1
        st.rerun()
    if next_clicked and idx < n - 1:
        st.session_state.idx_by_catalog[catalog] = idx + 1
        st.rerun()

    if st.sidebar.button('Jump to next unlabeled', width='stretch'):
        rel_to_pos = {str(f.relative_to(dirpath)): i for i, f in enumerate(files)}
        unlabeled_positions = sorted(
            rel_to_pos[r] for r in df.loc[df['label'] == '', 'filename'] if r in rel_to_pos and r in catalog_rels
        )
        if unlabeled_positions:
            nxt = next((p for p in unlabeled_positions if p > idx), unlabeled_positions[0])
            st.session_state.idx_by_catalog[catalog] = nxt
            st.rerun()

    # Main panel
    st.write(f'### {rel}  ({idx + 1} / {n})')
    ra, dec = row['ra'], row['dec']
    if pd.notna(ra) and pd.notna(dec):
        st.write(f'RA, Dec: ({float(ra):.5f}, {float(dec):.5f})')
    current = row['label'] if row['label'] else '*(unlabeled)*'
    st.write(f'Current label: **{current}**')

    show_pdf(files[idx])


if __name__ == '__main__':
    main()
