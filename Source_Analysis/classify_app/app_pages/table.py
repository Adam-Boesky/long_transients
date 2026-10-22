"""Every page in one sortable table: properties, verdicts and comments."""
import streamlit as st

from lib import chrome, filters, manifest as mf, pages_table, stores
from lib.workspace import get_workspace

ws = get_workspace()
member = st.session_state['member']

chrome.page_header(ws, member, 'Table', large=True)

pages = mf.load_manifest(ws)
latest = stores.latest_labels(ws)

# ── filters, shared with the Classify view ────────────────────────────────────
# Same widget keys, so a cut made here is still in force when you switch tabs.
# Rendered flat rather than inside an expander: `detail_filters` already puts
# the fiddlier cuts behind one of its own.

with st.container(border=True):
    scope_mask = filters.scope_controls(pages, include_search=False)
    detail_mask = filters.detail_filters(pages, member, latest)

filtered = filters.triage_order(pages[(scope_mask & detail_mask).to_numpy()])
table = pages_table.build(filtered, latest, member)

# ── search across everything shown ────────────────────────────────────────────

query = st.text_input(
    'Search', placeholder='filename, classification, bucket, comment or coordinates',
    label_visibility='collapsed', key='table_search',
)
if query.strip():
    needle = query.strip().lower()
    hits = pages_table.searchable_text(table).str.contains(needle, regex=False)
    table = table[hits.to_numpy()]

if table.empty:
    st.info('No pages match the current filters and search.',
            icon=':material/filter_alt_off:')
    st.stop()

st.caption(
    f'{len(table):,} of {len(pages):,} pages'
    + (f' · {len(filtered):,} after filters' if len(filtered) != len(table) else '')
    + ' · click a column header to sort'
)

pages_table.render(table)
