"""Entry point for the analysis-page classification app.

    classify_app <workspace_dir>

which is a thin wrapper around

    streamlit run streamlit_app.py -- --workspace <workspace_dir>

The workspace is built by `scripts/build_classify_manifest.py`. See
`plan_classify_app.md` for the overall design.
"""
import streamlit as st

from lib.identity import require_member
from lib.styles import inject as inject_styles
from lib.workspace import get_workspace

st.set_page_config(
    page_title='Classify pages',
    page_icon=':material/image_search:',
    layout='wide',
)

inject_styles()

ws = get_workspace()

# Gate before navigation so the picker is the whole screen rather than a page.
member = require_member(ws)

page = st.navigation([
    st.Page('app_pages/classify.py', title='Classify',
            icon=':material/image_search:', default=True),
    st.Page('app_pages/table.py', title='Table',
            icon=':material/table_rows:'),
    st.Page('app_pages/progress.py', title='Progress',
            icon=':material/insights:'),
])

# Each page opens with `chrome.page_header`, which puts its title and the
# account menu on one row -- so the account control is not rendered here.
page.run()
