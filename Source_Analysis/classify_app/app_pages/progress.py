"""Who has labeled what, how the buckets are filling up, and where people disagree."""
import pandas as pd
import streamlit as st

from lib import chrome, filters, manifest as mf, pages_table, stores
from lib.workspace import get_workspace

ws = get_workspace()
member = st.session_state['member']

pages = mf.load_manifest(ws)
latest = stores.latest_labels(ws)

chrome.page_header(ws, member, 'Progress', large=True)

bucketed = (
    latest[latest['bucket'].astype(str).str.strip() != '']
    if not latest.empty else pd.DataFrame(columns=stores.LABELS_HEADER)
)
if bucketed.empty:
    st.info('No buckets assigned yet.', icon=':material/hourglass_empty:')
    st.stop()

st.subheader('Buckets')
counts = (
    bucketed.groupby(['bucket', 'member']).size()
    .rename('pages').reset_index()
    .pivot(index='bucket', columns='member', values='pages')
    .fillna(0).astype(int)
)
# Height has to scale with the number of *bars* (buckets x members), not
# buckets alone, or Vega silently drops y-axis labels to make room.
st.bar_chart(
    counts, horizontal=True, stack=False,
    height=max(220, 30 * len(counts) * max(1, len(counts.columns)) + 70),
)

st.subheader('By member')
per_member = (
    bucketed.groupby('member')
    .agg(pages=('page_id', 'nunique'), buckets_used=('bucket', 'nunique'))
    .sort_values('pages', ascending=False)
)
per_member['share of sample'] = (per_member['pages'] / len(pages)).map('{:.1%}'.format)
comment_counts = (
    bucketed[bucketed['comment'].astype(str).str.strip() != '']
    .groupby('member').size().rename('comments')
)
st.dataframe(per_member.join(comment_counts).fillna(0))

st.subheader('Disagreements')
spread = bucketed.groupby('page_id')['bucket'].nunique()
contested = set(spread[spread > 1].index)

if not contested:
    st.caption('No page has been put in two different buckets.')
    st.stop()

st.caption(
    f'{len(contested)} page(s) where members chose different buckets. '
    'The bucket column shows the split and the comments the reasoning.'
)

# The same table as the Table view, narrowed to the contested pages -- the
# pooled bucket and comment columns already show who said what.
contested_pages = filters.triage_order(pages[pages['page_id'].isin(contested)])
pages_table.render(pages_table.build(contested_pages, latest, member),
                   height=min(620, 45 * len(contested_pages) + 45))
