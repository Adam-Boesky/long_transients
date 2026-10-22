"""Flip through filtered analysis pages, drop each into a bucket, leave a comment."""
from urllib.parse import urlencode, urlsplit, urlunsplit

import streamlit as st

from lib import chrome, filters, manifest as mf, pages_store, stores
from lib.constants import COLOR_NAMES
from lib.components import copy_link_button, key_nav, page_viewer
from lib.external import links_for
from lib.workspace import get_workspace

ws = get_workspace()
member = st.session_state['member']

flash = st.session_state.pop('flash', None)
if flash:
    st.toast(flash, icon=':material/check:')

pages = mf.load_manifest(ws)
latest = stores.latest_labels(ws)
buckets = stores.read_buckets(ws)
bucket_colors = stores.bucket_colors(ws)
mine = stores.my_verdicts(ws, member)

# Sidebar reading order is: nav, scope (catalog + search), buckets, then the
# rest of the filters. The bucket buttons need to know which page is current,
# which depends on every filter, so all three slots are reserved up front and
# then filled in computation order rather than display order.
with st.sidebar:
    scope_slot = st.container()
    bucket_slot = st.container()
    detail_slot = st.container()

with scope_slot:
    scope_mask = filters.scope_controls(pages)
with detail_slot:
    st.divider()
    detail_mask = filters.detail_filters(pages, member, latest)

filtered = filters.triage_order(pages[(scope_mask & detail_mask).to_numpy()])
order = filtered['page_id'].tolist()

if not order:
    st.info('No pages match the current filters.', icon=':material/filter_alt_off:')
    with bucket_slot:
        st.caption('Nothing selected.')
    st.stop()

# A `?page=` deep link wins on first load; after that the session drives it.
current = st.session_state.get('page_id')
if current is None:
    from_url = st.query_params.get('page')
    if from_url in set(order):
        current = from_url
if current not in order:
    current = order[0]
st.session_state['page_id'] = current

# Keep the URL in step so it is always copyable. Guarded because an
# unconditional write can bounce the script into another rerun.
if st.query_params.get('page') != current:
    st.query_params['page'] = current

position = order.index(current)
page = filtered.iloc[position]

stored_bucket = str(mine.loc[current, 'bucket']) if current in mine.index else ''
stored_comment = str(mine.loc[current, 'comment']) if current in mine.index else ''
comment_key = f'comment::{current}'


def draft_comment() -> str:
    return st.session_state.get(comment_key, stored_comment)


def commit(bucket: str, comment: str) -> None:
    stores.append_verdict(ws, member, page, bucket, comment)


def flush_comment() -> None:
    """Persist an edited comment before leaving the page.

    Keeps the existing bucket, so navigating away can never silently clear a
    classification -- and an arrow key can never discard typed text.
    """
    if draft_comment() != stored_comment:
        commit(stored_bucket, draft_comment())


def go_to(new_position: int) -> None:
    st.session_state['page_id'] = order[max(0, min(new_position, len(order) - 1))]


def share_url() -> str:
    """A link straight to this page, without carrying our own identity along."""
    raw = getattr(st.context, 'url', None) or ''
    parts = urlsplit(raw)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode({'page': current}), '')
    )


def is_bucketed(page_id: str) -> bool:
    return page_id in mine.index and bool(str(mine.loc[page_id, 'bucket']).strip())


# ── sidebar: buckets ──────────────────────────────────────────────────────────

with bucket_slot:
    st.subheader('Buckets')

    if buckets.empty:
        st.caption('No buckets yet. Add the first one below.')

    if not buckets.empty:
        # Streamlit buttons take no colour argument, so this is the one place
        # the palette needs CSS. `key` becomes an `st-key-` class, giving each
        # button its own hook -- underscores only, since a `::` in the key
        # would have to be escaped in the selector.
        #
        # The bucket a page is already filed under is filled with a wash of its
        # own colour rather than `type='primary'`: primary paints every button
        # the same blue, which would override the very colours this is for.
        rules = []
        for _, row in buckets.iterrows():
            hexcode = bucket_colors.get(row['display_name'], ('', '#888888'))[1]
            selector = f'.st-key-bucket_{row["bucket_id"]} button'
            rules.append(f'{selector} {{ border-left: 5px solid {hexcode}; }}')
            if row['display_name'] == stored_bucket:
                # 8-digit hex is an alpha channel: a light wash, so the label
                # stays readable in either theme.
                rules.append(
                    f'{selector} {{ background-color: {hexcode}22;'
                    f' border-color: {hexcode}; font-weight: 600; }}'
                )
        st.html('<style>' + '\n'.join(rules) + '</style>')

    for _, bucket in buckets.iterrows():
        name = bucket['display_name']
        bucket_id = bucket['bucket_id']

        with st.container(horizontal=True, vertical_alignment='center'):
            if st.button(
                name,
                key=f'bucket_{bucket_id}',
                width='stretch',
                icon=':material/check:' if name == stored_bucket else None,
                help=str(bucket['description']).strip() or None,
            ):
                # Stay on the page. The rerun is still needed so the button
                # shows as filed and the counts pick up the new verdict; use
                # Next or "Next unbucketed" to move on.
                commit(name, draft_comment())
                st.rerun()

            with st.popover(':material/more_horiz:', help=f'Edit “{name}”'):
                st.caption(f'Colour for **{name}**')
                # Note the name: `current` is the page_id for this whole
                # script, and a page module is one flat scope -- assigning to
                # it here would send the viewer off to load a page called
                # "green".
                current_color = stores.color_name_for(bucket_colors, name)
                color_key = f'color_{bucket_id}'
                # Only seed the default the first time: passing it alongside a
                # value already in session state makes Streamlit complain about
                # two sources of truth.
                seed = ({} if color_key in st.session_state
                        else {'default': current_color})
                picked = st.pills(
                    'Colour', COLOR_NAMES, key=color_key,
                    format_func=lambda c: f':{c}[{c}]',
                    label_visibility='collapsed', **seed,
                )
                if picked and picked != current_color:
                    if stores.set_bucket_color(ws, bucket_id, picked):
                        st.session_state['flash'] = f'“{name}” is now {picked}.'
                        st.rerun()
                    st.warning('Could not save that colour.')

    with st.popover('New bucket', icon=':material/add:', width='stretch'):
        with st.form('add_bucket', clear_on_submit=True, border=False):
            new_name = st.text_input('Bucket name')
            new_description = st.text_input('Description', placeholder='optional')
            submitted = st.form_submit_button('Add bucket', type='primary')
        if submitted:
            created, message = stores.add_bucket(ws, new_name, member, new_description)
            if created:
                st.session_state['flash'] = message
                st.rerun()
            st.warning(message)

# ── main panel ────────────────────────────────────────────────────────────────

chrome.page_header(ws, member, current)

with st.container(horizontal=True, vertical_alignment='center'):
    st.caption(
        ' · '.join([
            f'RA, Dec {page["ra"]:.5f}, {page["dec"]:.5f}',
            mf.dmag_line(page),
        ]),
        width='content',
    )
    # Which analysis-page directory this file came from. The name is shown
    # outright; the tooltip carries the full path.
    st.badge(ws.pages_dirname, icon=':material/folder:', color='gray',
             help=str(ws.pages_root))
    if stored_bucket:
        st.badge(stored_bucket, icon=':material/label:',
                 color=stores.color_name_for(bucket_colors, stored_bucket))
    else:
        st.badge('unbucketed', color='gray')
    copy_link_button(share_url(), key=f'copy::{current}')

page_viewer(pages_store.page_png(ws, current), label=current, key=f'viewer::{current}')

left, right = st.columns([2, 1])

with left:
    st.text_area('Your comment', value=stored_comment, key=comment_key,
                 height=120, placeholder='What do you make of this source?')
    if st.button('Save comment', icon=':material/save:',
                 disabled=draft_comment() == stored_comment):
        commit(stored_bucket, draft_comment())
        st.session_state['flash'] = 'Comment saved.'
        st.rerun()

with right:
    others = latest[(latest['page_id'] == current) & (latest['member'] != member)]
    others = others[others['comment'].astype(str).str.strip() != '']
    st.caption(f'Comments from others ({len(others)})')
    if others.empty:
        st.caption('None yet.')
    for _, row in others.iterrows():
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment='center'):
                st.markdown(f'**{row["member"]}**', width='content')
                if str(row['bucket']).strip():
                    st.badge(row['bucket'],
                             color=stores.color_name_for(bucket_colors, row['bucket']))
            st.markdown(row['comment'])
            st.caption(row['timestamp'])

# ── links out to the survey archives ──────────────────────────────────────────

with st.container(horizontal=True, vertical_alignment='center'):
    st.caption('Look up elsewhere', width='content')
    for link in links_for(page):
        st.link_button(link['label'], link['url'], icon=link['icon'],
                       help=link['help'])

# ── bottom navigation bar ─────────────────────────────────────────────────────

bucketed_here = sum(1 for page_id in order if is_bucketed(page_id))

with st.container(border=True, horizontal=True, vertical_alignment='center',
                  key='nav_bar'):
    if st.button('Prev', icon=':material/chevron_left:',
                 disabled=position == 0):
        flush_comment()
        go_to(position - 1)
        st.rerun()

    # The key carries the position so `value` re-seeds the box whenever the
    # page changes underneath it.
    jump = st.number_input(
        'Go to page', min_value=1, max_value=len(order), value=position + 1, step=1,
        key=f'jump::{current}::{len(order)}', width=96, label_visibility='collapsed',
    )
    st.caption(f'of {len(order)}', width='content')

    if st.button('Next', icon=':material/chevron_right:',
                 disabled=position == len(order) - 1):
        flush_comment()
        go_to(position + 1)
        st.rerun()

    if st.button('Next unbucketed', icon=':material/skip_next:'):
        flush_comment()
        remaining = [i for i, page_id in enumerate(order) if not is_bucketed(page_id)]
        ahead = [i for i in remaining if i > position]
        if remaining:
            go_to(ahead[0] if ahead else remaining[0])
            st.rerun()
        st.session_state['flash'] = 'Everything in this selection is bucketed.'
        st.rerun()

    st.caption(
        f'{bucketed_here} / {len(order)} bucketed'
        + (f' · {len(order) - bucketed_here} left' if bucketed_here < len(order) else ''),
        width='content',
    )

# Left/right arrows mirror Prev/Next. The component renders nothing -- it is
# here, after the bar, so it reads where it acts -- and reports a keypress as a
# one-rerun trigger. It refuses to fire while a field has focus, while the zoom
# overlay is up, or with a modifier held; see lib/components.py.
step = key_nav(key='key_nav')
if step:
    target = position + int(step['dir'])
    # Silently ignore a press off either end rather than rerunning to land on
    # the page we are already showing.
    if 0 <= target < len(order):
        flush_comment()
        go_to(target)
        st.rerun()

if int(jump) != position + 1:
    flush_comment()
    go_to(int(jump) - 1)
    st.rerun()
