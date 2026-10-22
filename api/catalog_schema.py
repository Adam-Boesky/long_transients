"""Grouping for the 305-column catalog row.

The detail page cannot render a flat 301-key object usefully, and the grouping
is stable across rebuilds, so it is served once as a schema the frontend can
cache rather than repeated inside every source response.

Groups are derived from column prefixes, which the enrichment layer already
applies consistently (`ZTF_*`, `sdss_*`, `simbad_*`, ...). Anything that does
not match a known prefix lands in `detection`, which is the per-band
extraction output (`g_x`, `r_Catalog_Flag`, `fieldid`, ...) plus a few
identifiers.
"""
from __future__ import annotations

import re

# (group key, display label, regex matched against the column name).
# Order matters: the first match wins, so `PanSTARR` is tried before the
# looser prefixes.
GROUP_RULES: list[tuple[str, str, re.Pattern]] = [
    ('ztf',       'ZTF',            re.compile(r'^ZTF[_.]')),
    ('panstarrs', 'Pan-STARRS',     re.compile(r'^(PSTARR|PanSTARR)[_.]')),
    ('wise',      'WISE',           re.compile(r'^wise[_.]')),
    ('simbad',    'SIMBAD',         re.compile(r'^simbad[_.]')),
    ('gaia',      'Gaia',           re.compile(r'^gaia[_.]')),
    ('sdss',      'SDSS',           re.compile(r'^sdss[_.]')),
    ('tns',       'TNS',            re.compile(r'^tns[_.]')),
    ('desi',      'DESI',           re.compile(r'^desi[_.]')),
    ('agn',       'AGN catalogues', re.compile(r'^agn[_.]')),
]

FALLBACK = ('detection', 'Detection & extraction')

# Served as fields on `sources`, not inside `catalog`, so the schema should not
# claim them.
PROMOTED = {'source_uuid', 'coord_slug', 'ra', 'dec'}

# Build provenance rather than science; useful to expose but worth separating
# so it does not clutter the page.
PROVENANCE = {'catalog_tier', 'row_hash'}


def group_of(column: str) -> str:
    if column in PROVENANCE:
        return 'provenance'
    for key, _, pattern in GROUP_RULES:
        if pattern.match(column):
            return key
    return FALLBACK[0]


LABELS = {key: label for key, label, _ in GROUP_RULES}
LABELS[FALLBACK[0]] = FALLBACK[1]
LABELS['provenance'] = 'Build provenance'


def build_schema(columns: list[str]) -> dict:
    """`{group: {label, columns}}` for the columns a source actually carries."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        if c in PROMOTED:
            continue
        groups.setdefault(group_of(c), []).append(c)

    order = [k for k, _, _ in GROUP_RULES] + [FALLBACK[0], 'provenance']
    return {
        'groups': [
            {'key': k, 'label': LABELS[k], 'n_columns': len(groups[k]),
             'columns': sorted(groups[k])}
            for k in order if k in groups
        ],
        'n_columns': sum(len(v) for v in groups.values()),
    }
