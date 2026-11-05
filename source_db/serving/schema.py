"""Schemas for the two serving databases.

Two files, not one, and the split is the point: `science.sqlite` is a static
artefact rebuilt wholesale from the analysis tiers, while `annotations.sqlite`
is the only mutable state and must survive every rebuild. They are joined by
`source_uuid`, which is exactly why the identity map was worth building.
"""
from __future__ import annotations

# ── science.sqlite (read-only in production) ──────────────────────────────────
#
# `sources` carries the handful of columns the candidate list filters and sorts
# on as real columns, and the remaining ~300 as one JSON blob. Promoting every
# column would make a 305-column table that SQLite handles poorly and nothing
# queries; keeping them reachable means the detail page still shows everything.
SCIENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_uuid   TEXT PRIMARY KEY,
    coord_slug    TEXT,
    ra            REAL,
    dec           REAL,
    n_epochs      INTEGER NOT NULL DEFAULT 0,
    surveys       TEXT,
    n_spectra     INTEGER NOT NULL DEFAULT 0,
    spec_surveys  TEXT,
    -- Promoted out of the blob because the classify page filters on it by
    -- default. Left inside, `json_extract(catalog_json, '$.catalog_tier')`
    -- is a full scan that has to inflate ~300 columns of JSON per row: 4.8 s
    -- over 198k sources, paid on every windowed page request.
    catalog_tier  TEXT,
    catalog_json  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_radec ON sources(ra, dec);
CREATE INDEX IF NOT EXISTS idx_sources_slug  ON sources(coord_slug);
CREATE INDEX IF NOT EXISTS idx_sources_tier  ON sources(catalog_tier);
-- The list's default sort is by uuid, so the composite lets a filtered
-- page be an ordered index walk instead of sorting 187k rows.
CREATE INDEX IF NOT EXISTS idx_sources_tier_uuid
    ON sources(catalog_tier, source_uuid);

-- One row per source. `payload` is gzipped JSON in exactly the shape the
-- frontend charts consume, so a page load is one primary-key fetch and the
-- API hands the bytes straight to the client with Content-Encoding: gzip.
CREATE TABLE IF NOT EXISTS light_curves (
    source_uuid  TEXT PRIMARY KEY,
    n_epochs     INTEGER NOT NULL,
    surveys      TEXT NOT NULL,
    payload      BLOB NOT NULL
);

-- One row per source that has any spectrum, payload gzipped like the light
-- curves. Only ~34% of sources have one, so this table is much shorter than
-- `sources` and a missing row means "no spectrum", not an error.
CREATE TABLE IF NOT EXISTS spectra (
    source_uuid  TEXT PRIMARY KEY,
    surveys      TEXT NOT NULL,
    n_spectra    INTEGER NOT NULL,
    payload      BLOB NOT NULL
);

-- Cutout *addresses*, not pixels: one row per (source, survey, band), and a
-- row exists if and only if that stamp can be made. Availability is row
-- presence rather than a flag alongside the address, so there is no second
-- thing to keep in sync -- which is what the old per-source JSON blob got
-- wrong, since whichever writer touched it last erased the other survey.
--
-- Both surveys are resolved offline into `raw/` and read from there at build
-- time, for the same reason in mirrored forms: a PanSTARRS stack filename
-- cannot be derived from a position, and whether ZTF ever built a reference
-- image for a band cannot be derived from its pointing. `origin` is whatever
-- locates the pixels -- the stack filename, or `field/ccd/quadrant`.
CREATE TABLE IF NOT EXISTS cutout_sources (
    source_uuid  TEXT NOT NULL,
    survey       TEXT NOT NULL,
    band         TEXT NOT NULL,
    origin       TEXT NOT NULL,
    PRIMARY KEY (source_uuid, survey, band)
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

# ── annotations.sqlite (read-write) ───────────────────────────────────────────
#
# One row per (source, author) so the three of you can disagree, which is the
# useful state for building a labelled set. `history` keeps every edit, because
# the point of the exercise is to analyse the labelling later.
#
# `status` is the follow-up observing state, separate from `label`: the label
# says what the source is, the status where it stands in the queue. NULL means
# untouched. Vocabulary in `api.models.FOLLOWUP_STATUSES`.
ANNOTATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    source_uuid  TEXT NOT NULL,
    author       TEXT NOT NULL,
    label        TEXT,
    confidence   TEXT,
    note         TEXT,
    tags         TEXT,
    status       TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (source_uuid, author)
);
CREATE INDEX IF NOT EXISTS idx_ann_source ON annotations(source_uuid);
CREATE INDEX IF NOT EXISTS idx_ann_author ON annotations(author);
CREATE INDEX IF NOT EXISTS idx_ann_label  ON annotations(label);
CREATE INDEX IF NOT EXISTS idx_ann_status ON annotations(status);

CREATE TABLE IF NOT EXISTS history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uuid  TEXT NOT NULL,
    author       TEXT NOT NULL,
    label        TEXT,
    confidence   TEXT,
    note         TEXT,
    tags         TEXT,
    status       TEXT,
    recorded_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_source ON history(source_uuid);

-- The label vocabulary, shared by everyone so the buckets mean the same thing
-- to all of you. Anyone can add one or drag them into a new order; `position`
-- is that order, and a label nobody has placed yet (NULL) sorts last, oldest
-- first. `color` is '#rrggbb', or NULL for the one the client derives from
-- the name. NOCASE so "trash" cannot sit beside "Trash".
-- (`api.db` adds the later columns to a labels table that predates them.)
CREATE TABLE IF NOT EXISTS labels (
    name         TEXT PRIMARY KEY COLLATE NOCASE,
    created_by   TEXT,
    created_at   TEXT NOT NULL,
    position     INTEGER,
    color        TEXT
);
INSERT OR IGNORE INTO labels (name, created_by, created_at) VALUES
    ('Trash',     NULL, '1970-01-01T00:00:00+00:00'),
    ('Follow-up', NULL, '1970-01-01T00:00:01+00:00');
"""

SCIENCE_DB = 'science.sqlite'
ANNOTATIONS_DB = 'annotations.sqlite'

# Columns promoted out of the 305-column catalog row onto `sources`, chosen
# because the candidate list filters or sorts on them.
LIST_COLUMNS = ['coord_slug', 'ra', 'dec']
