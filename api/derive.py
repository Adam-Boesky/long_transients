"""Values read off the ~300-column catalog row rather than stored beside it.

Shared because two views need the same answer from the same columns: the
table shows one classification string per source, and the classify page
shows the same thing for the source in front of you. Deriving it twice
invites the two disagreeing.
"""
from __future__ import annotations

import orjson

# (display prefix, column, the column holds a JSON list).
#
# SIMBAD is included: we cross-match it and it types sources, so leaving it
# out would make a source SIMBAD calls a quasar read as unclassified. `otype_label`
# rather than `otype` because it is the readable form -- "Quasar", not "QSO".
CLASS_SOURCES: list[tuple[str, str, bool]] = [
    ('SDSS', 'sdss_classes', True),
    ('DESI', 'desi_spectype', False),
    ('TNS', 'tns_type', False),
    ('SIMBAD', 'simbad_otype_label', False),
    ('AGN-DB', 'agn_db_best_class_all', True),
]

# The crossmatch flag saying a catalog knows the source at all, which is a
# different question from whether it typed it. TNS is where the two come
# apart in practice.
MATCHED_COLUMN = {
    'SDSS': 'sdss_matched',
    'DESI': 'desi_matched',
    'TNS': 'tns_matched',
    'SIMBAD': 'simbad_matched',
    'AGN-DB': 'agn_db_matched',
}

# The redshift each catalog carries, where it carries one. Keyed by the same
# display prefix as CLASS_SOURCES so a chip can pair them up without a second
# mapping. AGN-DB has three redshift columns; `best_z_merged` is the numeric
# one it nominates itself (`best_z_all` is a JSON list of per-sub-catalogue
# values). WISE, Gaia and the photometric catalogues have no redshift at all.
REDSHIFT_COLUMN = {
    'DESI': 'desi_z',
    'TNS': 'tns_redshift',
    'SIMBAD': 'simbad_z',
    'AGN-DB': 'agn_db_best_z_merged',
}

# Catalogs that give one redshift per class rather than one per source: SDSS
# keeps every spectrum of a source, and they need not agree. A JSON object,
# class -> median redshift of that class's spectra.
CLASS_REDSHIFT_COLUMN = {
    'SDSS': 'sdss_class_z',
}

# Stands in for "this catalog has the source but has not typed it".
UNCLASSIFIED = 'unknown'

# Cells that mean "absent" but arrive as text.
_BLANK = {'', '<NA>', 'nan', 'None'}


def as_list(value) -> list[str]:
    """Unpack a class cell.

    AGN-DB stores one class per contributing sub-catalog as a JSON list,
    `["unknown", "generic_agn"]`; everything else is a bare string.
    """
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if text in _BLANK:
            return []
        try:
            parsed = orjson.loads(text)
        except orjson.JSONDecodeError:
            return [text]
        items = parsed if isinstance(parsed, list) else [parsed]
    return [str(i).strip() for i in items if str(i).strip() not in _BLANK]


def classes_of(cat: dict) -> dict[str, list[str]]:
    """Display prefix -> the classes this source carries, deduplicated.

    A catalog that matched but did not type the source contributes
    `unknown`, so "TNS has it and said nothing" stays distinguishable from
    "TNS has never heard of it", which is the whole point of the flag.
    """
    out: dict[str, list[str]] = {}
    for prefix, column, is_list in CLASS_SOURCES:
        found = set(as_list(cat.get(column)))
        if not found and _truthy(cat.get(MATCHED_COLUMN[prefix])):
            found = {UNCLASSIFIED}
        if found:
            out[prefix] = sorted(found)
    return out


def redshifts_of(cat: dict) -> dict[str, float]:
    """Display prefix -> the redshift that catalog reports, where it has one.

    Deliberately separate from `classes_of` rather than folded into the class
    strings: the table renders those as a single string and does not want a
    redshift in it, while the classify page pairs the two per chip.

    A catalog can match, and even type the source, without a redshift --
    SIMBAD does it ~6,000 times (stars), TNS for all but 182 of its 1,183
    matches -- so a missing key here is normal and means "no redshift", not
    "no match". Negative values are kept: a few hundred SIMBAD/SDSS/DESI
    entries are blueshifted galactic objects, and hiding them would misreport
    what the catalog says.
    """
    out: dict[str, float] = {}
    for prefix, column in REDSHIFT_COLUMN.items():
        z = _as_float(cat.get(column))
        if z is not None:
            out[prefix] = z
    return out


def class_redshifts_of(cat: dict) -> dict[str, dict[str, float]]:
    """Display prefix -> {class: redshift}, for the `CLASS_REDSHIFT_COLUMN` catalogs."""
    out: dict[str, dict[str, float]] = {}
    for prefix, column in CLASS_REDSHIFT_COLUMN.items():
        value = cat.get(column)
        if value is None or str(value).strip() in _BLANK:
            continue
        try:
            parsed = orjson.loads(str(value))
        except orjson.JSONDecodeError:
            continue
        zs = {str(c): z for c, v in parsed.items() if (z := _as_float(v)) is not None}
        if zs:
            out[prefix] = zs
    return out


def _as_float(value) -> float | None:
    """A redshift cell, which reaches here as a float, None or text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value.strip() in _BLANK:
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        z = float(value)
    except (TypeError, ValueError):
        return None
    # NaN survives the JSON round trip as a float in some builds; it means the
    # catalog had no value, not a redshift of nan.
    return None if z != z else z


def _truthy(value) -> bool:
    """A crossmatch flag, which reaches here as a bool, number or string."""
    if isinstance(value, str):
        return value.strip().lower() in {'true', 't', '1', 'yes'}
    return bool(value)
