"""Response and request shapes.

Kept deliberately thin: the light-curve payload never becomes a pydantic model
because it is served as pre-compressed bytes straight from the database.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Follow-up observing state, in order. NULL is the fourth ("nothing") and is
# not a member: no decision needs no spelling.
FOLLOWUP_STATUSES = ('to queue', 'queued', 'observed')


class SourceSummary(BaseModel):
    """One row of the candidate list."""
    source_uuid: str
    coord_slug: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    n_epochs: int = 0
    surveys: list[str] = Field(default_factory=list)
    n_spectra: int = 0
    spec_surveys: list[str] = Field(default_factory=list)
    label: Optional[str] = None          # this author's label, when asked for
    n_annotations: int = 0               # how many authors have labelled it
    # Which sub-catalogue it came from, in the `catalog=` filter's own terms:
    # 'in_both', 'in_ztf' or 'in_ztf_wide'. None for a tier outside them.
    catalog: Optional[str] = None


class SourceDetail(BaseModel):
    """Everything the detail page needs except the light curves."""
    source_uuid: str
    coord_slug: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    n_epochs: int = 0
    surveys: list[str] = Field(default_factory=list)
    n_spectra: int = 0
    spec_surveys: list[str] = Field(default_factory=list)
    catalog: dict[str, Any]              # the remaining ~300 catalog columns
    # Cross-match classifications, derived from `catalog`: display prefix ->
    # classes. Derived server-side so the table and the classify page cannot
    # disagree about what SIMBAD or AGN-DB said.
    classes: dict[str, list[str]] = Field(default_factory=dict)
    # Same keys as `classes`, for the catalogs that report a redshift. A
    # catalog can appear in `classes` and not here (SIMBAD stars, most TNS
    # rows), so the two are read together rather than assumed parallel.
    redshifts: dict[str, float] = Field(default_factory=dict)
    # Per class, for catalogs whose spectra of a source can disagree (SDSS).
    class_redshifts: dict[str, dict[str, float]] = Field(default_factory=dict)
    annotations: list['Annotation'] = Field(default_factory=list)


class SourcePage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[SourceSummary]
    # How many of the `total` this author has already filed. Only meaningful
    # with `author`; 0 otherwise. The client holds a window rather than the
    # whole list now, so it cannot count this itself.
    n_labelled: int = 0


class Annotation(BaseModel):
    source_uuid: str
    author: str
    label: Optional[str] = None
    confidence: Optional[str] = None
    note: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    status: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AnnotationIn(BaseModel):
    """What the classify buttons send."""
    label: Optional[str] = None
    confidence: Optional[str] = None
    note: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    status: Optional[str] = None

    @field_validator('status')
    @classmethod
    def _known_status(cls, v: Optional[str]) -> Optional[str]:
        """Empty string clears it, so the UI can send one value for 'nothing'."""
        if v is None or v == '':
            return None
        if v not in FOLLOWUP_STATUSES:
            raise ValueError(f'status must be one of {FOLLOWUP_STATUSES} or empty')
        return v


class Label(BaseModel):
    """One entry in the shared label vocabulary."""
    name: str
    created_by: Optional[str] = None
    created_at: str
    color: Optional[str] = None          # '#rrggbb'; None = client's default
    n_sources: int = 0                   # sources anyone has filed under it


class LabelIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    author: Optional[str] = None


class LabelPatch(BaseModel):
    """A new colour for one label; null goes back to the default."""
    color: Optional[str] = Field(None, pattern=r'^#[0-9a-fA-F]{6}$')


class LabelOrder(BaseModel):
    """Every label's name, in the new order."""
    names: list[str]


class Stats(BaseModel):
    n_sources: int
    n_with_lightcurves: int
    n_epochs: int
    n_with_spectra: int = 0
    n_spectra: int = 0
    n_annotated: int
    by_label: dict[str, int]
    by_author: dict[str, int]
    built_at: Optional[str] = None


class MemberProgress(BaseModel):
    """One row of the Progress page's per-author table."""
    author: str
    n_sources: int                       # distinct sources this author filed
    n_labels: int                        # distinct labels they reached for
    n_comments: int                      # of those, how many carry a note


class Verdict(BaseModel):
    """What one author said about one source."""
    author: str
    label: Optional[str] = None
    note: Optional[str] = None


class Disagreement(BaseModel):
    """A source two authors filed under different labels."""
    source_uuid: str
    coord_slug: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    labels: list[str] = Field(default_factory=list)      # the distinct split
    verdicts: list[Verdict] = Field(default_factory=list)


class Progress(BaseModel):
    """Who has labelled what, and where people disagree.

    Only *labelled* annotations count: a bare comment says nothing about which
    bucket a source belongs in, so it is excluded from every count here except
    `n_comments`, which is scoped to the labelled rows for the same reason.
    """
    n_sources: int                       # the whole sample, for share-of
    n_labelled: int                      # distinct sources with any label
    authors: list[str] = Field(default_factory=list)
    by_label: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_member: list[MemberProgress] = Field(default_factory=list)
    n_disagreements: int = 0
    disagreements: list[Disagreement] = Field(default_factory=list)


SourceDetail.model_rebuild()
