"""Shared coordinate ↔ filename-stem encoding used across the project.

Convention
----------
  - Decimal point → 'p'
  - Minus sign    → 'n' (prefix)
  - 4 decimal places, always (no trailing-zero stripping)

Examples
--------
  coord_stem(126.2749, 26.706)   → '126p2749_26p7060'
  coord_stem(204.8789, -12.7908) → '204p8789_n12p7908'
  parse_coord_stem('126p2749_26p7060')   → (126.2749, 26.706)
  parse_coord_stem('126p27_26p706')      → (126.27, 26.706)   # manual names ok
"""

from __future__ import annotations

import re
from typing import Optional


def coord_stem(ra: float, dec: float) -> str:
    """Encode (ra, dec) as a filename stem at 4 decimal-place precision."""
    ra_str  = f"{ra:.4f}".replace(".", "p").replace("-", "n")
    dec_str = f"{dec:.4f}".replace(".", "p").replace("-", "n")
    return f"{ra_str}_{dec_str}"


# Matches an optional leading numeric ID (e.g. "112_"), then the RA and Dec parts.
# Works on stems with any decimal precision and an optional trailing suffix
# (e.g. "_spectrum").
_COORD_RE = re.compile(
    r"(?:^\d+_)?"        # optional leading "ID_"
    r"(n?[\d]+p[\d]+)"   # RA part
    r"_"
    r"(n?[\d]+p[\d]+)"   # Dec part
)


def _decode_part(s: str) -> float:
    negative = s.startswith("n")
    return (-1 if negative else 1) * float(s.lstrip("n").replace("p", "."))


def parse_coord_stem(stem: str) -> Optional[tuple[float, float]]:
    """Return (ra, dec) in degrees from a coordinate-encoded filename stem, or None."""
    m = _COORD_RE.search(stem)
    if not m:
        return None
    return _decode_part(m.group(1)), _decode_part(m.group(2))
