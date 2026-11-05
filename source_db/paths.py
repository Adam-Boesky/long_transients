"""Shared locations for the source-database build."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The database itself. One switch for the whole thing: every other path below
# and every database in `api.config` derives from it, so moving the data to an
# external drive is one variable rather than five that must agree. Setting an
# individual `LT_*` path still wins, for the case where (say) annotations stay
# on internal disk.
DEFAULT_OUT_DIR = Path(os.environ.get('LT_DATA_DIR', REPO / 'Data' / 'db'))
# Raw, native-format pulls from each service. Fetched once, never re-fetched;
# normalisation reads from here and is freely re-runnable.
RAW_DIR = DEFAULT_OUT_DIR / 'raw'

# Default inputs: the enriched tables from the current pipeline run.
DEFAULT_COMBINED = (REPO / 'Data'
                    / 'filter_results_final_10_22_2026' / 'combined')
