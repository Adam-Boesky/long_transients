"""Where the API finds its two databases."""
from __future__ import annotations

import os
from pathlib import Path

from source_db.paths import DEFAULT_OUT_DIR
from source_db.serving.schema import ANNOTATIONS_DB, SCIENCE_DB

# Overridable so a deployment can put the databases anywhere, and so tests can
# point at a throwaway build.
SCIENCE_PATH = Path(os.environ.get('LT_SCIENCE_DB', DEFAULT_OUT_DIR / SCIENCE_DB))
ANNOTATIONS_PATH = Path(os.environ.get('LT_ANNOTATIONS_DB',
                                       DEFAULT_OUT_DIR / ANNOTATIONS_DB))

# Background ztforce runs: a writable, regenerable store of queued and finished
# forced photometry. Kept out of science.sqlite (read-only on the server) and out
# of annotations.sqlite (the irreplaceable master copy of labels).
ZTFORCE_PATH = Path(os.environ.get('LT_ZTFORCE_DB', DEFAULT_OUT_DIR / 'ztforce.sqlite'))
# Sources carrying this annotation label (from any author) are queued for ztforce.
ZTFORCE_LABEL = os.environ.get('LT_ZTFORCE_LABEL', 'Run ztforce')
# Run the queue worker inside the API process. Set to 0 to turn it off (the tests do).
ZTFORCE_WORKER = os.environ.get('LT_ZTFORCE_WORKER', '1') not in ('0', 'false', 'no', '')
# Sources photometered at once. Each already runs 8 parallel downloads.
ZTFORCE_CONCURRENCY = int(os.environ.get('LT_ZTFORCE_CONCURRENCY', '4'))

# The candidate list is paginated; a page of 50 is ~15 kB of JSON.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# The compiled frontend, served by this process when it exists. Absent in dev,
# where Vite serves the page and proxies /api here instead.
WEB_DIST = Path(os.environ.get(
    'LT_WEB_DIST', Path(__file__).resolve().parent.parent / 'web' / 'dist'))

# CORS origins for the React dev server and whatever hosts the built frontend.
# Stripped: `LT_ALLOWED_ORIGINS=https://a.example, https://b.example` is the
# natural thing to write, and an unstripped ' https://b.example' never matches,
# which surfaces as a CORS failure that looks like a frontend bug.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    'LT_ALLOWED_ORIGINS',
    'http://localhost:3000,http://localhost:5173').split(',') if o.strip()]
