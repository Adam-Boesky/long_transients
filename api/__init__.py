"""Read API for the candidate-classification website.

Serves the static science tier (`science.sqlite`) and the mutable annotation
tier (`annotations.sqlite`) built by `source_db.serving`. Nothing here owns
data: the science database is a disposable artefact of the build, and this
process only ever reads it.
"""

# Settings are environment variables, read at import time across the package
# (and in source_db), so load `.env` (see `.env.example`) before anything else is
# imported. Variables already in the environment win. LT_ENV_FILE points at
# another file; an empty LT_ENV_FILE loads none (the tests do).
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from dotenv import load_dotenv as _load_dotenv  # noqa: E402

_env_file = _os.environ.get('LT_ENV_FILE', str(_Path(__file__).resolve().parents[1] / '.env'))
if _env_file:
    _load_dotenv(_env_file, override=False)
