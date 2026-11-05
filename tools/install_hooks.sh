#!/bin/sh
# Install this repo's git hooks beside the Git LFS ones. Safe to rerun.
#
#   post-commit: push the paper figures the commit changed to Overleaf
#                (tools/push_figures.py --committed; a no-op where there is no paper clone).
set -e
ROOT=$(git rev-parse --show-toplevel)
HOOK="$ROOT/.git/hooks/post-commit"
MARK='# long_transients: push paper figures'
if grep -qF "$MARK" "$HOOK" 2>/dev/null; then
  echo "post-commit hook already installed"
  exit 0
fi
[ -f "$HOOK" ] || printf '#!/bin/sh\n' > "$HOOK"
cat >> "$HOOK" <<HOOKEOF

$MARK
# Overleaf is best-effort: a failed push must never look like a failed commit.
PY=\$(command -v conda >/dev/null 2>&1 && echo "conda run -n long_transients python" || echo python3)
\$PY "$ROOT/tools/push_figures.py" --committed || echo "post-commit: pushing paper figures to Overleaf failed" >&2
HOOKEOF
chmod +x "$HOOK"
echo "installed post-commit hook in $HOOK"
