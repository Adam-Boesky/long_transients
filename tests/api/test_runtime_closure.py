"""What the deployed website is allowed to import.

The host installs `requirements.txt` -- twelve packages plus ztforce -- and
that list is the *serving* set, not everything the repo uses. Building the
catalog additionally needs pyvo, sparcl, hpgeom and requests, and the modules
that use them live in `source_db` next to the handful the website does need.

Nothing stops a router importing one. It would work here, where the full conda
environment is installed, and fail on the host with `ModuleNotFoundError` at
startup -- after a deploy, on a machine you are not sitting at.

So this pins the boundary: walk every module reachable from `api.main` and
assert it stays inside the serving set. A new import that crosses the line
fails here, with the file that did it, rather than on the server.

Adding to the website is meant to be easy: if the new module genuinely belongs
in the deployed set, add it to `SERVING` and check whether it pulls in a
package `requirements.txt` does not list.
"""
from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

# `source_db` modules the website may import. Everything else in that package
# builds the catalog and never runs on the host.
SERVING = {
    'source_db/__init__.py',
    'source_db/constants.py',
    'source_db/paths.py',
    'source_db/unified_class.py',
    'source_db/cutouts/__init__.py',
    'source_db/cutouts/render.py',          # cutouts are rendered per request
    'source_db/cutouts/epochs.py',          # single-epoch images, per click
    'source_db/serving/__init__.py',
    'source_db/serving/filters.py',
    'source_db/serving/schema.py',
}

# Third-party packages on the host, from requirements.txt. Import names, which
# are not always the distribution names (Pillow -> PIL, python-dotenv -> dotenv).
INSTALLED = {
    'fastapi', 'uvicorn', 'pydantic', 'orjson', 'dotenv', 'duckdb', 'pyarrow',
    'pandas', 'numpy', 'astropy', 'matplotlib', 'PIL', 'ztforce',
    # Pulled in by the above rather than named directly, but present on the
    # host all the same.
    'starlette', 'anyio', 'requests',
}


def _module_path(dotted: str) -> pathlib.Path | None:
    p = REPO / dotted.replace('.', '/')
    for candidate in (p.with_suffix('.py'), p / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _walk_from(entry: str) -> dict[str, set[str]]:
    """Every first-party module reachable from `entry`, and who imports what.

    Returns {relative path: set of third-party top-level packages it imports}.
    """
    found: dict[str, set[str]] = {}
    stack = [REPO / entry]
    while stack:
        path = stack.pop()
        rel = path.relative_to(REPO).as_posix()
        if rel in found:
            continue
        found[rel] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split('.')[0]
                if top in ('api', 'source_db'):
                    sub = _module_path(name)
                    if sub:
                        stack.append(sub)
                    # `from source_db.serving import filters` names a module
                    # in the `import` list, not the `from` clause.
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            deeper = _module_path(f'{name}.{alias.name}')
                            if deeper:
                                stack.append(deeper)
                else:
                    found[rel].add(top)
    return found


def test_the_website_imports_only_serving_modules():
    """A router reaching into the build pipeline breaks the host, not this box."""
    reachable = _walk_from('api/main.py')
    crossed = sorted(m for m in reachable
                     if m.startswith('source_db/') and m not in SERVING)
    assert not crossed, (
        'these build-only modules are now reachable from api.main, so the '
        'deployed host would need the catalog-building dependencies too:\n  '
        + '\n  '.join(crossed))


def test_the_website_imports_only_installed_packages():
    """Every third-party import must be something requirements.txt installs."""
    import sys

    reachable = _walk_from('api/main.py')
    missing: dict[str, set[str]] = {}
    for module, third_party in reachable.items():
        unlisted = {p for p in third_party
                    if p not in INSTALLED and p not in sys.stdlib_module_names}
        if unlisted:
            missing[module] = unlisted
    assert not missing, (
        'imported but not in requirements.txt, so the host would fail at '
        'startup:\n  '
        + '\n  '.join(f'{m}: {sorted(p)}' for m, p in sorted(missing.items())))


def test_the_serving_list_has_no_stale_entries():
    """Keeps `SERVING` honest: a module deleted or no longer used drops out."""
    reachable = set(_walk_from('api/main.py'))
    stale = sorted(m for m in SERVING
                   if m not in reachable and not (REPO / m).read_text().strip() == '')
    # `__init__.py` files are executed by the import machinery rather than
    # named in an import statement, so they never appear in the walk.
    stale = [m for m in stale if not m.endswith('__init__.py')]
    assert not stale, (
        f'listed as serving modules but nothing reaches them: {stale}')
