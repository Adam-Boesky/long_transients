"""The API the React frontend talks to.

    conda run -n long_transients uvicorn api.main:app --reload

Two databases behind it, and the split is deliberate: `science.sqlite` is a
read-only artefact rebuilt from the analysis tiers, `annotations.sqlite` is the
only thing that accumulates state. They are joined by `source_uuid`. A third,
`ztforce.sqlite`, holds the regenerable ztforce queue and its results.
"""
from __future__ import annotations

import contextlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import catalog_filters
from api import ztforce_worker
from api.config import (
    ALLOWED_ORIGINS,
    ANNOTATIONS_PATH,
    SCIENCE_PATH,
    ZTFORCE_CONCURRENCY,
    ZTFORCE_LABEL,
    ZTFORCE_PATH,
    ZTFORCE_WORKER,
    WEB_DIST,
)
from api.db import connect_annotations, connect_science
from api.routers import annotations, cutouts, filters, sources, table, ztforce


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Start the catalog filter engine before the first request.

    It reads 198k rows of four columns out of `science.sqlite` to attach the
    serving-only columns, which takes ~4 s. Doing it here rather than lazily
    means no user wears that wait, and a catalog that cannot be opened fails
    the process at boot instead of failing the candidate list at 2am.
    """
    con = connect_science(SCIENCE_PATH)
    try:
        catalog_filters.startup(con)
    finally:
        con.close()
    # Apply the annotations schema and its migrations here too. They used to
    # run inside whichever request connected first, so a failure showed up as
    # a 500 on an arbitrary endpoint instead of a process that refuses to boot
    # -- the opposite of the rule stated above.
    connect_annotations(ANNOTATIONS_PATH).close()
    # The ztforce queue worker, unless it runs as its own process (or not at all).
    worker = None
    if ZTFORCE_WORKER:
        worker = ztforce_worker.Worker(
            ZTFORCE_PATH, ANNOTATIONS_PATH, SCIENCE_PATH, ZTFORCE_LABEL, ZTFORCE_CONCURRENCY
        )
        worker.start()
    try:
        yield
    finally:
        if worker is not None:
            worker.stop()
        catalog_filters.shutdown()


app = FastAPI(
    title='long_transients candidate API',
    description='Static science data plus collaborator annotations.',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(sources.router, prefix='/api', tags=['sources'])
app.include_router(filters.router, prefix='/api', tags=['filters'])
app.include_router(cutouts.router, prefix='/api', tags=['cutouts'])
app.include_router(table.router, prefix='/api', tags=['table'])
app.include_router(annotations.router, prefix='/api', tags=['annotations'])
app.include_router(ztforce.router, prefix='/api', tags=['ztforce'])


@app.get('/api/health', tags=['meta'])
def health():
    return {
        'status': 'ok',
        'science_db': str(SCIENCE_PATH),
        'science_db_present': SCIENCE_PATH.exists(),
        'annotations_db': str(ANNOTATIONS_PATH),
        'ztforce_db': str(ZTFORCE_PATH),
        'ztforce_worker': ZTFORCE_WORKER,
        'serving_frontend': (WEB_DIST / 'index.html').is_file(),
    }


def _mount_frontend() -> None:
    """Serve the compiled site from this process, if it has been built.

    In development Vite serves the page and proxies `/api` here, so there are
    two processes and two ports. Deployed there is one of each: `npm run
    build` writes `web/dist`, and these routes hand it out next to the data.
    The frontend addresses the API with relative URLs, so same-origin also
    means CORS never enters into it.

    Registered last, after every router, because the catch-all below matches
    any path and would otherwise shadow them.
    """
    index = WEB_DIST / 'index.html'
    if not index.is_file():
        return                       # dev, or a deploy that forgot to build

    root = WEB_DIST.resolve()

    @app.get('/{path:path}', include_in_schema=False)
    def spa(path: str):
        # `/api` is this process's own namespace: an unknown endpoint under it
        # is a 404. Falling through to the page instead would answer a
        # mistyped API call with 200 and a lump of HTML, which surfaces much
        # later as a JSON parse error pointing at the wrong thing entirely.
        if path == 'api' or path.startswith('api/'):
            raise HTTPException(404, 'no such endpoint')

        target = (root / path).resolve()
        # `is_relative_to` rather than trusting the path: `..%2f..%2fetc`
        # arrives here already decoded, and without this it would be served.
        if path and target.is_file() and target.is_relative_to(root):
            # Vite fingerprints everything under assets/ with a content hash,
            # so those URLs can never mean different bytes later. index.html
            # is the opposite -- it names the current bundle, and a cached
            # copy would pin the browser to a bundle that no longer exists.
            immutable = target.parent == root / 'assets'
            return FileResponse(target, headers={
                'cache-control': 'public, max-age=31536000, immutable'
                                 if immutable else 'no-cache',
            })

        # Anything else is a client-side route (`/table`, `/classify`): the
        # router resolves it once the page loads. Without this, opening or
        # refreshing any URL but `/` would 404.
        return FileResponse(index, headers={'cache-control': 'no-cache'})


_mount_frontend()
