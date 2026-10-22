# Deploy

Laptop → external drive → host machine → ngrok.
`$DRIVE` is the mount point: `/Volumes/LT` (macOS), `/mnt/lt` (Linux).

---

## Steps

**1 — Build, on the laptop**

```sh
cd web && npm run build
./lt down                 # flushes annotations.sqlite's write-ahead log
```

**2 — Copy code (~3 MB)**

```sh
rsync -a --exclude __pycache__ api source_db requirements.txt .env.example $DRIVE/lt/
rsync -aR web/dist $DRIVE/lt/
```

Keep `api/` and `web/` as siblings; the API locates the pages relative to itself.

**3 — Copy data (~10 GB)**

```sh
mkdir -p $DRIVE/lt/db
rsync -a Data/db/{science.sqlite,sources.parquet,wise_colors.parquet,annotations.sqlite,ztforce.sqlite,release.json} $DRIVE/lt/db/
rsync -a Data/followup/ $DRIVE/lt/db/followup/      # finished ztforce photometry (~33 MB)
```

Skip `raw/`, `light_curves/`, `cutout_cache/`, `identity.parquet` — 14 GB the
server never opens.

**4 — Install, on the host**

Python 3.10–3.12. Not 3.13: numpy 1.26 has no wheels for it.
No Node needed — `web/dist` is already compiled.

```sh
cd $DRIVE/lt
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**5 — Configure**

Write `$DRIVE/lt/.env` — see [Settings](#settings).

**6 — Run**

```sh
uvicorn api.main:app --host 127.0.0.1 --port 8000
ngrok http 8000
```

Keep `127.0.0.1`. ngrok connects locally, so nothing else on the network can
reach the server directly.

---

## Settings

All settings are environment variables. They are read from `$DRIVE/lt/.env` at
startup; anything already in the environment wins. `.env.example` is a
commented template.

### Required

| Variable | Meaning |
|---|---|
| `LT_DATA_DIR` | Absolute path to the folder from step 3. Every path below except `LT_ZTFORCE_CACHE` defaults to a file inside it. |
| `LT_ZTFORCE_CACHE` | Where ztforce photometry is written. Defaults beside the code, not to the drive — see below. |

### Credentials

Only needed to run ztforce on the host. Without them ztforce jobs fail; the
rest of the site is unaffected.

| Variable | Meaning |
|---|---|
| `ZTFORCE_IRSA_USER` | IRSA login, for downloading ZTF science images |
| `ZTFORCE_IRSA_PASS` | |

### Paths — all default inside `LT_DATA_DIR`

| Variable | Default |
|---|---|
| `LT_SCIENCE_DB` | `$LT_DATA_DIR/science.sqlite` |
| `LT_ANNOTATIONS_DB` | `$LT_DATA_DIR/annotations.sqlite` |
| `LT_ZTFORCE_DB` | `$LT_DATA_DIR/ztforce.sqlite` |
| `LT_CATALOG_PARQUET` | `$LT_DATA_DIR/sources.parquet` |
| `LT_CUTOUT_CACHE` | `$LT_DATA_DIR/cutout_cache` |
| `LT_WEB_DIST` | `web/dist`, beside `api/` |
| `LT_ENV_FILE` | the `.env` above; empty loads none |

> `LT_ZTFORCE_CACHE` is the exception: it holds finished ztforce photometry,
> grows without limit, and defaults to `Data/followup/ztforce_photometry`
> *beside the code* rather than inside `LT_DATA_DIR`. Set it explicitly on the
> host so results land on the drive:
> `LT_ZTFORCE_CACHE=$DRIVE/lt/db/followup/ztforce_photometry`

### Tuning

| Variable | Default | Meaning |
|---|---|---|
| `LT_ZTFORCE_WORKER` | `1` | Run the queue worker in-process. `0` to disable. One process per ztforce database. |
| `LT_ZTFORCE_LABEL` | `Run ztforce` | Label that queues a source |
| `LT_ZTFORCE_CONCURRENCY` | `4` | Sources at once; each opens 8 IRSA downloads |
| `LT_CUTOUT_MAX_GB` | `20` | Cutout cache budget. `0` disables the sweep. |
| `LT_FILTER_THREADS` | `4` | DuckDB threads |
| `LT_FILTER_MEMORY_LIMIT` | `2GB` | DuckDB memory ceiling |
| `LT_ALLOWED_ORIGINS` | localhost | CORS. Unused when the API serves the pages, as here. |

---

## Verify

```sh
curl -s localhost:8000/api/health      # expect serving_frontend: true
```

Then open the ngrok URL: the table should list sources, and a classify page
should show cutouts and a light curve.

---

## Updating

| Changed | Do |
|---|---|
| Frontend | `npm run build`, re-copy `web/dist` |
| Backend | re-copy `api/` and `source_db/`, restart |
| Catalog rebuilt | re-copy `science.sqlite` and both `.parquet` |

> **Never re-copy `annotations.sqlite` from laptop to host.**
> After step 3 the host's copy is the live master — it is the only file holding
> labels and comments, and overwriting it destroys every verdict made there.
> Copy it host → laptop to back up, with the server stopped.

---

## Gotchas

- **Mount the drive before starting.** On an empty mountpoint SQLite silently
  creates a blank annotations database, and the real one reappears underneath
  it when the drive arrives. On systemd, set `RequiresMountsFor` in
  `lt-api.service`.
- **USB or internal disk only.** SQLite locking is unsafe over SMB/NFS.
- **The host needs internet** — cutouts are fetched from IRSA and PanSTARRS per
  request, not shipped with the data.
- **Free ngrok issues a new URL per restart.** One static domain is included;
  claim it in the dashboard, then `ngrok http 8000 --url=https://you.ngrok-free.app`.
- **There is no authentication.** Anyone with the URL can label, comment and
  delete buckets. `ngrok http 8000 --basic-auth "user:pass"` is the cheapest fix.
