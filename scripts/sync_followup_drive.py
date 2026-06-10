"""Sync follow-up data to Google Drive and update the status spreadsheet.

One-time setup
--------------
1. Create a Google Cloud project, enable the Drive and Sheets APIs.
2. Create OAuth 2.0 credentials (Desktop app), download the JSON file, and
   save it to ~/vault/gdrive_client_secret.json.
3. Create a Drive folder and a Google Sheet for follow-up data, then add their
   IDs to ~/vault/gdrive_followup.txt (two lines: DRIVE_FOLDER_ID, SHEET_ID).
4. Run once interactively so the browser OAuth flow can complete.  The token
   is saved to ~/vault/gdrive_token.json and reused on subsequent runs.

Drive layout managed by this script
------------------------------------
  Analysis Pages/
    in_both/          ← kde_analysis_pages/in_both/
    in_ztf/           ← kde_analysis_pages/in_ztf/
    in_panstarrs/     ← kde_analysis_pages/in_pstarr/
    0_flowchart.pdf   ┐
    1_flowchart.pdf   ├ filter_results_kde/combined/*_flowchart.pdf
    2_flowchart.pdf   ┘
  Spectra/            ← followup/spectra/plots/

Spreadsheet columns
--------------------
  RA (deg) | Dec (deg) | Analysis Page | Spectrum | Redshift | Classification

Redshifts are read from followup/spectra/redshifts.csv (add rows there when
new spectra are reduced). Classification is left blank for manual entry.

Usage
-----
    python scripts/sync_followup_drive.py

The script is idempotent: re-running it uploads only new files and upserts
spreadsheet rows without duplicating existing entries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def log(msg: str) -> None:
    print(msg, flush=True)


# ── configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Source_Analysis.coord_utils import parse_coord_stem  # noqa: E402

VAULT = Path.home() / "vault"

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
GDRIVE_CREDENTIALS_FILE = str(VAULT / "gdrive_client_secret.json")
TOKEN_FILE               = str(VAULT / "gdrive_token.json")

_ids = np.genfromtxt(VAULT / "gdrive_followup.txt", dtype="str")
DRIVE_FOLDER_ID: str = str(_ids[0])
SHEET_ID:        str = str(_ids[1])

SHEET_TAB = "Follow-up Status"
QUEUE_TAB = "Queue"
QUEUE_HEADERS = ["RA (deg)", "Dec (deg)", "Analysis Page", "Notes"]

# Local data paths
ANALYSIS_PAGES_DIR = REPO_ROOT / "Data" / "kde_analysis_pages"
FLOWCHARTS_DIR     = REPO_ROOT / "Data" / "filter_results_kde" / "combined"
SPECTRA_PLOTS_DIR  = REPO_ROOT / "Data" / "followup" / "spectra" / "plots"
REDSHIFTS_CSV      = REPO_ROOT / "Data" / "followup" / "spectra" / "redshifts.csv"

# Analysis-pages subdirectories: local name → Drive folder name
AP_SUBDIRS = {
    "in_both":   "in_both",
    "in_ztf":    "in_ztf",
    "in_pstarr": "in_panstarrs",
}

# Match radius for pairing analysis pages with manually-named spectrum files.
# Generated filenames use 4 d.p. (~0.36 arcsec precision); hand-typed spectrum
# filenames may have only 2 d.p., causing up to ~16 arcsec apparent separation
# for the same source. 30 arcsec is safely above that without risking false matches.
MATCH_RADIUS_ARCSEC = 30.0

SHEET_HEADERS = ["RA (deg)", "Dec (deg)", "Analysis Page", "Spectrum", "Redshift", "Classification"]


# ── redshift lookup ────────────────────────────────────────────────────────────

def load_redshifts() -> dict[str, float]:
    """Load redshifts from redshifts.csv. Returns {filename_stem: redshift}."""
    df = pd.read_csv(REDSHIFTS_CSV)
    return dict(zip(df["filename_stem"].astype(str), df["redshift"].astype(float)))


def build_redshift_map(
    analysis_stems: list[str],
    redshift_dict: dict[str, float],
) -> dict[str, Optional[float]]:
    """Match each analysis-page stem to a redshift by sky position."""
    a_coords, a_valid = [], []
    for s in analysis_stems:
        c = parse_coord_stem(s)
        if c:
            a_valid.append(s)
            a_coords.append(c)

    sp_stems = list(redshift_dict.keys())
    sp_coords, sp_valid = [], []
    for s in sp_stems:
        c = parse_coord_stem(s)
        if c:
            sp_valid.append(s)
            sp_coords.append(c)

    result: dict[str, Optional[float]] = {s: None for s in analysis_stems}
    if not a_valid or not sp_valid:
        return result

    a_sky  = SkyCoord(ra=[c[0] for c in a_coords]  * u.deg, dec=[c[1] for c in a_coords]  * u.deg)
    sp_sky = SkyCoord(ra=[c[0] for c in sp_coords] * u.deg, dec=[c[1] for c in sp_coords] * u.deg)

    idx, sep, _ = a_sky.match_to_catalog_sky(sp_sky)
    matched = 0
    for i, (stem, separation) in enumerate(zip(a_valid, sep)):
        if separation.arcsec <= MATCH_RADIUS_ARCSEC:
            result[stem] = redshift_dict[sp_valid[idx[i]]]
            matched += 1

    log(f"  {matched}/{len(a_valid)} analysis pages matched to a redshift")
    return result


# ── Google auth ────────────────────────────────────────────────────────────────

def _get_credentials() -> Credentials:
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log("  Refreshing expired token…")
            creds.refresh(Request())
        else:
            log("  No valid token — opening browser for OAuth flow…")
            flow = InstalledAppFlow.from_client_secrets_file(GDRIVE_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(TOKEN_FILE).write_text(creds.to_json())
        log(f"  Token saved to {TOKEN_FILE}")
    else:
        log(f"  Using cached token from {TOKEN_FILE}")
    return creds


# ── Drive helpers ──────────────────────────────────────────────────────────────

def _get_or_create_folder(drive, name: str, parent_id: str) -> str:
    q = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
        f" and '{parent_id}' in parents and trashed=false"
    )
    results = drive.files().list(q=q, fields="files(id)").execute()
    items = results.get("files", [])
    if items:
        return items[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = drive.files().create(body=meta, fields="id").execute()
    log(f"    Created Drive folder '{name}'")
    return folder["id"]


def _list_drive_files(drive, folder_id: str) -> dict[str, str]:
    """Return {filename: file_id} for all non-folder files in a Drive folder."""
    q = f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'"
    results = drive.files().list(q=q, fields="files(id, name)", pageSize=1000).execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


def _list_drive_files_recursive(drive, folder_id: str) -> dict[str, str]:
    """Return {stem: drive_url} for every PDF in a Drive folder tree (recursive)."""
    stem_to_url: dict[str, str] = {}

    # Files at this level
    files = _list_drive_files(drive, folder_id)
    for name, fid in files.items():
        if name.lower().endswith(".pdf"):
            stem_to_url[Path(name).stem] = _drive_url(fid)

    # Recurse into subfolders
    q = f"'{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
    results = drive.files().list(q=q, fields="files(id, name)", pageSize=1000).execute()
    for subfolder in results.get("files", []):
        stem_to_url.update(_list_drive_files_recursive(drive, subfolder["id"]))

    return stem_to_url


def _set_anyone_reader(drive, file_id: str) -> None:
    """Grant 'anyone with the link can view' on a Drive file."""
    drive.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()


def _upload_file(drive, local_path: Path, folder_id: str) -> str:
    meta = {"name": local_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=True)
    f = drive.files().create(body=meta, media_body=media, fields="id").execute()
    file_id = f["id"]
    _set_anyone_reader(drive, file_id)
    return file_id


def _ensure_folder_permissions(drive, folder_id: str) -> None:
    """Set 'anyone with the link can view' on all files already in a Drive folder."""
    q = f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'"
    results = drive.files().list(q=q, fields="files(id, name)", pageSize=1000).execute()
    files = results.get("files", [])
    for f in files:
        _set_anyone_reader(drive, f["id"])
    if files:
        log(f"    Set public link permission on {len(files)} existing file(s)")


def _drive_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def sync_pdfs(drive, local_dir: Path, drive_folder_id: str, label: str = "") -> dict[str, str]:
    """Upload new PDFs from local_dir to Drive. Returns {stem: drive_url}."""
    existing = _list_drive_files(drive, drive_folder_id)
    stem_to_url = {Path(fname).stem: _drive_url(fid) for fname, fid in existing.items()}

    pdfs = sorted(local_dir.glob("*.pdf"))
    to_upload = [p for p in pdfs if p.name not in existing]
    tag = f"[{label}] " if label else ""
    log(f"  {tag}{len(existing)} already on Drive, {len(to_upload)} new to upload")

    for i, pdf in enumerate(to_upload, 1):
        log(f"    [{i}/{len(to_upload)}] Uploading {pdf.name}…")
        fid = _upload_file(drive, pdf, drive_folder_id)
        stem_to_url[pdf.stem] = _drive_url(fid)

    return stem_to_url


# ── Sheets helpers ─────────────────────────────────────────────────────────────

def _ensure_sheet_tab(sheets, sheet_id: str, tab: str) -> None:
    meta = sheets.spreadsheets().get(spreadsheetId=sheet_id).execute()
    names = [s["properties"]["title"] for s in meta["sheets"]]
    if tab not in names:
        body = {"requests": [{"addSheet": {"properties": {"title": tab}}}]}
        sheets.spreadsheets().batchUpdate(spreadsheetId=sheet_id, body=body).execute()
        log(f"  Created sheet tab '{tab}'")
    else:
        log(f"  Sheet tab '{tab}' already exists")


def _read_sheet(sheets, sheet_id: str, tab: str) -> list[list[str]]:
    result = sheets.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=f"'{tab}'"
    ).execute()
    return result.get("values", [])


def _write_sheet(sheets, sheet_id: str, tab: str, rows: list[list]) -> None:
    sheets.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f"'{tab}'!A1",
        valueInputOption="USER_ENTERED",
        body={"values": rows},
    ).execute()


def _append_sheet(sheets, sheet_id: str, tab: str, rows: list[list]) -> None:
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=f"'{tab}'!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()


# ── sync steps ────────────────────────────────────────────────────────────────

def sync_analysis_pages(drive) -> dict[str, str]:
    """Upload analysis pages and flowcharts to Drive. Returns {stem: url}."""
    ap_root_id = _get_or_create_folder(drive, "Analysis Pages", DRIVE_FOLDER_ID)
    all_ap_urls: dict[str, str] = {}

    for local_name, drive_name in AP_SUBDIRS.items():
        local_dir = ANALYSIS_PAGES_DIR / local_name
        if not local_dir.exists():
            log(f"  WARNING: local dir not found, skipping: {local_dir}")
            continue
        log(f"  {local_name}/ → {drive_name}/")
        subdir_id = _get_or_create_folder(drive, drive_name, ap_root_id)
        urls = sync_pdfs(drive, local_dir, subdir_id, label=drive_name)
        _ensure_folder_permissions(drive, subdir_id)
        all_ap_urls.update(urls)

    log(f"  Total analysis pages tracked: {len(all_ap_urls)}")

    log("Syncing flowchart PDFs to Analysis Pages/ root…")
    flowcharts = sorted(FLOWCHARTS_DIR.glob("*_flowchart.pdf"))
    existing_root = _list_drive_files(drive, ap_root_id)
    for pdf in flowcharts:
        if pdf.name in existing_root:
            log(f"  [skip] {pdf.name} already on Drive")
            continue
        log(f"  Uploading {pdf.name}…")
        _upload_file(drive, pdf, ap_root_id)

    return all_ap_urls


def sync_spectra(drive) -> dict[str, str]:
    """Upload spectrum plots to Drive. Returns {stem: url}."""
    sp_root_id = _get_or_create_folder(drive, "Spectra", DRIVE_FOLDER_ID)
    log(f"Syncing spectra plots ({SPECTRA_PLOTS_DIR})…")
    sp_urls = sync_pdfs(drive, SPECTRA_PLOTS_DIR, sp_root_id, label="Spectra")
    _ensure_folder_permissions(drive, sp_root_id)
    return sp_urls


def _fetch_existing_urls(drive) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Read current Drive file lists without uploading anything.

    Returns (current_ap_urls, legacy_ap_urls, sp_urls).
    """
    ap_root_id  = _get_or_create_folder(drive, "Analysis Pages", DRIVE_FOLDER_ID)
    sp_root_id  = _get_or_create_folder(drive, "Spectra",        DRIVE_FOLDER_ID)
    leg_root_id = _get_or_create_folder(drive, "Legacy",         DRIVE_FOLDER_ID)

    all_ap_urls: dict[str, str] = {}
    for drive_name in AP_SUBDIRS.values():
        subdir_id = _get_or_create_folder(drive, drive_name, ap_root_id)
        existing = _list_drive_files(drive, subdir_id)
        all_ap_urls.update({Path(f).stem: _drive_url(fid) for f, fid in existing.items()})

    log("  Scanning Legacy/ folder tree for analysis pages…")
    legacy_ap_urls = _list_drive_files_recursive(drive, leg_root_id)
    log(f"  {len(legacy_ap_urls)} legacy files found")

    sp_existing = _list_drive_files(drive, sp_root_id)
    sp_urls = {Path(f).stem: _drive_url(fid) for f, fid in sp_existing.items()}

    return all_ap_urls, legacy_ap_urls, sp_urls


def _find_ap_url(
    ra: float, dec: float,
    all_ap_urls: dict[str, str],
    legacy_ap_urls: dict[str, str],
) -> str:
    """Return the best analysis page URL for (ra, dec), checking current then legacy."""
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for urls in (all_ap_urls, legacy_ap_urls):
        valid = [(s, c) for s in urls if (c := parse_coord_stem(s))]
        if not valid:
            continue
        cat_sky = SkyCoord(
            ra=[c[0] for _, c in valid] * u.deg,
            dec=[c[1] for _, c in valid] * u.deg,
        )
        sep = target.separation(cat_sky)
        idx = int(sep.arcsec.argmin())
        if sep.arcsec[idx] <= MATCH_RADIUS_ARCSEC:
            return urls[valid[idx][0]]
    return ""


def _match_stems_to_catalog(
    query_stems: list[str],
    catalog_stems: list[str],
) -> dict[str, Optional[str]]:
    """For each query stem, return the closest catalog stem within MATCH_RADIUS_ARCSEC, or None."""
    q_coords, q_valid = [], []
    for s in query_stems:
        c = parse_coord_stem(s)
        if c:
            q_valid.append(s)
            q_coords.append(c)

    cat_coords, cat_valid = [], []
    for s in catalog_stems:
        c = parse_coord_stem(s)
        if c:
            cat_valid.append(s)
            cat_coords.append(c)

    result: dict[str, Optional[str]] = {s: None for s in query_stems}
    if not q_valid or not cat_valid:
        return result

    q_sky   = SkyCoord(ra=[c[0] for c in q_coords]   * u.deg, dec=[c[1] for c in q_coords]   * u.deg)
    cat_sky = SkyCoord(ra=[c[0] for c in cat_coords]  * u.deg, dec=[c[1] for c in cat_coords] * u.deg)
    idx, sep, _ = q_sky.match_to_catalog_sky(cat_sky)
    for i, (stem, separation) in enumerate(zip(q_valid, sep)):
        if separation.arcsec <= MATCH_RADIUS_ARCSEC:
            result[stem] = cat_valid[idx[i]]
    return result


def sync_sheet(sheets, all_ap_urls: dict[str, str], legacy_ap_urls: dict[str, str],
               sp_urls: dict[str, str], redshift_dict: dict[str, float]) -> None:
    """Rebuild the follow-up status spreadsheet.

    Iterates over spectrum sources as primary. For each, looks for a matching
    analysis page first in current Analysis Pages/, then falls back to Legacy/.
    """
    sp_stems = list(sp_urls.keys())
    log(f"  {len(sp_stems)} spectrum source(s) to process")

    log(f"Matching spectra → current analysis pages (radius={MATCH_RADIUS_ARCSEC}\")…")
    sp_to_ap = _match_stems_to_catalog(sp_stems, list(all_ap_urls.keys()))
    current_matched = sum(1 for v in sp_to_ap.values() if v is not None)
    log(f"  {current_matched}/{len(sp_stems)} matched in current Analysis Pages/")

    # For spectra with no current match, try legacy
    unmatched = [s for s in sp_stems if sp_to_ap[s] is None]
    if unmatched and legacy_ap_urls:
        log(f"Falling back to Legacy/ for {len(unmatched)} unmatched spectrum(s)…")
        legacy_match = _match_stems_to_catalog(unmatched, list(legacy_ap_urls.keys()))
        leg_matched = sum(1 for v in legacy_match.values() if v is not None)
        log(f"  {leg_matched}/{len(unmatched)} found in Legacy/")
        for s, leg_stem in legacy_match.items():
            if leg_stem is not None:
                sp_to_ap[s] = ("__legacy__", leg_stem)  # tuple signals legacy source
    else:
        legacy_match = {}

    log("Building redshift map…")
    redshift_map = build_redshift_map(sp_stems, redshift_dict)

    # Read existing sheet to recover any manually-entered Classification values.
    # Use sky-coordinate matching so precision differences don't cause missed recoveries.
    log(f"Updating spreadsheet tab '{SHEET_TAB}'…")
    _ensure_sheet_tab(sheets, SHEET_ID, SHEET_TAB)
    existing_rows = _read_sheet(sheets, SHEET_ID, SHEET_TAB)
    log(f"  {len(existing_rows)} existing row(s) in sheet (including header)")

    existing_classifications: dict[str, str] = {}  # sp_stem → classification
    valid_sp = [s for s in sp_stems if parse_coord_stem(s)]
    if len(existing_rows) > 1 and valid_sp:
        ex_coords, ex_classifications = [], []
        for row in existing_rows[1:]:
            if len(row) >= 2:
                try:
                    ex_coords.append((float(row[0]), float(row[1])))
                    ex_classifications.append(row[5] if len(row) > 5 else "")
                except ValueError:
                    pass
        if ex_coords:
            ex_sky = SkyCoord(ra=[c[0] for c in ex_coords] * u.deg,
                              dec=[c[1] for c in ex_coords] * u.deg)
            sp_sky_all = SkyCoord(
                ra=[parse_coord_stem(s)[0] for s in valid_sp] * u.deg,
                dec=[parse_coord_stem(s)[1] for s in valid_sp] * u.deg,
            )
            idx, sep, _ = ex_sky.match_to_catalog_sky(sp_sky_all)
            for i, (classification, separation) in enumerate(zip(ex_classifications, sep)):
                if separation.arcsec <= MATCH_RADIUS_ARCSEC and classification:
                    existing_classifications[valid_sp[idx[i]]] = classification
        log(f"  Recovered {len(existing_classifications)} classification(s) from existing rows")

    # Rebuild sheet entirely — one row per spectrum, duplicates impossible.
    new_rows = [SHEET_HEADERS]
    skipped = 0
    for stem in sorted(sp_stems):
        coords = parse_coord_stem(stem)
        if not coords:
            log(f"  WARNING: could not parse coords from '{stem}', skipping")
            skipped += 1
            continue
        ra, dec = coords

        sp_url = sp_urls.get(stem, "")

        ap_match = sp_to_ap.get(stem)
        if isinstance(ap_match, tuple):  # legacy fallback
            _, leg_stem = ap_match
            ap_url = legacy_ap_urls.get(leg_stem, "")
        else:
            ap_url = all_ap_urls.get(ap_match, "") if ap_match else ""

        redshift = redshift_map.get(stem)
        new_rows.append([
            ra,
            dec,
            f'=HYPERLINK("{ap_url}", "view")' if ap_url else "",
            f'=HYPERLINK("{sp_url}", "view")' if sp_url else "",
            redshift if redshift is not None else "",
            existing_classifications.get(stem, ""),
        ])

    if skipped:
        log(f"  WARNING: {skipped} file(s) skipped due to unparseable filenames")

    log(f"  Writing {len(new_rows) - 1} total row(s) to sheet…")
    _write_sheet(sheets, SHEET_ID, SHEET_TAB, new_rows)
    log(f"  Done.")


def sync_queue(sheets, all_ap_urls: dict[str, str], legacy_ap_urls: dict[str, str]) -> None:
    """Fill in Analysis Page links for manually-added rows in the Queue tab.

    Rows are never deleted or reordered — only the Analysis Page cell is written
    when it is currently empty and a matching page can be found.
    """
    log(f"Syncing '{QUEUE_TAB}' tab…")
    _ensure_sheet_tab(sheets, SHEET_ID, QUEUE_TAB)
    rows = _read_sheet(sheets, SHEET_ID, QUEUE_TAB)

    if not rows:
        _write_sheet(sheets, SHEET_ID, QUEUE_TAB, [QUEUE_HEADERS])
        log("  Initialised Queue tab with headers (no sources yet)")
        return

    # Ensure header is correct without touching data rows
    if rows[0] != QUEUE_HEADERS:
        rows[0] = QUEUE_HEADERS

    updated = 0
    new_rows = [rows[0]]
    for row in rows[1:]:
        if len(row) < 2:
            new_rows.append(row)
            continue
        try:
            ra, dec = float(row[0]), float(row[1])
        except ValueError:
            new_rows.append(row)
            continue

        # Pad to at least 4 columns so indexing is safe
        padded = list(row) + [""] * max(0, 4 - len(row))
        current_ap = padded[2]

        if not current_ap:
            ap_url = _find_ap_url(ra, dec, all_ap_urls, legacy_ap_urls)
            if ap_url:
                padded[2] = f'=HYPERLINK("{ap_url}", "view")'
                updated += 1

        new_rows.append(padded)

    log(f"  {updated} Analysis Page link(s) filled in")
    _write_sheet(sheets, SHEET_ID, QUEUE_TAB, new_rows)
    log("  Done.")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Sync follow-up data to Google Drive and update the status spreadsheet.",
        epilog="With no flags, all three steps run. Pass flags to run only specific steps.",
    )
    parser.add_argument("--analysis-pages", action="store_true",
                        help="Upload analysis pages and flowcharts to Drive")
    parser.add_argument("--spectra", action="store_true",
                        help="Upload spectrum plots to Drive")
    parser.add_argument("--sheet", action="store_true",
                        help="Rebuild the Follow-up Status spreadsheet")
    parser.add_argument("--queue", action="store_true",
                        help="Fill in Analysis Page links in the Queue tab")
    args = parser.parse_args()

    # If no flags given, run everything
    run_all    = not any([args.analysis_pages, args.spectra, args.sheet, args.queue])
    do_ap      = run_all or args.analysis_pages
    do_spectra = run_all or args.spectra
    do_sheet   = run_all or args.sheet
    do_queue   = run_all or args.queue

    log("=== sync_followup_drive starting ===")
    log(f"  Steps: {'analysis-pages ' if do_ap else ''}{'spectra ' if do_spectra else ''}{'sheet ' if do_sheet else ''}{'queue' if do_queue else ''}".rstrip())

    log(f"Loading redshifts from {REDSHIFTS_CSV}…")
    redshift_dict = load_redshifts()
    log(f"  {len(redshift_dict)} redshifts loaded")

    log("Authenticating with Google…")
    creds  = _get_credentials()
    drive  = build("drive",  "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds) if (do_sheet or do_queue) else None
    log("Authentication OK")

    if do_ap:
        log("Syncing analysis pages…")
        all_ap_urls = sync_analysis_pages(drive)
    if do_spectra:
        sp_urls = sync_spectra(drive)

    if do_sheet or do_queue:
        # Fetch anything we didn't just upload
        if not do_ap or not do_spectra:
            log("Fetching current Drive file lists…")
            fetched_ap, fetched_legacy, fetched_sp = _fetch_existing_urls(drive)
            if not do_ap:
                all_ap_urls = fetched_ap
            if not do_spectra:
                sp_urls = fetched_sp
            legacy_ap_urls = fetched_legacy
        else:
            log("Scanning Legacy/ for fallback analysis pages…")
            leg_root_id = _get_or_create_folder(drive, "Legacy", DRIVE_FOLDER_ID)
            legacy_ap_urls = _list_drive_files_recursive(drive, leg_root_id)
            log(f"  {len(legacy_ap_urls)} legacy files found")

        if do_sheet:
            sync_sheet(sheets, all_ap_urls, legacy_ap_urls, sp_urls, redshift_dict)
        if do_queue:
            sync_queue(sheets, all_ap_urls, legacy_ap_urls)

    log("=== Done ===")


if __name__ == "__main__":
    main()
