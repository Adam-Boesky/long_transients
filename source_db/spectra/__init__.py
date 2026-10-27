"""Spectra: the last science tier.

Two archival sources, both already crossmatched in the catalog tier, so no
matching happens here -- `sources.parquet` already records which sources have
a spectrum and the identifiers needed to fetch it:

    SDSS DR20   sdss_matched, sdss_spec_url                -> a SAS file URL
    DESI DR1    desi_matched, desi_targetid                -> a SPARCL record

Local follow-up spectra in `Data/followup/spectra` are deliberately not wired
up yet: their filenames are truncated to 2-3 decimals (~10-30"), so matching
them to sources needs the precision recovery we deferred.
"""
