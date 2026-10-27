"""Spectra: the last science tier.

Two archival sources, both already crossmatched in the catalog tier, so no
matching happens here -- `sources.parquet` already records which sources have
a spectrum and the identifiers needed to fetch it:

    SDSS DR20   sdss_matched, sdss_spec_url                -> a SAS file URL
    DESI DR1    desi_matched, desi_targetid                -> a SPARCL record

Local follow-up spectra (`Data/followup/<slug>/spectra`, see
`Source_Analysis.followup`) are not wired up yet; each target's `target.yaml`
carries its `source_uuid`, so attaching them is a lookup, not a crossmatch.
"""
