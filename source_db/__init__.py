"""Everything that fills the source database.

This package exists only to *build* `Data/db/` -- the per-source results
database that backs the website. It is deliberately separate from
`Source_Analysis`, which is analysis code, and from `Source_Analysis.catalogs`,
which is bulk reference data used for matching and never leaves the laptop.

    paths          shared locations
    identity       source_uuid assignment and the identity map
    catalog_tier   the ~300-column source rows, from the enriched tables
    lightcurves/   per-catalog bulk fetch + normalisation
    spectra/       per-catalog bulk fetch + normalisation

Entry points live here too, so the whole subsystem is one directory:

    python -m source_db.build_catalog_tier
    python -m source_db.fetch_lightcurves --catalog ztf
    python -m source_db.build_lightcurve_table

See `plan_source_database.md` for the design.
"""
