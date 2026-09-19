"""Local crossmatch catalogs.

Each module wraps one catalog that lives on disk under `Data/catalogs/`, so
classifications can be looked up offline instead of queried from a remote
service. They share a cone-search interface:

    cone_search_many(ra, dec, radius_arcsec) -> one row per input, in input order
    cone_search(ra, dec, radius_arcsec)      -> all matches for one position

    agn     AGN-DB merged AGN/quasar catalog (8.1M rows) -- DuckDB against parquet,
            too wide to hold in memory, and queried per-source at plot time.
    sdss    SDSS DR17 specObj spectroscopic catalog (5.8M rows).
    desi    DESI DR1 (Iron) redshift catalog (28.4M rows).
    simbad  SIMBAD `basic` cross-identification database (22.2M rows) -- names,
            object types and the type hierarchy for positions the spectroscopic
            catalogs miss.
    tns     Transient Name Server object dump (207k rows).
    wise    AllWISE photometry, for W1-W2 colour.

`sdss`, `desi`, `simbad` and `tns` share `_base.LocalCatalog`, which holds the
slim parquet in memory and matches against a cached KD-tree. That is fast for
bulk work but costs up to ~13 GB for DESI and a further ~8 GB for SIMBAD, so
run them in a single process rather than inside a multiprocessing Pool.

`agn` and `simbad` therefore serve their single-position `cone_search` from
DuckDB against the parquet instead, which reads only the columns a query
touches. That is what lets `Sources.agn_match` and `Sources.simbad_match` run
inside a Pool without each worker holding a copy of the catalog.

`wise` is the exception: AllWISE is ~750M sources, too large to hold locally,
so it crossmatches remotely at IRSA and caches the answers per position. It
offers `cone_search_many` like the others but no `cone_search`, since a
single-position lookup there would mean a network round trip.
"""
