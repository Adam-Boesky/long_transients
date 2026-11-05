"""Image cutouts for the analysis pages.

Nothing here stores pixels. The decision was to not keep image data, so this
tier stores only what is needed to *address* an image -- for PanSTARRS, the
stack filename per band, from which a `fitscut.cgi` URL is built.

That distinction is what makes the page fast: with the filename already in
the database, the browser is handed an `<img src>` on first paint and fetches
the JPEG straight from STScI, with no round trip through our API and no
per-source lookup at view time.
"""
