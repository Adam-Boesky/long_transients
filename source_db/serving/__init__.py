"""The serving tier: per-source payloads the website reads.

Derived from the analysis tiers (`sources.parquet` and `light_curves/`) and
disposable -- rebuild it whenever the science data changes. Annotations live
in their own database so that rebuilding never touches them.
"""
