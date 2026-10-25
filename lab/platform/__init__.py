"""Decision platform: the engineering behind a decision, end to end, on public data.

Ingest (pipeline/build_markets.py, GitHub Actions) → layered data models with tests → governed metrics and KPIs →
model training with a registry and a promotion gate → a validated scoring service → monitoring with drift and
retrain rules. Every number on the page is computed from the repository's own data files.
"""
