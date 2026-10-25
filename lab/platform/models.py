"""Layered data models (bronze → silver → gold) built as SQL in an in-memory SQLite warehouse, with data tests.

Bronze is what the scheduled pipeline lands (data/*.csv). Silver cleans and types one source at a time. Gold joins
them into analysis-ready tables. Each model declares what it depends on, so the run order is a topological sort and
the lineage graph comes for free, the way dbt does it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass
class Model:
    name: str
    layer: str
    sql: str
    deps: list[str]
    description: str
    grain: str
    owner: str = "Data engineering"


@dataclass
class Test:
    model: str
    name: str
    kind: str
    sql: str            # returns the failing rows
    severity: str = "error"
    note: str = ""


@dataclass
class RunResult:
    models: pd.DataFrame
    tests: pd.DataFrame
    con: sqlite3.Connection = field(repr=False)


# Bronze: landed by the pipeline, loaded as-is.
BRONZE = {
    "bronze_markets": ("markets.csv", "One row per CBSA: Census population, FHFA house prices and HMDA aggregates, as landed by the monthly build.", "market"),
    "bronze_market_years": ("market_years.csv", "HMDA purchase activity per market and year, as landed.", "market × year"),
    "bronze_lenders": ("lenders.csv", "Top HMDA filers per market with shares, as landed.", "market × lender"),
}

MODELS: list[Model] = [
    Model("silver_market_year", "silver", """
        SELECT CAST(market_key AS INTEGER) AS market_key,
               CAST(year AS INTEGER) AS year,
               CAST(purchase_applications AS REAL) AS applications,
               CAST(purchase_originations AS REAL) AS originations,
               CASE WHEN purchase_applications > 0 THEN 1.0 * purchase_originations / purchase_applications END AS origination_rate,
               purchase_denial_rate_pct / 100.0 AS denial_rate,
               refinance_share_pct / 100.0 AS refinance_share,
               avg_loan_amount
        FROM bronze_market_years
        WHERE purchase_applications IS NOT NULL
    """, ["bronze_market_years"], "HMDA purchase activity, typed, with rates as fractions and the origination rate derived.", "market × year", "Data engineering"),
    Model("silver_market_profile", "silver", """
        SELECT CAST(market_key AS INTEGER) AS market_key, market_name,
               CASE WHEN lower(area_type) LIKE 'metro%' THEN 'Metro' WHEN lower(area_type) LIKE 'micro%' THEN 'Micro' ELSE 'Other' END AS area_type,
               primary_state AS state, population_latest AS population, population_cagr_pct / 100.0 AS population_cagr,
               hpi_1y_pct / 100.0 AS hpi_1y, hpi_5y_cagr_pct / 100.0 AS hpi_5y_cagr
        FROM bronze_markets
    """, ["bronze_markets"], "Market attributes: name, metro or micro, state, population and house-price momentum.", "market", "Data engineering"),
    Model("silver_lender_share", "silver", """
        SELECT CAST(market_key AS INTEGER) AS market_key, lender, CAST(rank AS INTEGER) AS rank,
               share_pct / 100.0 AS share, records
        FROM bronze_lenders WHERE share_pct IS NOT NULL
    """, ["bronze_lenders"], "Lender shares per market, as fractions.", "market × lender", "Data engineering"),
    Model("gold_market_concentration", "gold", """
        SELECT market_key, COUNT(*) AS lenders_listed,
               SUM(share * share) AS hhi_top, MAX(share) AS top_share,
               SUM(CASE WHEN rank <= 5 THEN share ELSE 0 END) AS top5_share
        FROM silver_lender_share GROUP BY market_key
    """, ["silver_lender_share"], "Competitive concentration per market from the listed lenders.", "market", "Analytics engineering"),
    Model("gold_market_year_features", "gold", """
        WITH g AS (
          SELECT y.*,
                 LAG(originations) OVER (PARTITION BY market_key ORDER BY year) AS prev_originations,
                 LEAD(originations) OVER (PARTITION BY market_key ORDER BY year) AS next_originations
          FROM silver_market_year y
        )
        SELECT g.market_key, g.year, p.state, p.area_type,
               g.applications, g.originations, g.origination_rate, g.denial_rate, g.refinance_share, g.avg_loan_amount,
               CASE WHEN g.prev_originations > 0 THEN g.originations / g.prev_originations - 1 END AS growth,
               CASE WHEN g.originations > 0 THEN g.next_originations / g.originations - 1 END AS next_growth
        FROM g JOIN silver_market_profile p USING (market_key)
    """, ["silver_market_year", "silver_market_profile"], "One row per market and year with this year's growth and next year's (the training label).", "market × year", "Analytics engineering"),
    Model("gold_markets", "gold", """
        SELECT p.*, y.year AS latest_year, y.applications, y.originations, y.origination_rate, y.denial_rate,
               y.refinance_share, y.avg_loan_amount, c.lenders_listed, c.hhi_top, c.top5_share
        FROM silver_market_profile p
        LEFT JOIN (SELECT * FROM silver_market_year WHERE year = (SELECT MAX(year) FROM silver_market_year)) y USING (market_key)
        LEFT JOIN gold_market_concentration c USING (market_key)
    """, ["silver_market_profile", "silver_market_year", "gold_market_concentration"], "The analysis-ready market table: profile, latest HMDA year and concentration.", "market", "Analytics engineering"),
]

TESTS: list[Test] = [
    Test("bronze_markets", "market_key is unique", "unique", "SELECT market_key FROM bronze_markets GROUP BY market_key HAVING COUNT(*) > 1"),
    Test("bronze_markets", "market_key is not null", "not_null", "SELECT * FROM bronze_markets WHERE market_key IS NULL"),
    Test("silver_market_year", "one row per market and year", "unique", "SELECT market_key, year FROM silver_market_year GROUP BY 1, 2 HAVING COUNT(*) > 1"),
    Test("silver_market_year", "originations ≤ applications", "expression", "SELECT * FROM silver_market_year WHERE originations > applications"),
    Test("silver_market_year", "denial rate between 0 and 1", "accepted_range", "SELECT * FROM silver_market_year WHERE denial_rate < 0 OR denial_rate > 1"),
    Test("silver_market_year", "every market has four HMDA years", "expression",
         "SELECT market_key FROM silver_market_year GROUP BY market_key HAVING COUNT(DISTINCT year) < 4", "warn",
         "A market missing a year cannot produce a lagged growth feature."),
    Test("silver_market_profile", "area type is Metro or Micro", "accepted_values", "SELECT * FROM silver_market_profile WHERE area_type NOT IN ('Metro','Micro')", "warn"),
    Test("silver_lender_share", "lender shares between 0 and 1", "accepted_range", "SELECT * FROM silver_lender_share WHERE share < 0 OR share > 1"),
    Test("silver_lender_share", "every lender row has a known market", "relationships",
         "SELECT l.* FROM silver_lender_share l LEFT JOIN silver_market_profile p USING (market_key) WHERE p.market_key IS NULL"),
    Test("gold_market_year_features", "every row joins a market profile", "relationships",
         "SELECT y.* FROM silver_market_year y LEFT JOIN gold_market_year_features g USING (market_key, year) WHERE g.market_key IS NULL"),
    Test("gold_markets", "row count matches the market list", "expression",
         "SELECT 1 WHERE (SELECT COUNT(*) FROM gold_markets) != (SELECT COUNT(*) FROM bronze_markets)"),
    Test("gold_markets", "house-price coverage ≥ 40% of markets", "coverage",
         "SELECT 1 WHERE (SELECT AVG(hpi_1y IS NOT NULL) FROM gold_markets) < 0.40", "warn",
         "FHFA publishes metro indexes only, so micro areas have no house-price series."),
]


def order() -> list[str]:
    """Topological run order: bronze first, then each model once its dependencies have run."""
    done = list(BRONZE)
    pending = {m.name: m for m in MODELS}
    while pending:
        ready = [n for n, m in pending.items() if all(d in done for d in m.deps)]
        if not ready:
            raise ValueError("cycle in model dependencies")
        for n in ready:
            done.append(n)
            pending.pop(n)
    return done


def lineage() -> list[tuple[str, str]]:
    return [(d, m.name) for m in MODELS for d in m.deps]


def build() -> RunResult:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    rows = []
    for name, (fname, desc, grain) in BRONZE.items():
        t0 = time.perf_counter()
        df = pd.read_csv(DATA / fname)
        df.to_sql(name, con, index=False)
        rows.append({"model": name, "layer": "bronze", "rows": len(df), "ms": (time.perf_counter() - t0) * 1000,
                     "grain": grain, "description": desc, "depends_on": f"data/{fname}", "owner": "Data engineering"})
    by_name = {m.name: m for m in MODELS}
    for name in order():
        if name in BRONZE:
            continue
        m = by_name[name]
        t0 = time.perf_counter()
        con.execute(f"CREATE TABLE {m.name} AS {m.sql}")
        n = con.execute(f"SELECT COUNT(*) FROM {m.name}").fetchone()[0]
        rows.append({"model": m.name, "layer": m.layer, "rows": n, "ms": (time.perf_counter() - t0) * 1000,
                     "grain": m.grain, "description": m.description, "depends_on": ", ".join(m.deps), "owner": m.owner})
    tests = []
    for t in TESTS:
        fails = con.execute(f"SELECT COUNT(*) FROM ({t.sql})").fetchone()[0]
        status = "pass" if fails == 0 else ("warn" if t.severity == "warn" else "fail")
        tests.append({"model": t.model, "test": t.name, "kind": t.kind, "failing_rows": fails, "status": status, "note": t.note})
    return RunResult(pd.DataFrame(rows), pd.DataFrame(tests), con)


def build_log() -> dict:
    out = {}
    for f in ("markets_meta.json", "build_log.json"):
        p = DATA / f
        if p.exists():
            try:
                out[f] = json.loads(p.read_text())
            except ValueError:
                out[f] = {}
    return out
