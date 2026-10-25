"""Governed metrics over the gold layer: one definition, one owner, computed the same way everywhere.

Ratios are computed from summed numerators and denominators, never by averaging market rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    owner: str
    fmt: str
    definition: str
    sql: str


METRICS = [
    Metric("applications", "Purchase applications", "Market strategy", "int", "Home-purchase applications reported under HMDA.", "SUM(applications)"),
    Metric("originations", "Purchase originations", "Market strategy", "int", "Home-purchase loans originated.", "SUM(originations)"),
    Metric("origination_rate", "Origination rate", "Credit", "pct", "Originations divided by applications, from the summed counts.", "SUM(originations) / SUM(applications)"),
    Metric("denial_rate", "Denial rate", "Credit", "pct", "Applications denied, weighted by applications.", "SUM(denial_rate * applications) / SUM(applications)"),
    Metric("refinance_share", "Refinance share", "Market strategy", "pct", "Refinance share of activity, weighted by applications.", "SUM(refinance_share * applications) / SUM(applications)"),
    Metric("avg_loan", "Average purchase loan", "Finance", "money", "Average loan amount, weighted by originations.", "SUM(avg_loan_amount * originations) / SUM(originations)"),
]
BY_KEY = {m.key: m for m in METRICS}


def compute(con, by: str | None = None, where: str = "", table: str = "gold_market_year_features") -> pd.DataFrame:
    sel = ", ".join(f"{m.sql} AS {m.key}" for m in METRICS)
    grp = f"{by}, " if by else ""
    q = f"SELECT {grp}{sel} FROM {table} {('WHERE ' + where) if where else ''} {('GROUP BY ' + by + ' ORDER BY ' + by) if by else ''}"
    return pd.read_sql_query(q, con)


def fmt(key: str, v: float) -> str:
    f = BY_KEY[key].fmt
    if pd.isna(v):
        return "–"
    if f == "int":
        return f"{v:,.0f}"
    if f == "pct":
        return f"{v:.1%}"
    return f"${v:,.0f}"
