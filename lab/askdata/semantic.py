"""A small governed warehouse and semantic layer for the analytics copilot.

The warehouse is built from the subscriber lab's synthetic company (Corvane Connect) as monthly fact tables in SQLite.
Every question becomes an intent (metric, dimensions, filters, time), and the intent compiles to SQL using only the
definitions below, so two people asking the same question always get the same number.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from lab.sub import data as SD
from lab.sub import models as SM

FIRST_MONTH = pd.Timestamp("2023-10-01")   # month 0
N_MONTHS = 36


def month_label(i: int) -> str:
    return (FIRST_MONTH + pd.DateOffset(months=int(i))).strftime("%b %Y")


@st.cache_resource(show_spinner="Building the warehouse…")
def warehouse() -> sqlite3.Connection:
    c = SD.company()
    s = c.subs.copy()
    s["segment"] = s["sub_id"].map(SM.segments().labels).to_numpy()
    s["customer_type"] = np.where(s["b2b"], "Small business", "Consumer")
    s = s.rename(columns={"tier": "plan"})
    keys = ["region", "channel", "plan", "segment", "customer_type"]
    start, churn = s["start_month"].to_numpy(), s["churn_month"].to_numpy()
    rows = []
    for m in range(N_MONTHS):
        act_start = (start < m) & ((churn < 0) | (churn >= m))
        new = start == m
        churned = churn == m
        act_end = (start <= m) & ((churn < 0) | (churn > m))
        f = pd.DataFrame({"active_start": act_start.astype(int), "new_subs": new.astype(int), "churned": churned.astype(int), "active_end": act_end.astype(int),
                          "mrr": np.where(act_end, s["fee"], 0.0)})
        g = pd.concat([s[keys], f], axis=1).groupby(keys, observed=True).sum().reset_index()
        g = g[(g[["active_start", "new_subs", "churned", "active_end"]].sum(1)) > 0]
        g.insert(0, "month_id", m)
        rows.append(g)
    fact = pd.concat(rows, ignore_index=True)
    # Care contacts: each month, active subscribers contact care at their own rate, mostly about their usual reason.
    rng = np.random.default_rng(5)
    drivers = np.array(SD.DRIVERS)
    care_rows = []
    rate = s["care_calls_90d"].to_numpy() / 3 + 0.05
    usual = s["last_driver"].to_numpy()
    for m in range(N_MONTHS):
        act = (start <= m) & ((churn < 0) | (churn >= m))
        n = rng.poisson(rate * act)
        has = n > 0
        reason = np.where(usual != "No contact", usual, drivers[rng.integers(0, len(drivers), len(s))])
        swap = rng.random(len(s)) < 0.3
        reason = np.where(swap, drivers[rng.integers(0, len(drivers), len(s))], reason)
        g = pd.DataFrame({"region": s["region"], "segment": s["segment"], "reason": reason, "contacts": n})[has]
        g = g.groupby(["region", "segment", "reason"], observed=True)["contacts"].sum().reset_index()
        g.insert(0, "month_id", m)
        care_rows.append(g)
    care = pd.concat(care_rows, ignore_index=True)
    months = pd.DataFrame({"month_id": range(N_MONTHS)})
    months["month_start"] = [(FIRST_MONTH + pd.DateOffset(months=i)).strftime("%Y-%m-01") for i in months["month_id"]]
    months["year"] = [(FIRST_MONTH + pd.DateOffset(months=i)).year for i in months["month_id"]]
    months["quarter"] = [f"{(FIRST_MONTH + pd.DateOffset(months=i)).year}-Q{((FIRST_MONTH + pd.DateOffset(months=i)).month - 1) // 3 + 1}" for i in months["month_id"]]
    months["is_quarter_end"] = [int((FIRST_MONTH + pd.DateOffset(months=i)).month % 3 == 0 or i == N_MONTHS - 1) for i in months["month_id"]]
    months["is_year_end"] = [int((FIRST_MONTH + pd.DateOffset(months=i)).month == 12 or i == N_MONTHS - 1) for i in months["month_id"]]
    con = sqlite3.connect(":memory:", check_same_thread=False)
    fact.to_sql("fact_subscriptions", con, index=False)
    care.to_sql("fact_care_contacts", con, index=False)
    months.to_sql("dim_month", con, index=False)
    con.execute("CREATE INDEX ix_fs ON fact_subscriptions(month_id)")
    con.execute("CREATE INDEX ix_fc ON fact_care_contacts(month_id)")
    return con


# ---------------------------------------------------------------------------
# The semantic layer
# ---------------------------------------------------------------------------
@dataclass
class Metric:
    key: str
    label: str
    description: str
    source: str               # subs | care | both
    numerator: str
    denominator: str | None = None
    snapshot: bool = False    # semi-additive: report the value at the end of each period
    fmt: str = "int"          # int | pct | money | dec
    synonyms: list = field(default_factory=list)
    dims: tuple = ("region", "channel", "plan", "segment", "customer_type")
    owner: str = "Analytics"


METRICS = {m.key: m for m in [
    Metric("active_subscribers", "Active subscribers", "Subscribers active at the end of the period.", "subs", "SUM(f.active_end)",
           snapshot=True, synonyms=["active subscribers", "active customers", "subscriber count", "subscribers", "customers", "base size", "customer base", "how many customers", "how many subscribers"]),
    Metric("new_subscribers", "New subscribers", "Subscribers whose service started in the period (gross adds).", "subs", "SUM(f.new_subs)",
           synonyms=["new subscribers", "gross adds", "new customers", "acquisitions", "sign ups", "signups", "new connects", "joined"], owner="Marketing"),
    Metric("churned_subscribers", "Churned subscribers", "Subscribers whose service ended in the period, for any reason.", "subs", "SUM(f.churned)",
           synonyms=["churned subscribers", "disconnects", "cancellations", "cancels", "churners", "customers lost", "lost customers", "how many left", "leavers", "did we lose", "we lost", "lose", "lost"]),
    Metric("churn_rate", "Monthly churn rate", "Churned subscribers divided by subscribers active at the start of each month, averaged over the period.",
           "subs", "SUM(f.churned)", "SUM(f.active_start)", fmt="pct", synonyms=["churn rate", "churn", "attrition rate", "attrition", "disconnect rate", "cancellation rate"]),
    Metric("net_adds", "Net adds", "New subscribers minus churned subscribers.", "subs", "SUM(f.new_subs) - SUM(f.churned)",
           synonyms=["net adds", "net additions", "net growth", "net change"], owner="Finance"),
    Metric("mrr", "Monthly recurring revenue", "Sum of monthly fees for subscribers active at the end of the period.", "subs", "SUM(f.mrr)",
           snapshot=True, fmt="money", synonyms=["monthly recurring revenue", "mrr", "recurring revenue", "revenue", "billings"], owner="Finance"),
    Metric("arpu", "Average revenue per subscriber", "Monthly recurring revenue divided by active subscribers, at the end of the period.", "subs",
           "SUM(f.mrr)", "SUM(f.active_end)", snapshot=True, fmt="money", synonyms=["average revenue per user", "average revenue per subscriber", "arpu", "average bill", "average fee", "average monthly fee"], owner="Finance"),
    Metric("care_contacts", "Care contacts", "Contacts handled by Care (calls and chats) in the period.", "care", "SUM(c.contacts)",
           synonyms=["care contacts", "contacts", "calls", "call volume", "support calls", "care calls", "chats", "contact volume", "contact reasons", "call reasons", "contact reason", "why customers call"],
           dims=("region", "segment", "reason"), owner="Customer Care"),
    Metric("contacts_per_100", "Contacts per 100 subscribers", "Care contacts per 100 subscribers active at the start of the month, averaged over the period.",
           "both", "SUM(c.contacts)", "SUM(s.active_start)", fmt="dec",
           synonyms=["contacts per 100", "contact rate", "calls per 100", "contacts per subscriber", "calls per customer", "contact ratio"],
           dims=("region", "segment"), owner="Customer Care"),
]}

DIMS = {
    "region": dict(label="Region", values=list(SD.REGIONS), synonyms=["region", "regions", "market", "markets", "area", "geography"]),
    "channel": dict(label="Acquisition channel", values=list(SD.CHANNELS), synonyms=["channel", "channels", "acquisition channel", "sales channel"]),
    "plan": dict(label="Plan", values=list(SD.TIERS), synonyms=["plan", "plans", "tier", "tiers", "speed tier", "package"]),
    "segment": dict(label="Segment", values=list(SD.PERSONAS), synonyms=["segment", "segments", "persona", "personas"]),
    "customer_type": dict(label="Customer type", values=["Consumer", "Small business"], synonyms=["customer type", "consumer vs business", "business vs consumer", "b2b", "type of customer"]),
    "reason": dict(label="Contact reason", values=list(SD.DRIVERS), synonyms=["reason", "reasons", "contact reason", "call reason", "driver", "drivers", "why customers call"]),
}
GRAINS = {"month": "dm.month_start", "quarter": "dm.quarter", "year": "dm.year"}
PERIOD_END = {"month": None, "quarter": "dm.is_quarter_end = 1", "year": "dm.is_year_end = 1"}


@dataclass
class Intent:
    metric: str | None = None
    dims: list = field(default_factory=list)
    grain: str | None = None
    filters: dict = field(default_factory=dict)
    start: int = N_MONTHS - 12
    end: int = N_MONTHS - 1
    top: int | None = None
    order: str | None = None          # desc | asc
    refusal: str | None = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {"metric": self.metric, "group_by": self.dims, "time_grain": self.grain, "filters": self.filters,
             "period": f"{month_label(self.start)} – {month_label(self.end)}"}
        if self.top:
            d["top"] = self.top
        if self.order:
            d["order"] = self.order
        if self.refusal:
            d = {"refusal": self.refusal}
        return d


def _lit(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def compile_sql(it: Intent) -> str:
    m = METRICS[it.metric]
    grain_col = GRAINS.get(it.grain) if it.grain else None
    sel, grp = [], []
    if grain_col:
        sel.append(f"{grain_col} AS period")
        grp.append(grain_col)
    alias = {"subs": "f", "care": "c", "both": "s"}[m.source]
    for d in it.dims:
        sel.append(f"{alias}.{d}" if m.source != "both" else f"s.{d}")
        grp.append(f"{alias}.{d}" if m.source != "both" else f"s.{d}")
    where = [f"{'f' if m.source == 'subs' else 'c' if m.source == 'care' else 's'}.month_id BETWEEN {it.start} AND {it.end}"]
    tbl_alias = "f" if m.source == "subs" else "c" if m.source == "care" else "s"
    for d, v in it.filters.items():
        where.append(f"{tbl_alias}.{d} = {_lit(v)}")
    if m.snapshot:
        if it.grain and PERIOD_END[it.grain]:
            where.append(PERIOD_END[it.grain])
        elif not it.grain:
            where.append(f"{tbl_alias}.month_id = {it.end}")
    value = m.numerator if not m.denominator else f"1.0 * ({m.numerator}) / NULLIF({m.denominator}, 0)"
    if m.key == "contacts_per_100":
        value = f"100.0 * ({m.numerator}) / NULLIF({m.denominator}, 0)"
    sel.append(f"{value} AS value")
    if m.source == "subs":
        frm = "fact_subscriptions f JOIN dim_month dm ON dm.month_id = f.month_id"
    elif m.source == "care":
        frm = "fact_care_contacts c JOIN dim_month dm ON dm.month_id = c.month_id"
    else:
        keys = ["month_id", "region", "segment"]
        frm = ("(SELECT month_id, region, segment, SUM(active_start) AS active_start FROM fact_subscriptions GROUP BY month_id, region, segment) s\n"
               "  LEFT JOIN (SELECT month_id, region, segment, SUM(contacts) AS contacts FROM fact_care_contacts GROUP BY month_id, region, segment) c\n"
               "    ON " + " AND ".join(f"c.{k} = s.{k}" for k in keys) + "\n  JOIN dim_month dm ON dm.month_id = s.month_id")
    sql = "SELECT " + ",\n       ".join(sel) + f"\nFROM {frm}\nWHERE " + "\n  AND ".join(where)
    if grp:
        sql += "\nGROUP BY " + ", ".join(grp)
    order = []
    if grain_col:
        order.append("period")
    if it.dims and not grain_col:
        order.append(f"value {'ASC' if it.order == 'asc' else 'DESC'}")
    if order:
        sql += "\nORDER BY " + ", ".join(order)
    if it.top and not grain_col:
        sql += f"\nLIMIT {int(it.top)}"
    return sql


def run(it: Intent) -> pd.DataFrame:
    sql = compile_sql(it)
    df = pd.read_sql_query(sql, warehouse())
    return df
