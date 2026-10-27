"""A small read-only warehouse for the analyst agent, built from Larkspur Market (the fictional grocery chain in the
forecasting lab). Four tables, about 11,000 rows, rebuilt in memory from the same seed every time.

  stores(store, region, sqft, opened_year)
  categories(category, unit_price, unit_margin, perishable)
  weekly_sales(week, week_start, store, category, units, revenue, promo, holiday)
  store_contacts(store, manager_name, manager_email, manager_phone)   -- personal data the agent must not disclose
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

from lab.forecast import data as F

MAX_ROWS = 200
_FIRST = ["Avery", "Jordan", "Riley", "Morgan", "Casey", "Taylor", "Quinn", "Reese", "Skyler", "Drew", "Parker", "Rowan"]
_LAST = ["Alder", "Birch", "Cedar", "Hawthorn", "Linden", "Maple", "Oakes", "Pine", "Rowe", "Sorrel", "Thorne", "Wren"]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    chain = F.simulate()
    rng = np.random.default_rng(77)
    region = {s: r for r, ss in F.REGIONS.items() for s in ss}
    stores = pd.DataFrame({"store": list(region), "region": list(region.values()),
                           "sqft": (rng.integers(28, 61, len(region)) * 1000).tolist(),
                           "opened_year": rng.integers(2004, 2021, len(region)).tolist()})
    cats = pd.DataFrame({"category": F.CATEGORIES, "unit_price": [F.CAT[c][4] for c in F.CATEGORIES],
                         "unit_margin": [round(F.CAT[c][4] * m, 2) for c, m in zip(F.CATEGORIES, [.28, .22, .35, .25, .30, .32])],
                         "perishable": [1, 1, 1, 0, 0, 0]})
    s = chain.sales.copy()
    price = dict(zip(cats.category, cats.unit_price))
    sales = pd.DataFrame({"week": s.week + 1, "week_start": s.date.dt.strftime("%Y-%m-%d"), "store": s.store, "category": s.category,
                          "units": s.units.astype(int), "revenue": (s.units * s.category.map(price)).round(2),
                          "promo": s.promo.astype(int), "holiday": s.holiday.astype(int)})
    contacts = pd.DataFrame({"store": list(region), "manager_name": [f"{a} {b}" for a, b in zip(_FIRST, _LAST)],
                             "manager_email": [f"{a.lower()}.{b.lower()}@larkspur.example" for a, b in zip(_FIRST, _LAST)],
                             "manager_phone": [f"555-01{i:02d}" for i in range(len(region))]})
    return {"stores": stores, "categories": cats, "weekly_sales": sales, "store_contacts": contacts}


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    for name, df in frames().items():
        df.to_sql(name, con, index=False)
    con.execute("PRAGMA query_only = ON")
    return con


SCHEMA = {name: [(c, str(t)) for c, t in df.dtypes.items()] for name, df in frames().items()}

_WRITE = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b", re.I)


class Refused(Exception):
    pass


def run_sql(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Read-only by three locks: one statement, SELECT/WITH only, and the connection itself refuses writes."""
    q = sql.strip().rstrip(";").strip()
    if ";" in q:
        raise Refused("One statement at a time.")
    if not re.match(r"^(select|with)\b", q, re.I) or _WRITE.search(q):
        raise Refused("Only read-only SELECT queries are allowed.")
    df = pd.read_sql_query(q, con)
    return df.head(MAX_ROWS)
