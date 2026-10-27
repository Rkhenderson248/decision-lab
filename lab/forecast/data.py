"""Larkspur Market: a fictional grocery chain, simulated week by week.

Three regions, four stores each, six categories: 72 store-category series of weekly unit sales from January 2023 to
December 2025 (156 weeks). Each series has its own level and trend, a yearly pattern that differs by category,
holiday weeks, promotions known in advance, and noise. One planted event: a competitor opens next to store S07 in
week 124 and takes about 16% of its volume. The data is synthetic with known structure, so every forecast can be
checked against what was planted. Deterministic: the same seed always gives the same chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20230101
WEEKS = 156
START = pd.Timestamp("2023-01-02")           # a Monday
REGIONS = {"North": ["S01", "S02", "S03", "S04"], "Central": ["S05", "S06", "S07", "S08"], "South": ["S09", "S10", "S11", "S12"]}
CATEGORIES = ["Produce", "Dairy", "Bakery", "Frozen", "Beverages", "Household"]
# category: (base units, yearly amplitude, peak week, promo lift, price per unit $)
CAT = {"Produce": (900, .22, 28, .30, 2.4), "Dairy": (700, .06, 50, .18, 3.1), "Bakery": (380, .10, 51, .25, 3.6),
       "Frozen": (420, .18, 30, .35, 4.2), "Beverages": (650, .30, 27, .40, 2.9), "Household": (300, .05, 47, .22, 6.5)}
BREAK_STORE, BREAK_WEEK, BREAK_SIZE = "S07", 124, 0.16
HOLIDAYS = {"Thanksgiving": [46, 98, 150], "Christmas": [51, 103, 155], "July 4": [26, 78, 130]}


def week_dates() -> pd.DatetimeIndex:
    return pd.date_range(START, periods=WEEKS, freq="W-MON")


@dataclass
class Chain:
    sales: pd.DataFrame       # long: week, date, region, store, category, units, promo, holiday
    wide: pd.DataFrame        # weeks x series ("S01|Produce")
    promo: pd.DataFrame       # weeks x series (0/1), known in advance
    holiday: np.ndarray       # weeks (0/1)
    hash: str


def simulate(seed: int = SEED) -> Chain:
    rng = np.random.default_rng(seed)
    w = np.arange(WEEKS)
    hol = np.zeros(WEEKS)
    for ws in HOLIDAYS.values():
        hol[[x for x in ws if x < WEEKS]] = 1
    rows, wide, promo = [], {}, {}
    for region, stores in REGIONS.items():
        rsize = {"North": 1.0, "Central": 1.15, "South": .9}[region]
        for s in stores:
            ssize = rsize * rng.uniform(.7, 1.35)
            strend = rng.normal(.0012, .0010)                       # weekly growth
            for c in CATEGORIES:
                base, amp, peak, lift, _ = CAT[c]
                lvl = base * ssize * rng.uniform(.85, 1.15)
                season = 1 + amp * np.cos(2 * np.pi * (w - peak) / 52.0) + .35 * amp * np.cos(4 * np.pi * (w - peak) / 52.0)
                hol_eff = 1 + hol * ({"Bakery": .55, "Produce": .35, "Beverages": .45}.get(c, .2))
                p = (rng.random(WEEKS) < .12).astype(int)                # ~6 promo weeks a year
                p[hol == 1] = 0
                mu = lvl * np.exp(strend * w) * season * hol_eff * (1 + lift * p)
                if s == BREAK_STORE:
                    mu = mu * np.where(w >= BREAK_WEEK, 1 - BREAK_SIZE, 1.0)
                y = np.maximum(0, np.round(mu * np.exp(rng.normal(0, .07, WEEKS) - .07 ** 2 / 2)))
                key = f"{s}|{c}"
                wide[key], promo[key] = y, p
                rows.append(pd.DataFrame({"week": w, "store": s, "region": region, "category": c, "units": y, "promo": p, "holiday": hol}))
    sales = pd.concat(rows, ignore_index=True)
    sales["date"] = week_dates()[sales["week"]]
    wide = pd.DataFrame(wide)
    promo = pd.DataFrame(promo)
    h = hashlib.sha256(pd.util.hash_pandas_object(wide, index=False).values.tobytes()).hexdigest()[:12]
    return Chain(sales, wide, promo, hol, h)


def hierarchy(columns: list[str]) -> tuple[list[str], np.ndarray]:
    """Summing matrix S for the grouped hierarchy: total, regions, stores, categories, region x category, bottom."""
    store_region = {s: r for r, ss in REGIONS.items() for s in ss}
    nodes, rows = [], []
    def add(name, f):
        nodes.append(name); rows.append([1.0 if f(c) else 0.0 for c in columns])
    add("Total", lambda c: True)
    for r in REGIONS:
        add(f"Region · {r}", lambda c, r=r: store_region[c.split('|')[0]] == r)
    for r, ss in REGIONS.items():
        for s in ss:
            add(f"Store · {s}", lambda c, s=s: c.split('|')[0] == s)
    for k in CATEGORIES:
        add(f"Category · {k}", lambda c, k=k: c.split('|')[1] == k)
    for r in REGIONS:
        for k in CATEGORIES:
            add(f"{r} · {k}", lambda c, r=r, k=k: store_region[c.split('|')[0]] == r and c.split('|')[1] == k)
    for c in columns:
        add(c, lambda x, c=c: x == c)
    return nodes, np.array(rows)


def level_of(node: str) -> str:
    if node == "Total":
        return "Total"
    if node.startswith("Region"):
        return "Region"
    if node.startswith("Store"):
        return "Store"
    if node.startswith("Category"):
        return "Category"
    if "|" in node:
        return "Store × category"
    return "Region × category"
