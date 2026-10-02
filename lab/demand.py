"""Synthetic hotel demand with booking curves, pickup forecasting and backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

CAPACITY = 150
LEADS = 60
DOW_EFFECT = np.array([0.82, 0.86, 0.92, 1.00, 1.18, 1.28, 0.94])  # Mon..Sun


def _curve(lead: np.ndarray, mid: float, slope: float) -> np.ndarray:
    """Share of final bookings already on the books at a given lead time (days before arrival)."""
    return 1 / (1 + np.exp((lead - mid) / slope))


@dataclass
class World:
    history: pd.DataFrame       # stay_date, dow, final, curve params
    otb: pd.DataFrame           # long: stay_date, lead, on_books (history, all leads)
    asof: pd.Timestamp
    future: pd.DataFrame        # next 30 stay dates: stay_date, lead, on_books_now, true_final


@st.cache_data(show_spinner=False)
def build_world(seed: int = 21, days_history: int = 540) -> World:
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp("2026-06-01")
    start = asof - pd.Timedelta(days=days_history)
    dates = pd.date_range(start, asof + pd.Timedelta(days=29), freq="D")
    doy = dates.dayofyear.to_numpy()
    season = 1 + 0.18 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    trend = 1 + 0.06 * (np.arange(len(dates)) / 365.25)
    events = np.ones(len(dates))
    for k in rng.choice(len(dates), size=34, replace=False):
        events[k:k + 2] *= rng.uniform(1.15, 1.45)
    mean = 118 * DOW_EFFECT[dates.dayofweek] * season * trend * events
    final = np.clip(rng.normal(mean, 0.09 * mean), 20, None)
    mid = rng.normal(16, 3, len(dates)).clip(6, 30)
    slope = rng.normal(7, 1.2, len(dates)).clip(3, 12)
    hist = pd.DataFrame({"stay_date": dates, "dow": dates.dayofweek, "demand": final, "mid": mid, "slope": slope})
    hist["final"] = np.minimum(hist["demand"], CAPACITY).round()  # sold rooms are censored at capacity

    leads = np.arange(LEADS + 1)
    rows = []
    for r in hist.itertuples():
        share = _curve(leads, r.mid, r.slope)
        share = share / share[0]
        noise = np.cumsum(rng.normal(0, 0.012, len(leads))[::-1])[::-1]
        on = np.clip(np.round(r.final * np.clip(share + noise * (leads > 0), 0, 1)), 0, r.final)
        on = np.minimum.accumulate(on[::-1])[::-1] if False else np.maximum.accumulate(on[::-1])[::-1]
        on[0] = r.final
        rows.append(pd.DataFrame({"stay_date": r.stay_date, "lead": leads, "on_books": on}))
    otb = pd.concat(rows, ignore_index=True)
    otb["booked_on"] = otb["stay_date"] - pd.to_timedelta(otb["lead"], unit="D")
    otb = otb[otb["booked_on"] <= asof].copy()  # nothing after the as-of date is known

    fut = hist[hist["stay_date"] >= asof].copy()
    fut["lead"] = (fut["stay_date"] - asof).dt.days
    now = otb.merge(fut[["stay_date", "lead"]], on=["stay_date", "lead"])
    fut = fut.merge(now[["stay_date", "on_books"]], on="stay_date", how="left").rename(columns={"on_books": "on_books_now"})
    fut = fut.rename(columns={"final": "true_final"})
    past = hist[hist["stay_date"] < asof].copy()
    return World(history=past, otb=otb, asof=asof, future=fut)


def pickup_table(world: World, end: pd.Timestamp, lookback: int = 70) -> pd.DataFrame:
    """Average additional rooms booked from each lead time to arrival, by weekday, from recent stays."""
    hist = world.history[(world.history["stay_date"] < end) & (world.history["stay_date"] >= end - pd.Timedelta(days=lookback))]
    o = world.otb.merge(hist[["stay_date", "dow", "final"]], on="stay_date")
    o["pickup"] = o["final"] - o["on_books"]
    return o.groupby(["dow", "lead"])["pickup"].mean().rename("avg_pickup").reset_index()


def last_year(world: World, stay: pd.Series) -> np.ndarray:
    """The prior approach: same weekday last year, as it happened."""
    h = world.history.set_index("stay_date")["final"]
    return (stay - pd.Timedelta(days=364)).map(h).to_numpy(dtype=float)


def seasonal_baseline(world: World, stay: pd.Series, dow: pd.Series, end: pd.Timestamp) -> np.ndarray:
    """Recent weekday level x last year's seasonal movement between now and the stay date."""
    h = world.history[world.history["stay_date"] < end]
    level = h.tail(56).groupby("dow")["final"].mean()
    smooth = world.history.set_index("stay_date")["final"].rolling(28, center=True, min_periods=14).mean()
    anchor = smooth.get(end - pd.Timedelta(days=364 + 14), np.nan)
    target = (stay - pd.Timedelta(days=364)).map(smooth)
    season = (target / anchor) if pd.notna(anchor) and anchor > 0 else 1.0
    return (dow.map(level) * pd.Series(season, index=stay.index).fillna(1.0)).to_numpy(dtype=float)


def forecast(world: World, stays: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """stays needs stay_date, dow, lead, on_books. Returns naive, pickup, blended forecasts."""
    pk = pickup_table(world, end)
    f = stays.merge(pk, on=["dow", "lead"], how="left")
    f["avg_pickup"] = f["avg_pickup"].fillna(0)
    f["naive"] = last_year(world, f["stay_date"])
    f["seasonal"] = seasonal_baseline(world, f["stay_date"], f["dow"], end)
    f["pickup"] = f["on_books"] + f["avg_pickup"]
    w = np.clip(f["lead"] / 40, 0.15, 0.75)  # lean on history far out, on the booking pace close in
    f["blend"] = (1 - w) * f["pickup"] + w * f["seasonal"].fillna(f["pickup"])
    f["naive"] = f["naive"].fillna(f["seasonal"])
    for c in ("naive", "pickup", "blend"):
        f[c] = np.clip(f[c], f["on_books"], CAPACITY)
    return f


@st.cache_data(show_spinner=False)
def backtest(seed: int = 21, leads: tuple = (3, 7, 14, 21, 30, 45)) -> pd.DataFrame:
    world = build_world(seed)
    test = world.history.tail(120)
    rows = []
    for L in leads:
        snap = world.otb[world.otb["lead"].eq(L)].merge(test[["stay_date", "dow", "final"]], on="stay_date")
        for end_date, g in snap.groupby(snap["stay_date"].dt.to_period("W")):
            cutoff = g["stay_date"].min() - pd.Timedelta(days=L)
            f = forecast(world, g[["stay_date", "dow", "lead", "on_books"]], cutoff)
            f = f.merge(g[["stay_date", "final"]], on="stay_date")
            for m in ("naive", "pickup", "blend"):
                rows.append(pd.DataFrame({"lead": L, "method": m, "err": f[m] - f["final"], "final": f["final"]}))
    bt = pd.concat(rows, ignore_index=True)
    return bt
