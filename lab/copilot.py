"""Models behind the Pricing & demand copilot.

Everything is synthetic and seeded. The hotel and its booking curves come
from lab.demand; this module adds the rate history (with the confounding a
real revenue team creates), elasticity estimation, the pricing optimiser,
switchback experiment maths and post-launch monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from lab import demand as dm

REF_RATE = 189.0
SEGMENTS = {
    # name: (share of demand, true elasticity)
    "Leisure": (0.6, -2.0),
    "Business": (0.4, -0.7),
}


# ---------------------------------------------------------------------------
# Rate history and demand response
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def rate_history(seed: int = 21) -> pd.DataFrame:
    """One row per past night: the rate charged, the rooms sold by segment, test flags.

    Revenue managers raise rates on nights they expect to be busy, so price and
    demand move together in the raw data. On about a third of nights a rate test
    nudged the price up or down at random; that random part identifies the true
    response.
    """
    world = dm.build_world(seed)
    rng = np.random.default_rng(seed + 7)
    h = world.history[["stay_date", "dow", "demand"]].copy().reset_index(drop=True)
    n = len(h)
    log_a = np.log(h["demand"].to_numpy())
    signal = log_a + rng.normal(0, 0.06, n)              # what the team could see coming
    tested = rng.random(n) < 0.32
    offset = np.where(tested, rng.choice([-0.10, -0.05, 0.05, 0.10], n), 0.0)
    log_p = np.log(REF_RATE) + 0.75 * (signal - log_a.mean()) + offset + rng.normal(0, 0.015, n)
    price = np.exp(log_p)

    total = np.zeros(n)
    for name, (share, eps) in SEGMENTS.items():
        mean = share * np.exp(log_a) * (price / REF_RATE) ** eps
        q = rng.poisson(np.clip(mean * rng.lognormal(0, 0.05, n), 0.1, None))
        h[name.lower()] = q
        total += q
    h["unconstrained"] = total
    h["sold"] = np.minimum(total, dm.CAPACITY)
    h["sold_out"] = total >= dm.CAPACITY
    h["rate"] = price.round(2)
    h["tested"] = tested
    h["offset"] = offset
    h["revpar"] = h["rate"] * h["sold"] / dm.CAPACITY
    doy = h["stay_date"].dt.dayofyear.to_numpy()
    h["sin"] = np.sin(2 * np.pi * doy / 365.25)
    h["cos"] = np.cos(2 * np.pi * doy / 365.25)
    h["t"] = np.arange(n) / 365.25
    return h


def _controls(df: pd.DataFrame) -> np.ndarray:
    dow = pd.get_dummies(df["dow"], prefix="d", drop_first=True, dtype=float).to_numpy()
    return np.column_stack([np.ones(len(df)), dow, df[["sin", "cos", "t"]].to_numpy()])


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


@dataclass
class Estimate:
    method: str
    segment: str
    value: float
    se: float

    @property
    def lo(self) -> float:
        return self.value - 1.96 * self.se

    @property
    def hi(self) -> float:
        return self.value + 1.96 * self.se


def estimate_elasticity(df: pd.DataFrame, column: str, segment: str) -> list[Estimate]:
    """Three estimates of the same number: naive, with controls, and instrumented by the rate tests."""
    d = df[~df["sold_out"] & (df[column] > 0)]
    y = np.log(d[column].to_numpy(dtype=float))
    lp = np.log(d["rate"].to_numpy())
    C = _controls(d)

    b, se = _ols(y, np.column_stack([np.ones(len(d)), lp]))
    naive = Estimate("Raw correlation", segment, b[1], se[1])

    b, se = _ols(y, np.column_stack([C, lp]))
    controlled = Estimate("With calendar controls", segment, b[-1], se[-1])

    # Two-stage least squares, with the random test offset as the instrument.
    Z = np.column_stack([C, d["offset"].to_numpy()])
    g, *_ = np.linalg.lstsq(Z, lp, rcond=None)
    lp_hat = Z @ g
    X2 = np.column_stack([C, lp_hat])
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    resid = y - np.column_stack([C, lp]) @ b2
    s2 = resid @ resid / (len(y) - X2.shape[1])
    cov = s2 * np.linalg.pinv(X2.T @ X2)
    iv = Estimate("Instrumented by rate tests", segment, b2[-1], float(np.sqrt(cov[-1, -1])))
    return [naive, controlled, iv]


@st.cache_data(show_spinner=False)
def elasticity_table(seed: int = 21) -> pd.DataFrame:
    h = rate_history(seed)
    rows = []
    for name in SEGMENTS:
        for e in estimate_elasticity(h, name.lower(), name):
            rows.append({"segment": name, "method": e.method, "value": e.value, "se": e.se,
                         "lo": e.lo, "hi": e.hi, "truth": SEGMENTS[name][1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data quality checks (the data contract)
# ---------------------------------------------------------------------------
def quality_checks(seed: int = 21) -> pd.DataFrame:
    world = dm.build_world(seed)
    h = rate_history(seed)
    otb = world.otb
    leak = int((otb["booked_on"] > world.asof).sum())
    mono = otb.sort_values(["stay_date", "lead"], ascending=[True, False]).groupby("stay_date")["on_books"].diff()
    decreasing = int((mono < -0.5).sum())
    sold_out = float(h["sold_out"].mean())
    tested = float(h["tested"].mean())
    gaps = int(pd.date_range(h["stay_date"].min(), h["stay_date"].max()).difference(h["stay_date"]).size)
    seg_ok = float((h["leisure"] + h["business"] - h["unconstrained"]).abs().max())
    return pd.DataFrame([
        ("As-of integrity", "No booking recorded after the snapshot date", f"{leak} rows", leak == 0, "Block"),
        ("Booking curves", "Rooms on the books never fall as arrival approaches", f"{decreasing} reversals",
         decreasing == 0, "Warn"),
        ("Calendar coverage", "Every night has a rate and an outcome", f"{gaps} missing nights", gaps == 0, "Block"),
        ("Segment totals", "Segment rooms add up to the total", f"max gap {seg_ok:.0f}", seg_ok < 1, "Block"),
        ("Censoring", "Sell-out nights flagged; demand on them is unknown", f"{sold_out:.1%} of nights",
         True, "Handle"),
        ("Price variation", "Enough random rate movement to measure response", f"{tested:.0%} of nights tested",
         tested >= 0.15, "Block"),
    ], columns=["check", "rule", "result", "passed", "on_fail"])


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def optimise_rate(to_come: float, sd: float, rooms_left: float, ref: float, elasticity: float,
                  floor: float, ceiling: float, max_move: float, seed: int = 3) -> dict:
    """Expected-revenue-maximising rate for the rooms still to sell, inside guardrails."""
    rng = np.random.default_rng(seed)
    draws = np.clip(rng.normal(to_come, max(sd, 1.0), 4000), 0, None)
    lo = max(floor, ref * (1 - max_move))
    hi = min(ceiling, ref * (1 + max_move))
    grid = np.linspace(ref * 0.55, ref * 1.9, 136)
    rev, sold = [], []
    for p in grid:
        dem = draws * (p / ref) ** elasticity
        s = np.minimum(dem, rooms_left)
        rev.append(float((p * s).mean()))
        sold.append(float(s.mean()))
    rev, sold = np.array(rev), np.array(sold)
    k_free = int(np.argmax(rev))
    allowed = (grid >= lo) & (grid <= hi)
    k = int(np.flatnonzero(allowed)[np.argmax(rev[allowed])]) if allowed.any() else k_free
    k_ref = int(np.argmin(abs(grid - ref)))
    return {"grid": grid, "rev": rev, "sold": sold, "k": k, "k_free": k_free, "k_ref": k_ref,
            "lo": lo, "hi": hi, "rate": grid[k], "draws": draws}


def protection_level(to_come: float, sd: float, late_share: float, discount_ratio: float, rooms_left: float) -> float:
    """Two-class EMSR-b: rooms to hold back for late full-rate demand."""
    mu, s = late_share * to_come, max(np.sqrt(max(late_share, 1e-6)) * sd, 0.5)
    return float(np.clip(mu + s * norm.ppf(1 - discount_ratio), 0, rooms_left))


# ---------------------------------------------------------------------------
# Switchback experiment
# ---------------------------------------------------------------------------
def nights_needed(sd: float, mean: float, mde: float, rho: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Total nights for a two-arm switchback on a nightly metric; rho is the covariate correlation."""
    delta = mde * mean
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    per_arm = 2 * (z ** 2) * (sd ** 2) * (1 - rho ** 2) / delta ** 2
    return int(np.ceil(2 * per_arm))


@st.cache_data(show_spinner=False)
def switchback(nights: int, true_lift: float = 0.045, seed: int = 5) -> pd.DataFrame:
    """Simulated pilot: nights alternate in random two-night blocks between manual and model pricing."""
    rng = np.random.default_rng(seed)
    h = rate_history()
    base = h["revpar"].to_numpy()
    start = rng.integers(0, max(1, len(base) - nights))
    expected = pd.Series(base).rolling(7, min_periods=1).mean().to_numpy()[start:start + nights]
    blocks = rng.integers(0, 2, int(np.ceil(nights / 2)))
    arm = np.repeat(blocks, 2)[:nights]
    common = rng.normal(0, 0.085, nights)        # demand swings the forecast can see
    idio = rng.normal(0, 0.05, nights)           # noise nobody can see
    revpar = expected * (1 + common + idio) * (1 + true_lift * arm)
    forecast = expected * (1 + common + rng.normal(0, 0.02, nights))  # the pre-registered covariate
    return pd.DataFrame({"night": np.arange(1, nights + 1), "arm": arm, "revpar": revpar, "forecast": forecast})


def readout(df: pd.DataFrame) -> dict:
    y, x, a = df["revpar"].to_numpy(), df["forecast"].to_numpy(), df["arm"].to_numpy()

    def diff(v: np.ndarray) -> tuple[float, float]:
        t_, c_ = v[a == 1], v[a == 0]
        se = np.sqrt(t_.var(ddof=1) / len(t_) + c_.var(ddof=1) / len(c_))
        return t_.mean() - c_.mean(), se

    raw, raw_se = diff(y)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    adj, adj_se = diff(y - theta * (x - x.mean()))
    base = y[a == 0].mean()
    return {"raw": raw / base, "raw_se": raw_se / base, "cuped": adj / base, "cuped_se": adj_se / base,
            "rho": float(np.corrcoef(y, x)[0, 1]), "control": base, "n": len(df)}


# ---------------------------------------------------------------------------
# Monitoring after launch
# ---------------------------------------------------------------------------
def psi(ref: np.ndarray, cur: np.ndarray, bins: np.ndarray) -> float:
    r = np.histogram(ref, bins)[0] / len(ref)
    c = np.histogram(cur, bins)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


@st.cache_data(show_spinner=False)
def monitoring(shift: bool, mape_limit: float, weeks_over: int, psi_limit: float, seed: int = 11) -> pd.DataFrame:
    """26 weeks after launch. With a shift, a competitor opens in week 12 and over five weeks guests book later."""
    rng = np.random.default_rng(seed)
    ref_leads = rng.gamma(2.2, 9.0, 6000)
    bins = np.array([0, 3, 7, 14, 21, 30, 45, 60, 400])
    rows, retrained_at, over = [], None, 0
    for w in range(1, 27):
        # Booking behaviour moves first; forecast errors follow as the affected nights arrive.
        ramp = min(1.0, max(0.0, (w - 11) / 3)) if shift else 0.0
        err_ramp = min(1.0, max(0.0, (w - 13) / 4)) if shift else 0.0
        shifted = ramp > 0 and retrained_at is None
        scale = 9.0 - 2.6 * ramp
        leads = rng.gamma(2.2, scale, 900)
        base_err = 0.072 + rng.normal(0, 0.006)
        drift_err = 0.075 * err_ramp if shifted else 0.0
        if retrained_at is not None and w == retrained_at + 1:
            drift_err = 0.012  # first week on the refreshed model
        err = base_err + drift_err
        p = psi(ref_leads, leads, bins) if retrained_at is None else psi(rng.gamma(2.2, scale, 6000), leads, bins)
        over = over + 1 if err > mape_limit else 0
        trigger = retrained_at is None and (over >= weeks_over or p > psi_limit)
        by = ("drift" if p > psi_limit else "accuracy") if trigger else ""
        rows.append({"week": w, "mape": err, "psi": p, "trigger": trigger, "by": by, "shifted": shifted})
        if trigger:
            retrained_at = w
    return pd.DataFrame(rows)
