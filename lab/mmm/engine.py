"""Marketing mix model and budget optimizer for Corvane Connect (fictional).

Three years of weekly spend across five acquisition channels and new subscribers, with planted carryover (adstock),
diminishing returns (saturation), seasonality, promotions and a geo holdout experiment on paid social. The model
recovers channel effects, is calibrated to the experiment, and an optimizer reallocates the budget under guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import lsq_linear, minimize

CHANNELS = ["Paid search", "Paid social", "Connected TV", "Direct mail", "Partners"]
TRUE = {  # adstock decay, half-saturation ($K a week, adstocked), max weekly subscribers, Hill shape
    "Paid search":  dict(decay=0.10, half=140.0, top=520.0, shape=1.2),
    "Paid social":  dict(decay=0.30, half=75.0, top=190.0, shape=1.3),
    "Connected TV": dict(decay=0.65, half=420.0, top=560.0, shape=1.6),
    "Direct mail":  dict(decay=0.35, half=90.0, top=210.0, shape=1.4),
    "Partners":     dict(decay=0.05, half=150.0, top=420.0, shape=1.1),
}
WEEKS = 156
CLV_NEW = 1500.0             # lifetime value of a new subscriber, from the subscriber lab
TEST_WEEKS = (130, 138)      # geo holdout on paid social
TEST_SHARE = 0.20            # share of regions where social was switched off


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill(a: np.ndarray, half: float, shape: float) -> np.ndarray:
    a = np.maximum(a, 0)
    return a ** shape / (a ** shape + half ** shape)


@dataclass
class Data:
    df: pd.DataFrame
    truth_contrib: pd.DataFrame
    experiment: dict


@st.cache_data(show_spinner=False)
def data(seed: int = 12) -> Data:
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    season = 1 + 0.18 * np.sin(2 * np.pi * (t - 6) / 52) + 0.06 * np.cos(4 * np.pi * t / 52)
    holiday = np.isin(t % 52, [46, 47, 48]).astype(float)
    promo = (rng.random(WEEKS) < 0.12).astype(float)
    spend = {}
    # Managers spend more when demand is high: the trap that makes raw correlations overstate media.
    spend["Paid search"] = 110 * season * rng.lognormal(0, 0.28, WEEKS)
    spend["Paid social"] = 85 * season ** 2.2 * rng.lognormal(0, 0.25, WEEKS)
    flights = (np.floor(t / 6) % 2 == 0).astype(float)                    # six weeks on, six off
    spend["Connected TV"] = 260 * flights * rng.lognormal(0, 0.1, WEEKS)
    drops = (t % 4 == 0).astype(float)
    spend["Direct mail"] = 190 * drops * rng.lognormal(0, 0.15, WEEKS)
    spend["Partners"] = 55 * rng.lognormal(0, 0.35, WEEKS)
    base = (1350 + 1.6 * t) * season * (1 + 0.25 * holiday) * (1 + 0.10 * promo)
    contrib = {}
    for c in CHANNELS:
        p = TRUE[c]
        contrib[c] = p["top"] * hill(adstock(spend[c], p["decay"]), p["half"], p["shape"]) * (0.85 + 0.15 * season)
    # The geo test: social off in 20% of regions for eight weeks, so those regions lose their share of its effect.
    test = (t >= TEST_WEEKS[0]) & (t < TEST_WEEKS[1])
    lost = np.where(test, contrib["Paid social"] * TEST_SHARE, 0.0)
    y = (base + sum(contrib.values()) - lost) * rng.lognormal(0, 0.035, WEEKS)
    df = pd.DataFrame({"week": t, "new_subs": y, "promo": promo, "holiday": holiday, **{c: spend[c] for c in CHANNELS}})
    df["date"] = pd.Timestamp("2023-10-02") + pd.to_timedelta(t * 7, unit="D")
    truth = pd.DataFrame({"base": base, **contrib})
    # What the experiment reports: incremental subscribers per week from social at current spend, with noise.
    true_inc = float(contrib["Paid social"][test].mean())
    measured = true_inc * (1 + rng.normal(0, 0.08))
    exp = {"weekly_incremental": measured, "se": true_inc * 0.10, "true": true_inc,
           "spend": float(spend["Paid social"][test].mean()), "weeks": TEST_WEEKS}
    return Data(df, truth, exp)


def _controls(df: pd.DataFrame) -> np.ndarray:
    t = df["week"].to_numpy()
    cols = [np.ones(len(df)), t / 52]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * t / 52), np.cos(2 * np.pi * k * t / 52)]
    cols += [df["holiday"].to_numpy(), df["promo"].to_numpy()]
    return np.column_stack(cols)


DECAYS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
HALF_MULT = [0.5, 1.0, 1.5, 2.5]


@dataclass
class Fit:
    decay: dict
    half: dict
    shape: float
    beta: dict           # max weekly subscribers per channel
    ctrl: np.ndarray
    mape: float
    calibrated: bool


def _features(df, decay, half, shape):
    return np.column_stack([hill(adstock(df[c].to_numpy(), decay[c]), half[c], shape) for c in CHANNELS])


def _solve(df, decay, half, shape, rows=None, prior=None):
    C = _controls(df)
    F = _features(df, decay, half, shape)
    X = np.column_stack([C, F])
    y = df["new_subs"].to_numpy()
    if prior is not None:  # pseudo-observation pulling social's implied effect toward the experiment
        row, target, weight = prior
        X = np.vstack([X, weight * row])
        y = np.concatenate([y, [weight * target]])
    lb = np.r_[np.full(C.shape[1], -np.inf), np.zeros(len(CHANNELS))]
    res = lsq_linear(X, y, bounds=(lb, np.full(X.shape[1], np.inf)), method="bvls")
    return res.x[:C.shape[1]], res.x[C.shape[1]:]


def _sse(df, decay, half, shape, prior=None):
    c, b = _solve(df, decay, half, shape, prior=prior)
    pred = _controls(df) @ c + _features(df, decay, half, shape) @ b
    return float(((df["new_subs"].to_numpy() - pred) ** 2).sum())


def _prior_row(df, decay, half, shape, exp, weight):
    test = (df["week"] >= exp["weeks"][0]) & (df["week"] < exp["weeks"][1])
    F = _features(df, decay, half, shape)
    row = np.zeros(_controls(df).shape[1] + len(CHANNELS))
    j = _controls(df).shape[1] + CHANNELS.index("Paid social")
    row[j] = F[test.to_numpy(), CHANNELS.index("Paid social")].mean() * TEST_SHARE
    return (row, exp["weekly_incremental"] * TEST_SHARE, weight)


@st.cache_data(show_spinner=False)
def fit(calibrated: bool = False, holdout: int = 26) -> Fit:
    d = data()
    df = d.df
    shape = 1.3
    decay = {c: 0.3 for c in CHANNELS}
    mean_spend = {c: float(df[c].mean()) for c in CHANNELS}
    half = {c: mean_spend[c] for c in CHANNELS}
    weight = 40.0 if calibrated else 0.0

    def prior_for(dec, hf):
        return _prior_row(df, dec, hf, shape, d.experiment, weight) if calibrated else None

    for _ in range(2):  # coordinate search over carryover and saturation, channel by channel
        for c in CHANNELS:
            best = None
            for dc in DECAYS:
                for hm in HALF_MULT:
                    dec = {**decay, c: dc}
                    hf = {**half, c: mean_spend[c] / (1 - dc) * hm}
                    s = _sse(df, dec, hf, shape, prior_for(dec, hf))
                    if best is None or s < best[0]:
                        best = (s, dc, hf[c])
            decay[c], half[c] = best[1], best[2]
    # Out-of-sample check: refit on all but the last 26 weeks, predict them.
    tr, te = df.iloc[:-holdout], df.iloc[-holdout:]
    c_tr, b_tr = _solve(tr, decay, half, shape)
    Fall = _features(df, decay, half, shape)
    pred_te = _controls(df) @ c_tr + Fall @ b_tr
    mape = float(np.mean(np.abs(pred_te[-holdout:] - te["new_subs"].to_numpy()) / te["new_subs"].to_numpy()))
    ctrl, beta = _solve(df, decay, half, shape, prior=prior_for(decay, half))
    return Fit(decay, half, shape, dict(zip(CHANNELS, beta)), ctrl, mape, calibrated)


def decompose(f: Fit) -> pd.DataFrame:
    df = data().df
    F = _features(df, f.decay, f.half, f.shape)
    out = pd.DataFrame({"week": df["week"], "date": df["date"], "actual": df["new_subs"], "base": _controls(df) @ f.ctrl})
    for i, c in enumerate(CHANNELS):
        out[c] = F[:, i] * f.beta[c]
    out["fitted"] = out["base"] + out[CHANNELS].sum(1)
    return out


def steady_response(f: Fit, c: str, weekly_spend) -> np.ndarray:
    """New subscribers a week from channel c if spend held at this level (adstock at steady state)."""
    x = np.asarray(weekly_spend, dtype=float) / (1 - f.decay[c])
    return f.beta[c] * hill(x, f.half[c], f.shape)


def true_response(c: str, weekly_spend) -> np.ndarray:
    p = TRUE[c]
    x = np.asarray(weekly_spend, dtype=float) / (1 - p["decay"])
    return p["top"] * hill(x, p["half"], p["shape"])


def roi_table(f: Fit) -> pd.DataFrame:
    df = data().df
    dec = decompose(f)
    rows = []
    for c in CHANNELS:
        spend = df[c].sum() * 1000
        subs = dec[c].sum()
        cur = df[c].tail(52).mean()
        marg = (steady_response(f, c, cur * 1.01) - steady_response(f, c, cur)) / (cur * 0.01 * 1000)
        t_subs = data().truth_contrib[c].sum()
        rows.append({"channel": c, "spend": spend, "subs": subs, "cost_per_sub": spend / max(subs, 1e-9),
                     "roi": subs * CLV_NEW / spend, "mroi": float(marg) * CLV_NEW, "true_roi": t_subs * CLV_NEW / spend,
                     "weekly_spend": cur})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def bootstrap(calibrated: bool, reps: int = 60, block: int = 8) -> pd.DataFrame:
    """Block bootstrap of channel ROI with carryover and saturation held at their fitted values."""
    f = fit(calibrated)
    d = data()
    df = d.df
    rng = np.random.default_rng(1)
    nb = WEEKS // block
    out = []
    F = _features(df, f.decay, f.half, f.shape)
    C = _controls(df)
    y = df["new_subs"].to_numpy()
    spend_tot = np.array([df[c].sum() * 1000 for c in CHANNELS])
    prior = _prior_row(df, f.decay, f.half, f.shape, d.experiment, 40.0) if calibrated else None
    for _ in range(reps):
        idx = np.concatenate([np.arange(b * block, (b + 1) * block) for b in rng.integers(0, nb, nb)])
        X = np.column_stack([C[idx], F[idx]])
        yy = y[idx]
        if prior is not None:
            X = np.vstack([X, prior[2] * prior[0]])
            yy = np.concatenate([yy, [prior[2] * prior[1]]])
        lb = np.r_[np.full(C.shape[1], -np.inf), np.zeros(len(CHANNELS))]
        b = lsq_linear(X, yy, bounds=(lb, np.full(X.shape[1], np.inf)), method="bvls").x[C.shape[1]:]
        subs = (F * b).sum(0)
        out.append(subs * CLV_NEW / spend_tot)
    return pd.DataFrame(out, columns=CHANNELS)


def optimize(f: Fit, budget: float, guard: float) -> pd.DataFrame:
    """Weekly budget ($K) across channels to maximize new subscribers, each channel within ±guard of today."""
    cur = np.array([data().df[c].tail(52).mean() for c in CHANNELS])
    scaled = cur * budget / cur.sum()            # today's mix at this budget is always allowed
    lo, hi = np.minimum(cur * (1 - guard), scaled), np.maximum(cur * (1 + guard), scaled)

    def neg(x):
        return -sum(float(steady_response(f, c, x[i])) for i, c in enumerate(CHANNELS))

    x0 = np.clip(cur * budget / cur.sum(), lo, hi)
    res = minimize(neg, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - budget}], options={"maxiter": 500, "ftol": 1e-9})
    x = res.x
    out = pd.DataFrame({"channel": CHANNELS, "current": cur, "optimal": x})
    out["subs_current"] = [float(steady_response(f, c, cur[i])) for i, c in enumerate(CHANNELS)]
    out["subs_optimal"] = [float(steady_response(f, c, x[i])) for i, c in enumerate(CHANNELS)]
    out["true_current"] = [float(true_response(c, cur[i])) for i, c in enumerate(CHANNELS)]
    out["true_optimal"] = [float(true_response(c, x[i])) for i, c in enumerate(CHANNELS)]
    return out


@st.cache_data(show_spinner=False)
def frontier(calibrated: bool, guard: float) -> pd.DataFrame:
    f = fit(calibrated)
    cur = sum(data().df[c].tail(52).mean() for c in CHANNELS)
    rows = []
    for m in np.linspace(0.6, 1.4, 17):
        o = optimize(f, cur * m, guard)
        rows.append({"budget": cur * m, "subs": o["subs_optimal"].sum(), "subs_current_mix": float(sum(
            steady_response(f, c, data().df[c].tail(52).mean() * m) for c in CHANNELS))})
    return pd.DataFrame(rows)
