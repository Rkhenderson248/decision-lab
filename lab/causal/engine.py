"""Causal impact lab: measuring a program that was not randomized.

Corvane Connect (fictional) launched a loyalty price lock in eight of forty markets in week 52, in markets chosen by
regional managers, and separately offered an autopay discount that customers chose to join. Both are synthetic, and
the true effects are planted, so every estimator can be scored against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MARKETS = ["Albany", "Akron", "Boise", "Charleston", "Chattanooga", "Colorado Springs", "Columbia", "Des Moines",
           "Durham", "El Paso", "Eugene", "Fort Wayne", "Fresno", "Grand Rapids", "Greenville", "Harrisburg",
           "Knoxville", "Lansing", "Lexington", "Little Rock", "Madison", "McAllen", "Mobile", "Modesto",
           "Ogden", "Omaha", "Pensacola", "Peoria", "Provo", "Reno", "Rochester", "Salem", "Savannah", "Spokane",
           "Springfield", "Syracuse", "Toledo", "Tucson", "Tulsa", "Wichita"]
WEEKS = 104
LAUNCH = 52
TRUE_EFFECT = -0.40          # churners per 1,000 subscribers a week, once fully ramped
RAMP = 6
N_TREATED = 8
TRUE_ATT_MATCH = -0.020      # autopay discount: 2-point drop in 6-month churn


@dataclass
class Panel:
    df: pd.DataFrame          # market, week, rate, treated, post, d
    treated: list
    scenario: str


@st.cache_data(show_spinner=False)
def panel(scenario: str = "level", seed: int = 7) -> Panel:
    """scenario 'level': managers picked high-churn markets (parallel trends hold).
    scenario 'trend': managers picked markets whose churn was already rising (parallel trends fail)."""
    rng = np.random.default_rng(seed)
    n = len(MARKETS)
    level = rng.normal(2.6, 0.45, n)
    trend = rng.normal(0, 0.0015, n)
    t = np.arange(WEEKS)
    if scenario == "level":
        treated_idx = np.argsort(-level)[:N_TREATED]
    else:
        rising = rng.choice(n, 2 * N_TREATED, replace=False)       # sixteen markets were already getting worse
        trend[rising] = rng.normal(0.02, 0.002, len(rising))
        treated_idx = rising[:N_TREATED]                            # managers picked eight of them
    season = 0.28 * np.sin(2 * np.pi * (t + 8) / 52) + 0.12 * np.sin(4 * np.pi * t / 52)
    size = rng.integers(8_000, 60_000, n)
    rows = []
    is_t = np.zeros(n, bool)
    is_t[treated_idx] = True
    for i, m in enumerate(MARKETS):
        e = np.zeros(WEEKS)
        for k in range(1, WEEKS):
            e[k] = 0.55 * e[k - 1] + rng.normal(0, 0.12)
        mu = level[i] + trend[i] * (t - LAUNCH) + season * (1 + 0.08 * rng.normal()) + e
        ramp = np.clip((t - LAUNCH + 1) / 6, 0, 1)            # the effect builds over six weeks
        mu = mu + TRUE_EFFECT * ramp * is_t[i]
        obs = rng.poisson(np.clip(mu, 0.2, None) * size[i] / 1000) / size[i] * 1000
        for k in range(WEEKS):
            rows.append((m, k, obs[k], is_t[i], k >= LAUNCH, int(is_t[i] and k >= LAUNCH), size[i]))
    df = pd.DataFrame(rows, columns=["market", "week", "rate", "treated", "post", "d", "subscribers"])
    return Panel(df, [MARKETS[i] for i in treated_idx], scenario)


# ---------------------------------------------------------------------------
# Naive comparisons
# ---------------------------------------------------------------------------
def naive(p: Panel) -> dict:
    df = p.df
    tr = df[df["treated"]]
    # The dashboard comparison: the twelve weeks after launch against the twelve weeks before.
    before_after = (tr[tr["week"].between(LAUNCH, LAUNCH + 11)]["rate"].mean()
                    - tr[tr["week"].between(LAUNCH - 12, LAUNCH - 1)]["rate"].mean())
    post = df[df["post"]]
    vs_control = post[post["treated"]]["rate"].mean() - post[~post["treated"]]["rate"].mean()
    return {"12 weeks after vs 12 weeks before": before_after, "Treated vs untreated (after launch)": vs_control}


# ---------------------------------------------------------------------------
# Difference-in-differences with two-way fixed effects
# ---------------------------------------------------------------------------
def did(p: Panel) -> dict:
    df = p.df[(p.df["week"] < LAUNCH) | (p.df["week"] >= LAUNCH + RAMP)]   # skip the six-week ramp
    y = df.pivot(index="market", columns="week", values="rate")
    d = df.pivot(index="market", columns="week", values="d").astype(float)

    def demean(a):
        return a - a.mean(1).to_numpy()[:, None] - a.mean(0).to_numpy()[None, :] + a.to_numpy().mean()

    yt, dt = demean(y), demean(d)
    beta = float((dt * yt).to_numpy().sum() / (dt ** 2).to_numpy().sum())
    resid = yt - beta * dt
    g = (dt * resid).sum(1).to_numpy()
    G = len(g)
    se = float(np.sqrt((G / (G - 1)) * (g ** 2).sum()) / (dt ** 2).to_numpy().sum())
    return {"estimate": beta, "se": se, "lo": beta - 1.96 * se, "hi": beta + 1.96 * se}


@st.cache_data(show_spinner=False)
def event_study(scenario: str, bin_weeks: int = 4) -> pd.DataFrame:
    """Effect by weeks relative to launch, binned, with market and week fixed effects. Bin −1 is the reference."""
    p = panel(scenario)
    df = p.df.copy()
    rel = df["week"] - LAUNCH
    df["bin"] = np.where(df["treated"], np.floor(rel / bin_weeks).astype(int), -999)
    bins = sorted(b for b in df["bin"].unique() if b not in (-999, -1))
    X = [np.ones(len(df))]
    names = ["const"]
    for m in MARKETS[1:]:
        X.append((df["market"] == m).to_numpy(float))
        names.append("m")
    for w in range(1, WEEKS):
        X.append((df["week"] == w).to_numpy(float))
        names.append("w")
    for b in bins:
        X.append((df["bin"] == b).to_numpy(float))
        names.append(f"b{b}")
    X = np.column_stack(X)
    y = df["rate"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # Cluster-robust covariance by market.
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for m in MARKETS:
        idx = (df["market"] == m).to_numpy()
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    cov = XtX_inv @ meat @ XtX_inv
    k0 = len(names) - len(bins)
    out = pd.DataFrame({"bin": bins, "estimate": beta[k0:], "se": np.sqrt(np.clip(np.diag(cov)[k0:], 0, None))})
    out = pd.concat([out, pd.DataFrame({"bin": [-1], "estimate": [0.0], "se": [0.0]})]).sort_values("bin")
    out["week"] = out["bin"] * bin_weeks + bin_weeks / 2
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic control
# ---------------------------------------------------------------------------
def _sc_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    k = donors.shape[1]
    obj = lambda w: ((target - donors @ w) ** 2).mean()
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 300, "ftol": 1e-10})
    return res.x


@st.cache_data(show_spinner=False)
def synthetic_control(scenario: str, pre_weeks: int = 52) -> dict:
    """Average of the treated markets against a weighted blend of untreated ones, fitted on the pre-period."""
    p = panel(scenario)
    y = p.df.pivot(index="week", columns="market", values="rate")
    treated = y[p.treated].mean(1).to_numpy()
    donor_names = [m for m in MARKETS if m not in p.treated]
    D = y[donor_names].to_numpy()
    pre = slice(LAUNCH - pre_weeks, LAUNCH)
    # Demeaned synthetic control: match the shape of the pre-period path, allowing a constant level difference.
    t_mu, d_mu = treated[pre].mean(), D[pre].mean(0)
    w = _sc_weights(treated[pre] - t_mu, D[pre] - d_mu)
    synth = (D - d_mu) @ w + t_mu
    gap = treated - synth
    effect = float(gap[LAUNCH + RAMP:].mean())             # after the six-week ramp
    rmspe = lambda g, s: float(np.sqrt((g[s] ** 2).mean()))
    ratio_t = rmspe(gap, slice(LAUNCH, WEEKS)) / max(rmspe(gap, pre), 1e-9)
    placebo = []
    for j, name in enumerate(donor_names):
        others = np.delete(D, j, axis=1)
        o_mu, j_mu = others[pre].mean(0), D[pre, j].mean()
        wj = _sc_weights(D[pre, j] - j_mu, others[pre] - o_mu)
        gj = D[:, j] - ((others - o_mu) @ wj + j_mu)
        placebo.append({"market": name, "gap": gj, "ratio": rmspe(gj, slice(LAUNCH, WEEKS)) / max(rmspe(gj, pre), 1e-9),
                        "effect": float(gj[LAUNCH + RAMP:].mean())})
    rank = 1 + sum(pl["ratio"] >= ratio_t for pl in placebo)
    weights = pd.Series(w, index=donor_names).sort_values(ascending=False)
    return {"treated": treated, "synthetic": synth, "gap": gap, "effect": effect, "weights": weights,
            "placebo": placebo, "p_value": rank / (len(placebo) + 1), "pre_rmspe": rmspe(gap, pre)}


# ---------------------------------------------------------------------------
# Propensity matching (customer level)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def customers(hidden_confounder: bool = False, seed: int = 8, n: int = 24_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.gamma(2.2, 12, n)
    age = np.clip(rng.normal(44, 14, n), 19, 90)
    digital = rng.beta(2.5, 2.0, n)
    care = rng.poisson(1.0, n)
    fee = rng.lognormal(np.log(85), 0.35, n)
    engagement = rng.normal(0, 1, n)                     # never observed by the company
    z_opt = -0.9 + 0.025 * tenure - 0.02 * (age - 44) + 2.2 * (digital - 0.55) - 0.25 * care + 0.004 * (fee - 85)
    if hidden_confounder:
        z_opt = z_opt + 0.9 * engagement
    joined = rng.random(n) < 1 / (1 + np.exp(-z_opt))
    z_churn = -1.35 - 0.035 * tenure + 0.006 * (age - 44) - 1.1 * (digital - 0.55) + 0.32 * care + 0.006 * (fee - 85)
    if hidden_confounder:
        z_churn = z_churn - 0.55 * engagement
    p0 = 1 / (1 + np.exp(-z_churn))
    p1 = np.clip(p0 + TRUE_ATT_MATCH * (p0 / p0.mean()), 0, 1)   # effect proportional to risk, averaging -2 pts
    churn = rng.random(n) < np.where(joined, p1, p0)
    return pd.DataFrame({"tenure": tenure, "age": age, "digital": digital, "care": care, "fee": fee, "joined": joined,
                         "churned": churn, "true_effect": (p1 - p0)})


COVARIATES = ["tenure", "age", "digital", "care", "fee"]
COV_LABELS = {"tenure": "Tenure (months)", "age": "Age", "digital": "Digital usage share", "care": "Care calls", "fee": "Monthly fee"}


def smd(a: pd.DataFrame, b: pd.DataFrame, cols=COVARIATES) -> pd.Series:
    return pd.Series({c: (a[c].mean() - b[c].mean()) / np.sqrt((a[c].var() + b[c].var()) / 2) for c in cols})


@st.cache_data(show_spinner=False)
def matching(hidden_confounder: bool = False, caliper: float = 0.2) -> dict:
    c = customers(hidden_confounder)
    X = StandardScaler().fit_transform(c[COVARIATES])
    ps = LogisticRegression(max_iter=1000).fit(X, c["joined"]).predict_proba(X)[:, 1]
    c = c.assign(ps=ps, logit=np.log(ps / (1 - ps)))
    t = c[c["joined"]].sort_values("logit")
    ctrl = c[~c["joined"]].sort_values("logit")
    cal = caliper * c["logit"].std()
    cl = ctrl["logit"].to_numpy()
    idx = np.searchsorted(cl, t["logit"].to_numpy())
    lo = np.clip(idx - 1, 0, len(cl) - 1)
    hi = np.clip(idx, 0, len(cl) - 1)
    pick = np.where(np.abs(cl[lo] - t["logit"].to_numpy()) <= np.abs(cl[hi] - t["logit"].to_numpy()), lo, hi)
    dist = np.abs(cl[pick] - t["logit"].to_numpy())
    ok = dist <= cal                                     # matching with replacement, within the caliper
    mt, mc = t[ok], ctrl.iloc[pick[ok]]
    att = mt["churned"].mean() - mc["churned"].mean()
    # Inverse probability weighting (ATT weights).
    w = np.where(c["joined"], 1.0, ps / (1 - ps))
    ipw = c.loc[c["joined"], "churned"].mean() - np.average(c.loc[~c["joined"], "churned"], weights=w[~c["joined"].to_numpy()])
    naive_diff = c.loc[c["joined"], "churned"].mean() - c.loc[~c["joined"], "churned"].mean()
    # Bootstrap interval for the matched estimate (cheap: resample matched pairs).
    rng = np.random.default_rng(0)
    yt, yc = mt["churned"].to_numpy(float), mc["churned"].to_numpy(float)
    boots = [(yt[s] - yc[s]).mean() for s in (rng.integers(0, len(yt), len(yt)) for _ in range(300))]
    return {"naive": naive_diff, "matched": att, "ipw": ipw, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "truth": float(c.loc[c["joined"], "true_effect"].mean()), "matched_share": float(ok.mean()),
            "smd_before": smd(c[c["joined"]], c[~c["joined"]]), "smd_after": smd(mt, mc), "ps": c[["ps", "joined"]]}


def e_value(rr: float) -> float:
    """VanderWeele–Ding E-value for a risk ratio: how strong an unmeasured confounder would need to be."""
    rr = 1 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1)))
