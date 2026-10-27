"""Forecasting at scale: baselines, a global gradient-boosting model, hierarchical reconciliation, conformal intervals,
an inventory decision and drift monitoring, all on the Larkspur Market chain (lab/forecast/data.py).

Everything is backtested with rolling origins: the model only ever sees weeks up to the forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from lab.forecast import data as D

H = 13                                     # forecast horizon, weeks (a quarter)
ORIGINS = [103, 116, 129, 142]             # last observed week of each backtest fold
SEED = 11


# ---------------------------------------------------------------- baselines
def seasonal_naive(Y: np.ndarray, o: int, h: int = H) -> np.ndarray:
    """Same week last year. Y: weeks x series. Returns h x series."""
    return np.stack([Y[o + k - 52] for k in range(1, h + 1)])


def ets_lite(Y: np.ndarray, o: int, h: int = H, alpha=.25, beta=.02, gamma=.15, phi=.98) -> np.ndarray:
    """Damped-trend Holt-Winters with multiplicative yearly seasonality, run on every series at once."""
    Y = np.asarray(Y, float)
    lvl = Y[:52].mean(0) + 1e-6
    tr = np.zeros(Y.shape[1])
    s = Y[:52] / lvl                                                    # 52 x series
    for t in range(52, o + 1):
        si = s[t % 52]
        prev = lvl
        lvl = alpha * Y[t] / np.maximum(si, 1e-3) + (1 - alpha) * (prev + phi * tr)
        tr = beta * (lvl - prev) + (1 - beta) * phi * tr
        s[t % 52] = gamma * Y[t] / np.maximum(lvl, 1e-6) + (1 - gamma) * si
    damp = np.cumsum(phi ** np.arange(1, h + 1))
    return np.stack([(lvl + damp[k - 1] * tr) * s[(o + k) % 52] for k in range(1, h + 1)])


# ---------------------------------------------------------------- global gradient boosting (direct, multi-horizon)
def _features(Y, P, hol, o, ks, codes):
    """One row per (series, horizon) for origin o. Everything here is known at o."""
    L = np.log1p(Y)
    n = Y.shape[1]
    recent4, recent13, last = L[o - 3:o + 1].mean(0), L[o - 12:o + 1].mean(0), L[o]
    ly_recent = L[o - 52 - 12:o - 52 + 1].mean(0)                        # same 13 weeks last year
    rows = []
    for k in ks:
        t = o + k
        woy = t % 52
        rows.append(np.column_stack([
            L[t - 52], recent4, recent13, last, recent13 - ly_recent, L[t - 52] + (recent13 - ly_recent),
            np.full(n, np.sin(2 * np.pi * woy / 52)), np.full(n, np.cos(2 * np.pi * woy / 52)),
            P[t], P[t - 52], np.full(n, hol[t]), np.full(n, hol[t - 52]), np.full(n, k), codes]))
    return np.vstack(rows)


FEATURES = ["log sales, same week last year", "log mean, last 4 weeks", "log mean, last 13 weeks", "log sales, last week",
            "year-over-year growth, last 13 weeks", "last year × growth", "week of year (sin)", "week of year (cos)",
            "promotion this week", "promotion same week last year", "holiday this week", "holiday same week last year",
            "weeks ahead", "store", "category", "region"]


def gbm_forecast(Y, P, hol, codes, o, h=H, step=2, seed=SEED):
    """Train on every past origin whose targets are already observed by week o, then forecast o+1..o+h."""
    Xs, ys = [], []
    for oo in range(64, o - 1, step):
        ks = [k for k in range(1, h + 1) if oo + k <= o]
        if not ks:
            continue
        Xs.append(_features(Y, P, hol, oo, ks, codes))
        ys.append(np.concatenate([np.log1p(Y[oo + k]) for k in ks]))
    X, y = np.vstack(Xs), np.concatenate(ys)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, categorical_features=[13, 14, 15], random_state=seed)
    m.fit(X, y)
    Xf = _features(Y, P, hol, o, range(1, h + 1), codes)
    f = np.expm1(m.predict(Xf)).reshape(h, Y.shape[1])
    return np.maximum(f, 0), m, len(y)


# ---------------------------------------------------------------- metrics
def wape(y, f):
    return float(np.abs(y - f).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, f):
    return float((f - y).sum() / max(np.abs(y).sum(), 1e-9))


def mase(y, f, Y, o):
    scale = np.abs(Y[52:o + 1] - Y[:o + 1 - 52]).mean(0)                # in-sample seasonal-naive error
    return float((np.abs(y - f) / np.maximum(scale, 1e-9)).mean())


# ---------------------------------------------------------------- reconciliation
def reconcile(S: np.ndarray, base: np.ndarray, how: str) -> np.ndarray:
    """base: nodes x h. Returns coherent forecasts for every node."""
    nb = S.shape[1]
    if how == "bottom-up":
        return S @ base[-nb:]
    if how == "ols":
        W = np.eye(S.shape[0])
    elif how == "wls":                                                   # structural scaling: weight by series count
        W = np.diag(S.sum(1))
    else:
        raise ValueError(how)
    Wi = np.linalg.inv(W)
    G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    return S @ (G @ base)


# ---------------------------------------------------------------- the full study
@dataclass
class Study:
    chain: D.Chain
    columns: list
    nodes: list
    S: np.ndarray
    folds: list = field(default_factory=list)       # per origin: dict of forecasts and actuals
    scores: pd.DataFrame = None
    by_level: pd.DataFrame = None
    coverage: dict = None
    inventory: pd.DataFrame = None
    tracking: pd.DataFrame = None
    importance: pd.DataFrame = None
    train_rows: int = 0


def run(chain: D.Chain | None = None) -> Study:
    chain = chain or D.simulate()
    Y, P, hol = chain.wide.values.astype(float), chain.promo.values, chain.holiday
    cols = list(chain.wide.columns)
    st = {s: i for i, s in enumerate(sorted({c.split('|')[0] for c in cols}))}
    ct = {c: i for i, c in enumerate(D.CATEGORIES)}
    rg = {s: i for i, (r, ss) in enumerate(D.REGIONS.items()) for s in ss}
    codes = np.array([[st[c.split('|')[0]], ct[c.split('|')[1]], rg[c.split('|')[0]]] for c in cols], float)
    nodes, S = D.hierarchy(cols)
    study = Study(chain, cols, nodes, S)
    rows = []
    for o in ORIGINS:
        y = Y[o + 1:o + 1 + H]
        f_sn, f_ets = seasonal_naive(Y, o), ets_lite(Y, o)
        f_gbm, model, ntr = gbm_forecast(Y, P, hol, codes, o)
        study.train_rows = ntr
        agg = Y @ S.T                                                   # weeks x nodes
        base = ets_lite(agg, o).T                                       # nodes x h, independent base forecasts
        base[-len(cols):] = f_gbm.T                                     # bottom level from the global model
        rec = {k: reconcile(S, base, k) for k in ("bottom-up", "wls")}
        study.folds.append(dict(o=o, y=y, sn=f_sn, ets=f_ets, gbm=f_gbm, base=base, rec=rec, model=model, agg_y=agg[o + 1:o + 1 + H].T))
        for name, f in (("Seasonal naive", f_sn), ("Holt-Winters (damped)", f_ets), ("Global gradient boosting", f_gbm)):
            rows.append({"origin": o, "model": name, "WAPE": wape(y, f), "Bias": bias(y, f), "MASE": mase(y, f, Y, o)})
    study.scores = pd.DataFrame(rows)

    # accuracy by level: incoherent base vs reconciled
    lv = np.array([D.level_of(n) for n in nodes])
    out = []
    for fd in study.folds:
        for name, F in (("Base (incoherent)", fd["base"]), ("Bottom-up", fd["rec"]["bottom-up"]), ("Reconciled (WLS)", fd["rec"]["wls"])):
            for L in ["Total", "Region", "Store", "Category", "Region × category", "Store × category"]:
                m = lv == L
                out.append({"origin": fd["o"], "method": name, "level": L, "WAPE": wape(fd["agg_y"][m], F[m])})
    study.by_level = pd.DataFrame(out).groupby(["method", "level"], sort=False)["WAPE"].mean().reset_index()

    # conformal intervals: residual ratios from the first three folds, coverage on the last
    ratio = np.concatenate([(fd["y"] / np.maximum(fd["gbm"], 1) - 1).reshape(H, -1) for fd in study.folds[:-1]], axis=1)   # h x (series*folds)
    last = study.folds[-1]
    cov = {}
    for lvl in (.8, .95):
        lo, hi = np.quantile(ratio, (1 - lvl) / 2, axis=1), np.quantile(ratio, 1 - (1 - lvl) / 2, axis=1)
        L, U = last["gbm"] * (1 + lo[:, None]), last["gbm"] * (1 + hi[:, None])
        cov[lvl] = dict(lo=lo, hi=hi, coverage=float(((last["y"] >= L) & (last["y"] <= U)).mean()), L=L, U=U)
    study.coverage = cov
    study.ratio = ratio

    # inventory: newsvendor order at the critical ratio, on the last fold, by forecasting method
    price = np.array([D.CAT[c.split('|')[1]][4] for c in cols])
    waste = np.array([{"Produce": .45, "Bakery": .55, "Dairy": .30}.get(c.split('|')[1], .12) for c in cols])
    cu, co = price * .30, price * waste                                 # lost margin vs cost of leftover
    cr = cu / (cu + co)
    inv = []
    sn_ratio = np.concatenate([(fd["y"] / np.maximum(fd["sn"], 1) - 1).reshape(H, -1) for fd in study.folds[:-1]], axis=1)
    for name, f, rr in (("Seasonal naive", last["sn"], sn_ratio), ("Global gradient boosting", last["gbm"], ratio)):
        q = np.array([np.quantile(rr, c) for c in cr])                  # per-series quantile of the residual ratio
        order = f * (1 + q[None, :])
        y = last["y"]
        cost = (cu * np.maximum(y - order, 0) + co * np.maximum(order - y, 0)).sum()
        fill = float(np.minimum(order, y).sum() / y.sum())
        inv.append({"method": name, "cost": float(cost), "fill_rate": fill, "units_left": float(np.maximum(order - y, 0).sum())})
    inv.append({"method": "Perfect foresight", "cost": 0.0, "fill_rate": 1.0, "units_left": 0.0})
    study.inventory = pd.DataFrame(inv)
    study.cr = cr

    # monitoring: each store's weekly sales against "same week last year x recent year-over-year growth"; the
    # 8-week cumulative percentage error, scaled by its normal spread, is a z-score. Growth is measured on weeks
    # 9 to 34 back, so a sudden shift is not absorbed into the expectation before it can be seen.
    stores = sorted(st)
    store_y = np.stack([Y[:, [i for i, c in enumerate(cols) if c.startswith(s)]].sum(1) for s in stores], axis=1)
    exp = np.full_like(store_y, np.nan)
    for t in range(88, D.WEEKS):
        g = store_y[t - 35:t - 9].sum(0) / store_y[t - 35 - 52:t - 9 - 52].sum(0)
        exp[t] = store_y[t - 52] * g
    pe = (store_y - exp) / exp
    sd = pd.DataFrame(pe).rolling(26, min_periods=12).std().shift(8).values
    z = pd.DataFrame(pe).rolling(8).sum().values / (sd * np.sqrt(8))
    study.tracking = pd.DataFrame(z, columns=stores)
    study.store_y, study.store_f = pd.DataFrame(store_y, columns=stores), pd.DataFrame(exp, columns=stores)

    # what the model leans on: permutation importance on the last fold's own features (cheap version: gain proxy)
    m = last["model"]
    Xf = _features(Y, P, hol, last["o"], range(1, H + 1), codes)
    yt = np.log1p(last["y"]).reshape(-1)
    base_err = np.mean((m.predict(Xf) - yt) ** 2)
    rng = np.random.default_rng(SEED)
    imp = []
    for j, name in enumerate(FEATURES):
        Xp = Xf.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append({"feature": name, "increase in error": float(np.mean((m.predict(Xp) - yt) ** 2) - base_err)})
    study.importance = pd.DataFrame(imp).sort_values("increase in error", ascending=False)
    return study


def alerts(study: Study, threshold: float = 4.0) -> pd.DataFrame:
    """First week each store's drift z-score crosses the threshold."""
    out = []
    for s in study.tracking.columns:
        ts = study.tracking[s]
        hit = ts[ts.abs() > threshold]
        out.append({"store": s, "first alert week": int(hit.index[0]) if len(hit) else None, "latest signal": float(ts.iloc[-1])})
    return pd.DataFrame(out)
