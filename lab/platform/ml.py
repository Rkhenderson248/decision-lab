"""Train, register, serve and monitor one model on the gold layer.

The decision: which markets will grow purchase lending next year. The model predicts next year's change in purchase
originations from this year's HMDA activity. Training uses 2023 features to predict 2024; the out-of-time check
uses 2024 features to predict 2025; production scores 2025 features for 2026.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "growth": ("This year's origination growth", -1.0, 3.0),
    "origination_rate": ("Origination rate", 0.0, 1.0),
    "denial_rate": ("Denial rate", 0.0, 1.0),
    "refinance_share": ("Refinance share", 0.0, 1.0),
    "log_applications": ("Applications (log)", 0.0, 15.0),
    "log_avg_loan": ("Average loan (log)", 9.0, 16.0),
    "is_metro": ("Metro area", 0.0, 1.0),
}
TARGET = "next_growth"
MIN_APPS = 100
TRAIN_YEAR, TEST_YEAR, SCORE_YEAR = 2023, 2024, 2025


def frame(con) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM gold_market_year_features", con)
    df["log_applications"] = np.log1p(df["applications"])
    df["log_avg_loan"] = np.log(df["avg_loan_amount"].clip(lower=10_000))
    df["is_metro"] = (df["area_type"] == "Metro").astype(float)
    df["growth"] = df["growth"].clip(-0.9, 2.0)
    df[TARGET] = df[TARGET].clip(-0.9, 2.0)
    return df


def split(df: pd.DataFrame):
    ok = df["applications"] >= MIN_APPS
    cols = list(FEATURES)
    tr = df[ok & (df.year == TRAIN_YEAR)].dropna(subset=cols + [TARGET])
    te = df[ok & (df.year == TEST_YEAR)].dropna(subset=cols + [TARGET])
    sc = df[ok & (df.year == SCORE_YEAR)].dropna(subset=cols)
    return tr, te, sc


def _wmae(y, p, w):
    return float(np.average(np.abs(y - p), weights=w))


def _spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _top_decile_lift(y, p):
    q = np.quantile(p, 0.9)
    return float(np.mean(y[p >= q]) - np.mean(y))


@dataclass
class Version:
    version: str
    name: str
    params: str
    model: object
    mae: float
    spearman: float
    lift: float
    train_rows: int
    data_hash: str
    trained_ms: float
    stage: str = "Archived"
    gate: str = ""


class Median:
    """Baseline: every market grows at last year's median rate."""

    def fit(self, X, y, sample_weight=None):
        self.m = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class Persistence:
    """Baseline: next year's growth equals this year's."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.asarray(X["growth"], dtype=float)


def train(con) -> tuple[list[Version], dict]:
    df = frame(con)
    tr, te, sc = split(df)
    cols = list(FEATURES)
    w_tr, w_te = np.sqrt(tr["applications"]), np.sqrt(te["applications"])
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(tr[cols + [TARGET]], index=False).values.tobytes()).hexdigest()[:10]
    cands = [
        ("Median baseline", "growth = median of training year", Median()),
        ("Persistence baseline", "next growth = this year's growth", Persistence()),
        ("Ridge regression", "alpha=1.0, standardized", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Gradient boosting", "max_depth=3, 200 iters, lr=0.05, min_leaf=25",
         HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=25, random_state=7)),
    ]
    versions = []
    for i, (name, params, mdl) in enumerate(cands, 1):
        t0 = time.perf_counter()
        if isinstance(mdl, (Median, Persistence)):
            mdl.fit(tr[cols], tr[TARGET])
        elif hasattr(mdl, "steps"):
            mdl.fit(tr[cols], tr[TARGET], ridge__sample_weight=w_tr)
        else:
            mdl.fit(tr[cols], tr[TARGET], sample_weight=w_tr)
        ms = (time.perf_counter() - t0) * 1000
        p = mdl.predict(te[cols])
        versions.append(Version(f"v{i}", name, params, mdl, _wmae(te[TARGET].values, p, w_te), _spearman(te[TARGET].values, p),
                                _top_decile_lift(te[TARGET].values, p), len(tr), data_hash, ms))
    # Promotion gate. The decision is which markets to prioritize, so the gate is about ranking: rank markets better
    # than chance (rank correlation >= 0.10) and put real growth in the top decile (>= 1 point above average). Error
    # against the best baseline is reported as calibration: a model can rank well and still miss the level.
    base = min(v.mae for v in versions[:2])
    passing = []
    for v in versions[2:]:
        ok = v.spearman >= 0.10 and v.lift >= 0.01
        v.gate = ("Pass" if ok else "Fail") + f" · rank corr {v.spearman:.2f}, top-decile lift {v.lift * 100:+.1f} pts"
        if ok:
            passing.append(v)
    for v in versions[:2]:
        v.gate = "Baseline"
    champ = max(passing, key=lambda v: v.spearman) if passing else versions[0]
    champ.stage = "Production"
    for v in passing:
        if v is not champ:
            v.stage = "Staging"
    info = {"train": tr, "test": te, "score": sc, "baseline_mae": base, "champion": champ, "frame": df}
    return versions, info


def validate(payload: dict) -> list[str]:
    errs = []
    for k, (label, lo, hi) in FEATURES.items():
        if k not in payload:
            errs.append(f"missing field: {k}")
            continue
        try:
            v = float(payload[k])
        except (TypeError, ValueError):
            errs.append(f"{k} must be a number")
            continue
        if not np.isfinite(v) or v < lo or v > hi:
            errs.append(f"{k}={v} outside the allowed range [{lo}, {hi}]")
    return errs


def score(version: Version, payload: dict, reference: pd.DataFrame) -> dict:
    """The scoring service: validate, predict, explain. Reason codes come from moving one feature at a time to the
    training median, so they work for any model."""
    t0 = time.perf_counter()
    errs = validate(payload)
    if errs:
        return {"status": 422, "model_version": version.version, "errors": errs}
    cols = list(FEATURES)
    x = pd.DataFrame([{k: float(payload[k]) for k in cols}])
    pred = float(version.model.predict(x)[0])
    med = reference[cols].median()
    reasons = []
    for k in cols:
        x2 = x.copy()
        x2[k] = med[k]
        d = pred - float(version.model.predict(x2)[0])
        reasons.append((k, d))
    reasons.sort(key=lambda r: -abs(r[1]))
    return {
        "status": 200,
        "model_version": version.version,
        "model": version.name,
        "prediction": {"next_year_origination_growth": round(pred, 4)},
        "reason_codes": [{"feature": FEATURES[k][0], "effect": round(d, 4)} for k, d in reasons[:3] if abs(d) > 1e-6],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e, a = expected.dropna().values, actual.dropna().values
    if len(e) < 20 or len(a) < 20:
        return float("nan")
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    pe = np.histogram(e, qs)[0] / len(e)
    pa = np.histogram(a, qs)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift(train: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (label, _, _) in FEATURES.items():
        if k == "is_metro":
            continue
        v = psi(train[k], current[k])
        rows.append({"feature": label, "psi": v, "status": "Stable" if v < 0.1 else ("Watch" if v < 0.25 else "Retrain")})
    return pd.DataFrame(rows)
