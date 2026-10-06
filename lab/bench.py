"""Model bench: five algorithms on the same decision, judged on what the business needs, not only accuracy.

Shared by the lending lab (underwriting, a regulated decision) and the subscriber lab (churn, an unregulated one).
Kept in its own module so Streamlit Cloud picks it up cleanly on redeploy (see lab/widgets.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:  # the CPU-only wheel (xgboost-cpu) keeps the deploy small
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

ALGOS = ["Linear regression", "Logistic regression", "Random forest", "XGBoost", "Support vector machine"]

# What each algorithm can say about a single decision, and how simple it is to run and audit (lower = simpler).
PROFILE = {
    "Linear regression": dict(reasons="Exact (coefficients)", exact=True, simplicity=1,
                              note="A straight-line probability. Can predict below 0 or above 100%, so it needs clipping."),
    "Logistic regression": dict(reasons="Exact (points per characteristic)", exact=True, simplicity=2,
                                note="The scorecard family: every point is traceable to one characteristic."),
    "Random forest": dict(reasons="Approximate (SHAP)", exact=False, simplicity=4,
                          note="Hundreds of deep trees averaged. Robust, but large and slow to score."),
    "XGBoost": dict(reasons="Approximate (SHAP)", exact=False, simplicity=3,
                    note="Boosted trees: usually the most accurate on tabular data. Reasons need SHAP and review."),
    "Support vector machine": dict(reasons="None natively", exact=False, simplicity=5,
                                   note="An RBF-kernel boundary (Nyström approximation). Probabilities only after calibration."),
}


def _models(seed: int = 0) -> dict:
    xgb = (XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=5, reg_lambda=1.0, tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)
           if HAS_XGB else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=16, random_state=seed))
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
        "Random forest": RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                                                n_jobs=2, random_state=seed),
        "XGBoost": xgb,
        "Support vector machine": CalibratedClassifierCV(
            make_pipeline(StandardScaler(), Nystroem(gamma=0.08, n_components=300, random_state=seed),
                          LinearSVC(C=0.5, dual="auto", max_iter=4000)), method="sigmoid", cv=3),
    }


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted and actual rate across risk deciles."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)
    gap = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            gap += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(gap)


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    return pd.DataFrame({"pred": p, "y": y, "b": q}).groupby("b").agg(pred=("pred", "mean"), actual=("y", "mean")).reset_index(drop=True)


@dataclass
class Bench:
    table: pd.DataFrame            # one row per algorithm
    preds: dict                    # algorithm -> test-set probabilities
    y: np.ndarray
    importance: pd.DataFrame | None = None


def run(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
        value_fn: Callable[[np.ndarray, np.ndarray], float], group_b: np.ndarray | None = None,
        approve_share: float | None = None, seed: int = 0, svm_rows: int = 12000) -> Bench:
    """Fit every algorithm on the same rows and score it on the same out-of-time test set.

    value_fn(y, p) returns the money the decision makes when it is driven by p.
    group_b / approve_share: when given, the adverse impact ratio at that approval share is reported.
    """
    rng = np.random.default_rng(seed)
    rows, preds = [], {}
    for name, model in _models(seed).items():
        Xf, yf = X_train, y_train
        if name == "Support vector machine" and len(X_train) > svm_rows:  # kernel methods do not scale; say so
            keep = rng.choice(len(X_train), svm_rows, replace=False)
            Xf, yf = X_train.iloc[keep], y_train[keep]
        t0 = time.perf_counter()
        model.fit(Xf, yf)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = model.predict(X_test) if name == "Linear regression" else model.predict_proba(X_test)[:, 1]
        score_ms = (time.perf_counter() - t0) / len(X_test) * 1000 * 1000  # ms per 1,000 decisions
        raw_out_of_range = float(((p < 0) | (p > 1)).mean()) if name == "Linear regression" else 0.0
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        preds[name] = p
        top = p >= np.quantile(p, 0.9)
        row = {"algorithm": name, "auc": roc_auc_score(y_test, p), "brier": brier_score_loss(y_test, p),
               "ece": _ece(y_test, p), "top_decile_capture": y_test[top].sum() / max(y_test.sum(), 1),
               "value": float(value_fn(y_test, p)), "fit_s": fit_s, "score_ms": score_ms,
               "out_of_range": raw_out_of_range, "trained_on": len(Xf), **PROFILE[name]}
        if group_b is not None and approve_share is not None:
            ok = p <= np.quantile(p, approve_share)  # approve the lowest-risk share
            row["air"] = ok[group_b].mean() / max(ok[~group_b].mean(), 1e-9)
        rows.append(row)
    table = pd.DataFrame(rows)
    imp = None
    if HAS_XGB:
        xgb = _models(seed)["XGBoost"].fit(X_train, y_train)
        lr = _models(seed)["Logistic regression"].fit(X_train, y_train)
        imp = pd.DataFrame({"feature": X_train.columns,
                            "XGBoost (gain share)": xgb.feature_importances_ / xgb.feature_importances_.sum(),
                            "Logistic (|standardized coef| share)": np.abs(lr[-1].coef_[0]) / np.abs(lr[-1].coef_[0]).sum()})
    return Bench(table, preds, np.asarray(y_test), imp)


def recommend(table: pd.DataFrame, regulated: bool, tolerance: float = 0.01) -> tuple[str, str]:
    """Pick the champion. Regulated decisions need exact reasons; within 1% of the best value, the simpler model wins."""
    pool = table[table["exact"]] if regulated else table
    best_value = pool["value"].max()
    near = pool[pool["value"] >= best_value - abs(best_value) * tolerance]
    pick = near.sort_values("simplicity").iloc[0]
    top = table.loc[table["value"].idxmax()]
    if regulated and pick["algorithm"] != top["algorithm"]:
        gap = top["value"] - pick["value"]
        why = (f"{top['algorithm']} makes the most money on paper, ${gap:,.0f} more than {pick['algorithm']}, but every decline "
               f"needs exact, auditable reasons. {pick['algorithm']} gives them, so it is the champion and "
               f"{top['algorithm']} runs as a challenger.")
    elif pick["algorithm"] != top["algorithm"]:
        why = (f"{top['algorithm']} is best by a hair, within {tolerance:.0%}. {pick['algorithm']} earns almost the same and is "
               "simpler to run, explain and monitor, so it wins the tie.")
    else:
        runner = pool.drop(index=pick.name).sort_values("value", ascending=False)
        gap = pick["value"] - (runner["value"].iloc[0] if len(runner) else 0)
        why = (f"{pick['algorithm']} delivers the most value (${gap:,.0f} ahead of the next model) and this decision does not need "
               "point-by-point reasons, so the extra accuracy is worth the extra complexity.")
    return str(pick["algorithm"]), why
