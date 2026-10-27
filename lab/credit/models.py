"""Probability of default: a constrained gradient-boosting champion against a logistic benchmark, validated the way
an external reviewer would test it.

Target: charge-off within 12 months on book. Development sample: 2021–2022 originations. Out-of-time sample: 2023
originations (observed for 12 months by the cutoff). The 2024 book is scored for the reserve.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

# Feature, label, direction of risk (+1 more risk as it rises, -1 less, 0 unconstrained). The directions are the
# monotone constraints: a reviewer can check every one against credit sense.
FEATURES = [
    ("credit_score", "Credit score", -1),
    ("rent_to_income", "Rent to income", 1),
    ("deposit_coverage", "Deposit coverage", -1),
    ("prior_evictions", "Prior evictions", 1),
    ("months_at_address", "Months at prior address", -1),
    ("occupancy", "Property occupancy", -1),
    ("income_verified", "Income verified", -1),
    ("class_c", "Class C property", 1),
    ("unemployment_at_orig", "Unemployment at origination", 1),
    ("amount_k", "Loan amount ($K)", 0),
    ("term", "Term (months)", 0),
]
COLS = [f[0] for f in FEATURES]
LABEL = {f[0]: f[1] for f in FEATURES}
MONO = tuple(f[2] for f in FEATURES)
GRADES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_EDGES = [0, 0.025, 0.05, 0.08, 0.12, 0.18, 0.28, 1.0]
SEED = 7


def frame(loans: pd.DataFrame) -> pd.DataFrame:
    x = loans.copy()
    x["class_c"] = (x["property_class"] == "C").astype(int)
    x["amount_k"] = x["amount"] / 1000.0
    return x


def split(loans: pd.DataFrame):
    x = frame(loans)
    dev = x[(x.vintage <= 2022) & (x.observed_12m == 1)]
    oot = x[(x.vintage == 2023) & (x.observed_12m == 1)]
    book = x[(x.vintage == 2024)]
    return dev, oot, book


def ks(y, p) -> float:
    o = np.argsort(p)
    y = np.asarray(y)[o]
    cb = np.cumsum(y) / max(y.sum(), 1)
    cg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cb - cg)))


def hosmer_lemeshow(y, p, g: int = 10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), g, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    chi = float((((t.obs - t.exp) ** 2) / (t.exp * (1 - t.exp / t.n))).sum())
    return chi, float(1 - stats.chi2.cdf(chi, g - 2)), t.assign(obs_rate=t.obs / t.n, exp_rate=t.exp / t.n).reset_index()


def psi(a, b, bins: int = 10) -> float:
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    pa = np.clip(np.histogram(a, q)[0] / len(a), 1e-4, None)
    pb = np.clip(np.histogram(b, q)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def grade(p) -> np.ndarray:
    return np.array(GRADES)[np.clip(np.searchsorted(GRADE_EDGES, p, side="right") - 1, 0, len(GRADES) - 1)]


@dataclass
class Fit:
    name: str
    model: object
    p_dev: np.ndarray
    p_oot: np.ndarray
    p_book: np.ndarray
    auc_dev: float
    auc_oot: float
    ks_oot: float
    brier_oot: float
    hl_chi: float
    hl_p: float
    calib: pd.DataFrame
    extra: dict = field(default_factory=dict)

    @property
    def gini_oot(self) -> float:
        return 2 * self.auc_oot - 1


def _evaluate(name, model, dev, oot, book, predict) -> Fit:
    pd_, po, pb = predict(dev), predict(oot), predict(book)
    chi, p, cal = hosmer_lemeshow(oot["default_12m"].values, po)
    return Fit(name, model, pd_, po, pb, roc_auc_score(dev["default_12m"], pd_), roc_auc_score(oot["default_12m"], po),
               ks(oot["default_12m"].values, po), brier_score_loss(oot["default_12m"], po), chi, p, cal)


def train(loans: pd.DataFrame):
    dev, oot, book = split(loans)
    y = dev["default_12m"].values
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    lr.fit(dev[COLS], y)
    bench = _evaluate("Logistic benchmark", lr, dev, oot, book, lambda d: lr.predict_proba(d[COLS])[:, 1])
    if HAS_XGB:
        gb = XGBClassifier(n_estimators=320, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                           min_child_weight=25, reg_lambda=3.0, monotone_constraints=MONO, tree_method="hist",
                           n_jobs=2, random_state=SEED, verbosity=0, eval_metric="logloss")
    else:  # pragma: no cover
        gb = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.04, max_depth=3, monotonic_cst=list(MONO), random_state=SEED)
    # Fit on 80% of development, calibrate on the held-back 20% (Platt scaling on the log-odds), so the PDs
    # can be used as probabilities in the loss model, not only as a ranking.
    rng = np.random.default_rng(SEED)
    fit_m = rng.random(len(dev)) < 0.8
    gb.fit(dev[COLS][fit_m], y[fit_m])
    raw = lambda d: np.clip(gb.predict_proba(d[COLS])[:, 1], 1e-5, 1 - 1e-5)  # noqa: E731
    lo = lambda p: np.log(p / (1 - p))  # noqa: E731
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lo(raw(dev[~fit_m])).reshape(-1, 1), y[~fit_m])
    model = Calibrated(gb, cal)
    champ = _evaluate("Gradient boosting (monotone, calibrated)", model, dev, oot, book, lambda d: model.predict_proba(d[COLS])[:, 1])
    uncal = _evaluate("Gradient boosting before calibration", gb, dev, oot, book, raw)
    champ.extra["uncalibrated"] = uncal
    return champ, bench, (dev, oot, book)


class Calibrated:
    """The boosted model followed by a one-variable logistic calibration of its log-odds."""

    def __init__(self, booster, calibrator):
        self.booster, self.calibrator = booster, calibrator

    def predict_proba(self, x):
        p = np.clip(self.booster.predict_proba(x)[:, 1], 1e-5, 1 - 1e-5)
        q = self.calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - q, q])

    def get_booster(self):
        return self.booster.get_booster()


def reason_codes(fit: Fit, rows: pd.DataFrame, top: int = 4) -> list[list[tuple[str, float]]]:
    """Per-loan contributions (TreeSHAP, in log-odds) from XGBoost; the top drivers that raise risk."""
    if not HAS_XGB:
        return [[] for _ in range(len(rows))]
    import xgboost as xgb
    contrib = fit.model.get_booster().predict(xgb.DMatrix(rows[COLS]), pred_contribs=True)[:, :-1]
    out = []
    for c in contrib:
        o = np.argsort(-c)[:top]
        out.append([(LABEL[COLS[i]], float(c[i])) for i in o])
    return out, contrib


def monotone_check(fit: Fit, sample: pd.DataFrame) -> pd.DataFrame:
    """Move each constrained feature across its range for real loans; the average PD must move the stated way."""
    rows = []
    for col, label, d in FEATURES:
        if d == 0:
            continue
        lo, hi = np.quantile(sample[col], [0.05, 0.95])
        if lo == hi:
            lo, hi = sample[col].min(), sample[col].max()
        grid = np.linspace(lo, hi, 9)
        means = []
        for g in grid:
            x = sample[COLS].copy()
            x[col] = g
            means.append(float(fit.model.predict_proba(x)[:, 1].mean()))
        diffs = np.diff(means)
        ok = bool(np.all(diffs * d >= -1e-9))
        rows.append({"Feature": label, "Expected": "rises" if d > 0 else "falls", "PD at low end": means[0],
                     "PD at high end": means[-1], "Holds at every step": ok})
    return pd.DataFrame(rows)


def backtest_grades(y, p) -> pd.DataFrame:
    g = grade(p)
    rows = []
    for k in GRADES:
        m = g == k
        n = int(m.sum())
        if n == 0:
            continue
        obs = int(np.asarray(y)[m].sum())
        exp_rate = float(np.mean(np.asarray(p)[m]))
        test = stats.binomtest(obs, n, exp_rate, alternative="greater")
        z = (obs / n - exp_rate) / np.sqrt(exp_rate * (1 - exp_rate) / n)
        light = "Green" if test.pvalue >= 0.05 else ("Amber" if test.pvalue >= 0.01 else "Red")
        rows.append({"Grade": k, "Loans": n, "Predicted PD": exp_rate, "Observed": obs / n, "Defaults": obs,
                     "z": z, "p (under-prediction)": test.pvalue, "Traffic light": light})
    return pd.DataFrame(rows)


def data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df[COLS + ["default_12m"]], index=False).values.tobytes()).hexdigest()[:12]


def forward_du(loans: pd.DataFrame) -> np.ndarray:
    """Average unemployment over a loan's first 12 months minus unemployment at origination: the part of the
    economy a point-in-time score cannot see when it is made."""
    from lab.credit.data import unemployment
    u = unemployment()
    o = loans["orig_month"].to_numpy()
    fwd = np.array([u[min(m + 1, len(u) - 1):min(m + 13, len(u))].mean() for m in o])
    return fwd - u[o]


@dataclass
class Overlay:
    model: LogisticRegression
    beta_du: float
    slope: float

    def apply(self, p, du) -> np.ndarray:
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return self.model.predict_proba(np.column_stack([np.log(p / (1 - p)), du]))[:, 1]


def macro_overlay(fit: Fit, dev: pd.DataFrame) -> Overlay:
    """Remediation for the out-of-time under-prediction: re-fit the champion's log-odds with the forward change in
    unemployment, on development data only, so the PD becomes a 12-month forward-looking estimate."""
    p = np.clip(fit.p_dev, 1e-5, 1 - 1e-5)
    X = np.column_stack([np.log(p / (1 - p)), forward_du(dev)])
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, dev["default_12m"].values)
    return Overlay(m, float(m.coef_[0][1]), float(m.coef_[0][0]))
