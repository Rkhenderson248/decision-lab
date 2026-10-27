"""Loss modeling: loss given default, exposure, lifetime expected credit loss (CECL-style) and macro scenarios.

ECL for each open loan = sum over its remaining months of
    marginal PD(t) × LGD × EAD(t) × discount(t)
where the marginal PD curve comes from observed default timing by month on book, scaled to the loan's own 12-month
PD from the champion model, and shifted by the macro scenario through the unemployment sensitivity estimated from
the book's own history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lab.credit import data as D

LGD_FEATURES = [("deposit_coverage", "Deposit coverage"), ("credit_score", "Credit score"), ("income_verified", "Income verified"),
                ("class_b", "Class B property"), ("class_c", "Class C property")]
SCENARIOS = {"Baseline": 0.0, "Adverse": 2.0, "Severe": 4.5}      # unemployment shock, points
WEIGHTS = {"Baseline": 0.6, "Adverse": 0.3, "Severe": 0.1}
SUPPORTABLE = 12                                               # months


def _lx(df: pd.DataFrame) -> np.ndarray:
    x = pd.DataFrame({"deposit_coverage": df["deposit_coverage"], "credit_score": (df["credit_score"] - 660) / 60,
                      "income_verified": df["income_verified"], "class_b": (df["property_class"] == "B").astype(int),
                      "class_c": (df["property_class"] == "C").astype(int)})
    return x.values


@dataclass
class LGDModel:
    model: LogisticRegression
    mae: float
    mae_base: float
    mean_lgd: float
    train_n: int
    test_n: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_lx(df))[:, 1]


def fit_lgd(loans: pd.DataFrame) -> LGDModel:
    """Fractional logit: a logistic model of the share lost, fitted by weighting each defaulted loan's loss and
    recovery shares (the quasi-likelihood used for LGD because it keeps predictions inside 0 to 1)."""
    dfl = loans[loans["lgd"].notna()].copy()
    tr = dfl[dfl.vintage <= 2022]
    te = dfl[dfl.vintage >= 2023]
    X = np.vstack([_lx(tr), _lx(tr)])
    y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    w = np.r_[tr["lgd"].values, 1 - tr["lgd"].values]
    m = LogisticRegression(C=10.0, max_iter=2000).fit(X, y, sample_weight=w)
    pred = m.predict_proba(_lx(te))[:, 1]
    return LGDModel(m, float(np.mean(np.abs(pred - te["lgd"]))), float(np.mean(np.abs(tr["lgd"].mean() - te["lgd"]))),
                    float(dfl["lgd"].mean()), len(tr), len(te))


def marginal_curve(loans: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """Discrete default hazard by term and month on book from the development vintages: of the loans still open in a
    month, the share that charge off after it. Late defaults that land after the scheduled maturity are booked to the
    final scheduled month, so the curve never spikes on a shrinking denominator."""
    dev = loans[loans.vintage <= 2022]
    p = perf[perf.loan_id.isin(dev.loan_id)].merge(dev[["loan_id", "term"]], on="loan_id")
    out = []
    for term, g in p.groupby("term"):
        at_risk = g[g.mob < term].groupby("mob").size()
        d = dev[(dev.term == term) & (dev.default_month >= 0)]
        ev = np.minimum(d["default_month"] - d["orig_month"] - 1, term - 1).value_counts()
        h = pd.DataFrame({"at_risk": at_risk}).join(ev.rename("defaults")).fillna(0)
        h["hazard"] = (h["defaults"] / h["at_risk"]).clip(0, 1).rolling(3, center=True, min_periods=1).mean()
        h["cum_pd"] = 1 - (1 - h["hazard"]).cumprod()
        out.append(h.reset_index().rename(columns={"index": "mob"}).assign(term=term))
    return pd.concat(out, ignore_index=True)


def macro_beta(loans: pd.DataFrame, perf: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Unemployment sensitivity of the monthly roll from current to 30 days past due, fitted on aggregated monthly
    counts (one logistic on events and non-events by month). Returns the log-odds per point and the series."""
    p = perf.sort_values(["loan_id", "month"])
    nxt = p.groupby("loan_id")["status"].shift(-1)
    cur = p[(p.status == 0) & nxt.notna()].assign(roll=(nxt[(p.status == 0) & nxt.notna()] == 1).astype(int))
    agg = cur.groupby("month").agg(n=("roll", "size"), e=("roll", "sum")).reset_index()
    u = D.unemployment()
    agg["u"] = u[agg["month"]]
    agg["mob_mix"] = cur.groupby("month")["mob"].mean().values
    agg = agg[agg.n > 200]
    X = np.r_[agg[["u", "mob_mix"]].values, agg[["u", "mob_mix"]].values]
    y = np.r_[np.ones(len(agg)), np.zeros(len(agg))]
    w = np.r_[agg["e"].values, (agg["n"] - agg["e"]).values]
    m = LogisticRegression(C=1e6, max_iter=3000).fit(X, y, sample_weight=w)
    agg["rate"] = agg["e"] / agg["n"]
    return float(m.coef_[0][0]), agg


def ecl(book: pd.DataFrame, pd12: np.ndarray, lgd: np.ndarray, curve: pd.DataFrame, beta_u: float,
        shock: float = 0.0, lgd_mult: float = 1.0, pd_mult: float = 1.0) -> pd.DataFrame:
    """Lifetime expected credit loss for each open loan at the cutoff (vectorised over remaining months)."""
    act = (book["end_state"] == "Active").to_numpy()
    open_ = book[act].copy()
    pdv = np.asarray(pd12)[act]
    lgv = np.clip(np.asarray(lgd)[act] * lgd_mult, 0, 1)
    amt, apr, term = open_["amount"].to_numpy(), open_["apr"].to_numpy(), open_["term"].to_numpy()
    mob_now = D.CUTOFF - open_["orig_month"].to_numpy()
    T = int(term.max())
    t = mob_now[:, None] + np.arange(T)[None, :]                     # future months on book
    live = t < term[:, None]
    haz = np.zeros(t.shape)
    anchor = np.ones(len(open_))
    for term_k, c in curve.groupby("term"):
        m = term == term_k
        hk = c.set_index("mob")["hazard"].reindex(np.arange(int(term_k))).ffill().fillna(0).to_numpy()
        haz[m] = hk[np.clip(t[m], 0, int(term_k) - 1)]
        anchor[m] = 1 - np.prod(1 - hk[:12])
    # Reasonable and supportable period: the scenario holds for 12 months, then reverts straight-line to the
    # long-run average over the next 12 (the CECL reversion convention), so far-out months are not stressed.
    rev = np.clip((2 * SUPPORTABLE - np.arange(T)) / SUPPORTABLE, 0, 1)[None, :]
    k = (pdv / np.maximum(anchor, 1e-6) * pd_mult)[:, None] * np.exp(beta_u * shock * rev)
    odds = haz / (1 - haz) * k
    hz = np.where(live, np.clip(odds / (1 + odds), 0, 0.95), 0)
    surv = np.cumprod(np.c_[np.ones(len(hz)), 1 - hz[:, :-1]], axis=1)
    bal = D.annuity_balance(amt[:, None], apr[:, None], term[:, None], t)
    disc = (1 + apr[:, None] / 12) ** -(np.arange(T)[None, :] + 1)
    open_["ead_now"] = D.annuity_balance(amt, apr, term, mob_now)
    open_["pd12"] = pdv
    open_["lgd_hat"] = lgv
    open_["lifetime_pd"] = (surv * hz).sum(1)
    open_["ecl"] = (surv * hz * bal * disc).sum(1) * lgv
    return open_


def reserve(book, pd12, lgd, curve, beta_u, weights=None, **kw) -> dict:
    weights = weights or WEIGHTS
    res = {}
    for s, shock in SCENARIOS.items():
        e = ecl(book, pd12, lgd, curve, beta_u, shock=shock, **kw)
        res[s] = float(e["ecl"].sum())
    res["Weighted"] = sum(res[s] * weights[s] for s in SCENARIOS)
    return res
