"""Models for the subscriber value lab. Every model uses only what the company could observe; the hidden personas are
used for nothing except checking that the segmentation recovered real structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lab.sub import data as D

DISCOUNT = 0.10 / 12          # monthly discount rate
HORIZON = 60                  # months of future value counted in CLV


# ---------------------------------------------------------------------------
# Observable features (what a real company's warehouse would hold)
# ---------------------------------------------------------------------------
def features(s: pd.DataFrame, tenure: np.ndarray) -> pd.DataFrame:
    x = pd.DataFrame(index=s.index)
    x["tenure"] = tenure
    x["log_tenure"] = np.log1p(np.maximum(tenure, 0))
    to_end = np.where(s["contract"] > 0, s["contract"] - tenure, 99)
    x["months_to_contract_end"] = np.clip(to_end, -6, 24)
    x["contract_end_window"] = ((to_end >= -1) & (to_end <= 2) & (s["contract"] > 0)).astype(float)
    x["promo_end_window"] = (s["promo"] & (tenure >= 10) & (tenure <= 14)).astype(float)
    x["no_contract"] = (s["contract"] == 0).astype(float)
    x["promo"] = s["promo"].astype(float)
    x["fee"] = s["fee"]
    x["mobile_lines"] = s["mobile_lines"]
    x["log_usage"] = np.log(s["usage_gb"])
    x["gig_or_better"] = (s["tier"] != "300 Mbps").astype(float)
    x["autopay"] = s["autopay"].astype(float)
    x["paperless"] = s["paperless"].astype(float)
    x["care_calls_90d"] = s["care_calls_90d"]
    x["repeat_contacts"] = s["repeat_contacts"]
    x["outage_hrs_90d"] = s["outage_hrs_90d"]
    x["late_payments_12m"] = s["late_payments_12m"]
    x["age"] = s["age"]
    x["b2b"] = s["b2b"].astype(float)
    for drv in D.DRIVERS:
        x[f"driver:{drv}"] = (s["last_driver"] == drv).astype(float)
    return x


NICE = {
    "tenure": "Tenure", "log_tenure": "Tenure (log)", "months_to_contract_end": "Months to contract end", "contract_end_window": "At contract end",
    "promo_end_window": "At promotion end", "no_contract": "No contract", "promo": "Joined on a promotion", "fee": "Monthly fee",
    "mobile_lines": "Mobile lines", "log_usage": "Data usage", "gig_or_better": "Gig plan or faster", "autopay": "Autopay",
    "paperless": "Paperless billing", "care_calls_90d": "Care calls (90 days)", "repeat_contacts": "Repeat contacts",
    "outage_hrs_90d": "Outage hours (90 days)", "late_payments_12m": "Late payments (12 months)", "age": "Age", "b2b": "Business account",
    **{f"driver:{d_}": f"Last call: {d_.lower()}" for d_ in D.DRIVERS},
}


def active_at(s: pd.DataFrame, m: int) -> np.ndarray:
    """Subscribed at the start of calendar month m."""
    return ((s["start_month"] < m) & ((s["churn_month"] < 0) | (s["churn_month"] >= m))).to_numpy()


def churned_in(s: pd.DataFrame, a: int, b: int) -> np.ndarray:
    return ((s["churn_month"] >= a) & (s["churn_month"] < b)).to_numpy()


# ---------------------------------------------------------------------------
# 1 · Trends
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cohort_curves() -> pd.DataFrame:
    s = D.company().subs
    s = s[(s["start_month"] >= 0) & (s["start_month"] < 24)].copy()
    s["cohort"] = "Q" + (s["start_month"] // 3 + 1).astype(str)
    rows = []
    for c, g in s.groupby("cohort"):
        life = np.where(g["churned"], g["churn_month"] - g["start_month"], 10_000)
        horizon = D.TODAY - g["start_month"].max() - 1
        for k in range(0, min(horizon, 24) + 1):
            rows.append({"cohort": c, "month": k, "retained": (life > k).mean(), "n": len(g)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def hazard_by_tenure() -> pd.DataFrame:
    """Observed monthly churn rate by month of tenure, split by contract length."""
    s = D.company().subs
    rows = []
    for c, label in ((0, "No contract"), (12, "12-month contract"), (24, "24-month contract")):
        g = s[s["contract"] == c]
        life = np.where(g["churned"], g["churn_month"] - g["start_month"], D.TODAY - g["start_month"])
        ev = g["churned"].to_numpy()
        for k in range(1, 37):
            at_risk = (life >= k).sum()
            rows.append({"contract": label, "tenure": k, "hazard": ((life == k) & ev).sum() / max(at_risk, 1), "at_risk": at_risk})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2 · Segmentation (K-means on observables, named after profiling)
# ---------------------------------------------------------------------------
SEG_FEATURES = ["fee", "mobile_lines", "log_usage", "age", "b2b", "no_contract", "promo", "care_calls_90d", "late_payments_12m", "autopay", "log_tenure"]


@dataclass
class Segments:
    labels: pd.Series        # sub_id -> segment name
    ari: float
    profile: pd.DataFrame


@st.cache_data(show_spinner=False)
def segments() -> Segments:
    s = D.company().subs
    x = features(s, s["tenure_today"].to_numpy())[SEG_FEATURES]
    X = StandardScaler().fit_transform(x)
    km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(X)
    # Name clusters by profile: match each cluster to the persona whose average profile it resembles most.
    zp = pd.DataFrame(X, columns=SEG_FEATURES).groupby(s["persona"].to_numpy()).mean()
    zc = pd.DataFrame(X, columns=SEG_FEATURES).groupby(km.labels_).mean()
    cost = ((zc.to_numpy()[:, None, :] - zp.to_numpy()[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    names = {int(i): zp.index[j] for i, j in zip(r, c)}
    lab = pd.Series([names[i] for i in km.labels_], index=s["sub_id"], name="segment")
    prof = s.assign(segment=lab.to_numpy()).groupby("segment").agg(
        subscribers=("sub_id", "size"), fee=("fee", "mean"), lines=("mobile_lines", "mean"), usage=("usage_gb", "median"),
        age=("age", "median"), no_contract=("contract", lambda v: (v == 0).mean()), care=("care_calls_90d", "mean"),
        churned=("churned", "mean"), b2b=("b2b", "mean"))
    return Segments(lab, float(adjusted_rand_score(s["persona"], km.labels_)), prof)


SEGMENT_ORDER = list(D.PERSONAS)
SEG_COLORS = dict(zip(SEGMENT_ORDER, ["#CBFA7C", "#EF936F", "#8FA2F0", "#E07AA0", "#D9C24A", "#5CC3EE"]))


# ---------------------------------------------------------------------------
# 3 · Customer lifetime value (discrete-time survival × margin)
# ---------------------------------------------------------------------------
HZ_FEATURES = None  # set on first fit


@st.cache_data(show_spinner=False)
def hazard_model():
    """Monthly churn hazard fitted on subscriber-months 0–29, with every churn month kept and quiet months sampled."""
    s = D.company().subs
    rng = np.random.default_rng(0)
    frames, ys, ws = [], [], []
    for m in range(0, 30):
        act = active_at(s, m)
        idx = np.flatnonzero(act)
        y = churned_in(s, m, m + 1)[idx]
        keep = y | (rng.random(len(idx)) < 0.08)
        idx, y = idx[keep], y[keep]
        sub = s.iloc[idx]
        frames.append(features(sub, (m - sub["start_month"]).to_numpy()))
        ys.append(y)
        ws.append(np.where(y, 1.0, 1 / 0.08))
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X), y, sample_weight=w)
    return lr, sc, list(X.columns)


def _hazard(s: pd.DataFrame, tenure: np.ndarray, shift: np.ndarray | float = 0.0) -> np.ndarray:
    lr, sc, cols = hazard_model()
    z = lr.decision_function(sc.transform(features(s, tenure)[cols])) + shift
    return 1 / (1 + np.exp(-z))


def clv_for(s: pd.DataFrame, fee: np.ndarray | None = None, shift: np.ndarray | float = 0.0, horizon: int = HORIZON) -> pd.DataFrame:
    """Expected discounted margin over the horizon, and expected remaining months, from today's tenure."""
    fee = s["fee"].to_numpy() if fee is None else fee
    margin = fee * D.MARGIN - s["care_calls_90d"].to_numpy() / 3 * D.CARE_COST
    surv = np.ones(len(s))
    clv = np.zeros(len(s))
    months = np.zeros(len(s))
    for j in range(1, horizon + 1):
        h = _hazard(s, s["tenure_today"].to_numpy() + j, shift)
        surv = surv * (1 - h)
        clv += surv * margin / (1 + DISCOUNT) ** j
        months += surv
    return pd.DataFrame({"clv": clv, "exp_months": months, "margin": margin, "surv_12m": np.nan}, index=s.index)


@st.cache_data(show_spinner=False)
def base_today() -> pd.DataFrame:
    """Every active subscriber today with segment, CLV, churn risk and treatment uplift: the table the stages share."""
    c = D.company()
    s = c.subs[~c.subs["churned"]].copy()
    s["segment"] = s["sub_id"].map(segments().labels).to_numpy()
    v = clv_for(s)
    s["clv"], s["exp_months"], s["margin"] = v["clv"], v["exp_months"], v["margin"]
    surv12 = np.ones(len(s))
    for j in range(1, 13):
        surv12 *= 1 - _hazard(s, s["tenure_today"].to_numpy() + j)
    s["surv_12m"] = surv12
    ch = churn()
    s["churn_6m"] = ch.score_today(s)
    up = uplift()
    s["uplift"] = up.score(s)
    return s


# ---------------------------------------------------------------------------
# 4 · Churn propensity (bench + champion)
# ---------------------------------------------------------------------------
def churn_sets():
    s = D.company().subs
    tr = active_at(s, 24)
    te = active_at(s, 30) & ~s["treated"].to_numpy()
    Xtr = features(s[tr], (24 - s.loc[tr, "start_month"]).to_numpy())
    Xte = features(s[te], (30 - s.loc[te, "start_month"]).to_numpy())
    ytr = churned_in(s, 24, 30)[tr].astype(int)
    yte = churned_in(s, 30, 36)[te].astype(int)
    return Xtr, ytr, Xte, yte


SAVE_RATE, SAVE_VALUE, CONTACT_COST, CONTACT_SHARE = 0.30, 1200.0, 25.0, 0.20


@st.cache_data(show_spinner=False)
def churn_bench():
    from lab import bench as B

    Xtr, ytr, Xte, yte = churn_sets()

    def value(y, p):
        top = p >= np.quantile(p, 1 - CONTACT_SHARE)
        return float((y[top] * SAVE_RATE * SAVE_VALUE).sum() - CONTACT_COST * top.sum())

    return B.run(Xtr, ytr, Xte, yte, value)


@dataclass
class Churn:
    model: object
    cols: list
    auc: float
    deciles: pd.DataFrame
    drivers: pd.DataFrame

    def score_today(self, s: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features(s, s["tenure_today"].to_numpy())[self.cols])[:, 1]


@st.cache_resource(show_spinner=False)
def churn() -> Churn:
    from sklearn.inspection import permutation_importance

    from lab import bench as B

    Xtr, ytr, Xte, yte = churn_sets()
    model = B._models(0)["XGBoost"].fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    dec = pd.DataFrame({"p": p, "y": yte})
    dec["decile"] = 10 - pd.qcut(dec["p"].rank(method="first"), 10, labels=False)
    deciles = dec.groupby("decile").agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    deciles["capture"] = dec.groupby("decile")["y"].sum().cumsum().to_numpy() / max(yte.sum(), 1)
    rs = np.random.default_rng(0).choice(len(Xte), min(6000, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[rs], yte[rs], scoring="roc_auc", n_repeats=3, random_state=0)
    drivers = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    drivers["label"] = drivers["feature"].map(NICE)
    return Churn(model, list(Xtr.columns), float(roc_auc_score(yte, p)), deciles, drivers)


# ---------------------------------------------------------------------------
# 5 · Retention treatment (uplift from the randomized campaign)
# ---------------------------------------------------------------------------
@dataclass
class Uplift:
    lr: object
    sc: object
    cols: list
    test: pd.DataFrame
    qini_area: dict

    def _design(self, X: pd.DataFrame, t: float) -> np.ndarray:
        Z = self.sc.transform(X[self.cols])
        return np.column_stack([Z, np.full(len(Z), t), Z * t])

    def score_frame(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p1 = self.lr.predict_proba(self._design(X, 1.0))[:, 1]
        p0 = self.lr.predict_proba(self._design(X, 0.0))[:, 1]
        return p0 - p1, p0   # churn avoided by the offer, churn risk without it

    def score(self, s: pd.DataFrame) -> np.ndarray:
        return self.score_frame(features(s, s["tenure_today"].to_numpy()))[0]


def _qini_area(cp: pd.DataFrame, col: str) -> float:
    q = qini(cp, col)
    return float((q["saved"] - q["saved"].iloc[-1] * q["share"]).mean())


@st.cache_resource(show_spinner=False)
def uplift() -> Uplift:
    """Logistic regression with treatment interactions (an S-learner). It beat a boosted two-model learner on held-out
    campaign rows: with a 6% outcome rate, the simpler model's stability matters more than flexibility."""
    c = D.company()
    cp = c.campaign.copy()
    X = features(cp, (D.CAMPAIGN_MONTH - cp["start_month"]).to_numpy())
    t = cp["treated"].to_numpy().astype(float)
    y = cp["churned_6m"].to_numpy().astype(int)
    test = np.random.default_rng(3).random(len(cp)) < 0.4
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    lr = LogisticRegression(max_iter=3000, C=0.3).fit(np.column_stack([Z, t, Z * t[:, None]])[~test], y[~test])
    up = Uplift(lr, sc, list(X.columns), cp, {})
    cp["uplift"], cp["risk"] = up.score_frame(X)
    # Boosted two-model learner, kept only as the comparison that lost.
    params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    tb = t.astype(bool)
    m1 = HistGradientBoostingClassifier(**params).fit(X[~test & tb], y[~test & tb])
    m0 = HistGradientBoostingClassifier(**params).fit(X[~test & ~tb], y[~test & ~tb])
    cp["uplift_boosted"] = m0.predict_proba(X)[:, 1] - m1.predict_proba(X)[:, 1]
    cp["segment"] = cp["sub_id"].map(segments().labels).to_numpy()
    cp["test"] = test
    cp["clv"] = clv_for(cp.assign(tenure_today=D.CAMPAIGN_MONTH - cp["start_month"]), horizon=36)["clv"].to_numpy()
    cp["value_uplift"] = cp["uplift"] * cp["clv"] - D.OFFER_COST
    up.test = cp
    up.qini_area = {k: _qini_area(cp, col) for k, col in
                    (("Uplift (logistic, interactions)", "uplift"), ("Uplift (boosted, two models)", "uplift_boosted"), ("Churn risk", "risk"))}
    return up


def qini(cp: pd.DataFrame, by: str, value: bool = False) -> pd.DataFrame:
    """Cumulative churners saved (or value saved) when contacting in order of `by`, on the held-out campaign rows."""
    te = cp[cp["test"]].sort_values(by, ascending=False)
    t = te["treated"].to_numpy()
    y = te["churned_6m"].to_numpy().astype(float)
    w = te["clv"].to_numpy() if value else np.ones(len(te))
    nt, nc = np.cumsum(t), np.cumsum(~t)
    yt, yc = np.cumsum(y * w * t), np.cumsum(y * w * ~t)
    saved = yc * np.divide(nt, np.maximum(nc, 1)) - yt
    share = np.arange(1, len(te) + 1) / len(te)
    step = max(1, len(te) // 120)
    scale = len(cp) / len(te)  # held-out rows stand in for the whole campaign
    return pd.DataFrame({"share": share[::step], "saved": saved[::step] * scale})


# ---------------------------------------------------------------------------
# 6 · Pricing: response to price increases, estimated from the randomized test
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def price_response():
    pt = D.company().price_test.copy()
    pt["segment"] = pt["sub_id"].map(segments().labels).to_numpy()
    X = _price_design(pt, pt["increase"].to_numpy(), (D.PRICE_TEST_MONTH - pt["start_month"]).to_numpy())
    lr = LogisticRegression(max_iter=3000, C=2.0).fit(X, pt["churned_6m"].astype(int))
    return lr, list(X.columns), pt


def _price_design(s: pd.DataFrame, inc: np.ndarray, tenure: np.ndarray) -> pd.DataFrame:
    f = features(s, tenure)
    X = pd.DataFrame(index=s.index)
    for c in ["contract_end_window", "promo_end_window", "no_contract", "care_calls_90d", "late_payments_12m", "autopay", "log_tenure"]:
        X[c] = f[c]
    for seg in SEGMENT_ORDER:
        is_seg = (s["segment"] == seg).astype(float)
        X[f"seg:{seg}"] = is_seg
        X[f"inc:{seg}"] = inc * is_seg
    X["inc:no_contract"] = inc * f["no_contract"]
    X["inc_sq"] = inc ** 2
    return X


def churn_6m_with_increase(s: pd.DataFrame, inc: np.ndarray) -> np.ndarray:
    lr, cols, _ = price_response()
    return lr.predict_proba(_price_design(s, inc, s["tenure_today"].to_numpy())[cols])[:, 1]


def evaluate_increase(base: pd.DataFrame, inc_by_seg: dict, cap: float, exempt_risk: float, exempt_window: bool,
                      protect_value: bool) -> pd.DataFrame:
    """Per subscriber: the increase applied after guardrails, the extra churn it causes and the net 12-month value."""
    s = base.copy()
    inc = s["segment"].map(inc_by_seg).fillna(0).to_numpy(float)
    inc = np.minimum(inc, cap)
    exempt = np.zeros(len(s), bool)
    if exempt_risk < 1:
        exempt |= s["churn_6m"].to_numpy() >= np.quantile(s["churn_6m"], exempt_risk)
    if exempt_window:
        f = features(s, s["tenure_today"].to_numpy())
        exempt |= (f["contract_end_window"] + f["promo_end_window"]).to_numpy() > 0
    if protect_value:
        exempt |= (s["clv"] >= s["clv"].quantile(0.8)).to_numpy() & (s["churn_6m"] >= s["churn_6m"].median()).to_numpy()
    inc = np.where(exempt, 0.0, inc)
    p0 = churn_6m_with_increase(s, np.zeros(len(s)))
    p1 = churn_6m_with_increase(s, inc)
    extra = np.clip(p1 - p0, 0, None)
    s["increase"] = inc
    s["exempt"] = exempt
    s["extra_churn"] = extra
    s["revenue_12m"] = inc * 12 * s["surv_12m"] * (1 - extra)
    s["value_lost"] = extra * s["clv"]
    s["net"] = s["revenue_12m"] - s["value_lost"]  # a price increase is all margin
    return s


# ---------------------------------------------------------------------------
# 7 · Test and learn
# ---------------------------------------------------------------------------
def sample_size(p0: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8, variance_cut: float = 0.0) -> int:
    """Subscribers per arm to detect a relative change in a rate, two-sided."""
    p1 = p0 * (1 - mde_rel)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    var = (p0 * (1 - p0) + p1 * (1 - p1)) * (1 - variance_cut)
    return int(np.ceil((za + zb) ** 2 * var / (p0 - p1) ** 2))


@st.cache_data(show_spinner=False)
def cuped_rho() -> float:
    """Correlation between the pre-period churn score and the outcome: the share of variance CUPED-style adjustment removes."""
    Xtr, ytr, Xte, yte = churn_sets()
    p = churn().model.predict_proba(Xte)[:, 1]
    return float(np.corrcoef(p, yte)[0, 1])


@st.cache_data(show_spinner=False)
def peeking(looks: int = 8, n_per_look: int = 1500, p0: float = 0.06, sims: int = 2000, seed: int = 4) -> dict:
    """A/A tests (no real difference) checked after every batch. How often does someone declare a winner anyway?"""
    rng = np.random.default_rng(seed)
    a = rng.binomial(n_per_look, p0, (sims, looks)).cumsum(1)
    b = rng.binomial(n_per_look, p0, (sims, looks)).cumsum(1)
    n = n_per_look * np.arange(1, looks + 1)
    pa, pb = a / n, b / n
    pool = (a + b) / (2 * n)
    z = (pa - pb) / np.sqrt(pool * (1 - pool) * 2 / n)
    sig = np.abs(z) > 1.96
    # Pocock-style constant boundary for 8 looks keeps the overall error at 5%.
    pocock = {1: 1.96, 2: 2.18, 3: 2.29, 4: 2.36, 5: 2.41, 6: 2.45, 8: 2.51, 10: 2.56}.get(looks, 2.5)
    return {"final_only": float(sig[:, -1].mean()), "any_look": float(sig.any(1).mean()),
            "corrected": float((np.abs(z) > pocock).any(1).mean()),
            "by_look": [float(sig[:, :k + 1].any(1).mean()) for k in range(looks)]}


@st.cache_data(show_spinner=False)
def save_flow_test(seed: int = 9) -> pd.DataFrame:
    """A completed test: the old save-desk script vs a new flow that leads with a plan review. Funnel counts by arm."""
    rng = np.random.default_rng(seed)
    n = 4800
    stages = ["Cancel intent", "Reached an agent", "Heard an offer", "Accepted an offer", "Still active at 90 days"]
    rates = {"Current script": [1, 0.78, 0.66, 0.31, 0.27], "New plan-review flow": [1, 0.80, 0.74, 0.37, 0.315]}
    rows = []
    for arm, r in rates.items():
        cnt = n
        for i, st_ in enumerate(stages):
            if i:
                cnt = rng.binomial(cnt, r[i] / r[i - 1])
            rows.append({"arm": arm, "stage": st_, "count": int(cnt)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8 · Care and collections
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def care_classifier():
    notes = D.company().notes
    rng = np.random.default_rng(0)
    test = rng.random(len(notes)) < 0.3
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(notes.loc[~test, "note"])
    clf = LogisticRegression(max_iter=2000, C=0.6).fit(Xtr, notes.loc[~test, "driver"])
    pred = clf.predict(vec.transform(notes.loc[test, "note"]))
    truth = notes.loc[test, "driver"].to_numpy()
    acc = float((pred == truth).mean())
    f1 = pd.Series(f1_score(truth, pred, labels=D.DRIVERS, average=None), index=D.DRIVERS)
    conf = pd.crosstab(pd.Series(truth, name="Actual"), pd.Series(pred, name="Predicted")).reindex(index=D.DRIVERS, columns=D.DRIVERS, fill_value=0)
    return vec, clf, acc, f1, conf


@st.cache_data(show_spinner=False)
def reminder_effects() -> pd.DataFrame:
    r = D.company().reminder.copy()
    r["band"] = pd.cut(r["late_payments_12m"], [-1, 1, 2, 3, 99], labels=["0–1 late", "2 late", "3 late", "4+ late"])
    g = r.groupby(["band", "reminded"], observed=True)["disconnected_60d"].agg(["mean", "size"]).unstack()
    out = pd.DataFrame({"control": g[("mean", False)], "reminded": g[("mean", True)], "n": g[("size", False)] + g[("size", True)]})
    out["effect"] = out["control"] - out["reminded"]
    se = np.sqrt(g[("mean", False)] * (1 - g[("mean", False)]) / g[("size", False)] + g[("mean", True)] * (1 - g[("mean", True)]) / g[("size", True)])
    out["lo"], out["hi"] = out["effect"] - 1.96 * se, out["effect"] + 1.96 * se
    return out.reset_index()


# ---------------------------------------------------------------------------
# 9 · Governance helpers
# ---------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


# ---------------------------------------------------------------------------
# Precomputed grids the page uses so sliders respond instantly
# ---------------------------------------------------------------------------
MAX_INC = 10


@st.cache_data(show_spinner=False)
def price_grid() -> np.ndarray:
    """6-month churn probability for every active subscriber at each whole-dollar increase 0–10."""
    b = base_today()
    lr, cols, _ = price_response()
    X = _price_design(b, np.zeros(len(b)), b["tenure_today"].to_numpy())[cols]
    seg_cols = [c for c in cols if c.startswith("inc:") and c != "inc:no_contract"]
    ind = {c: (b["segment"] == c[4:]).to_numpy(float) for c in seg_cols}
    out = []
    for i in range(MAX_INC + 1):
        for c in seg_cols:
            X[c] = i * ind[c]
        X["inc:no_contract"] = i * X["no_contract"]
        X["inc_sq"] = float(i * i)
        out.append(lr.predict_proba(X[cols])[:, 1])
    return np.column_stack(out)


@st.cache_data(show_spinner=False)
def window_flags() -> np.ndarray:
    b = base_today()
    f = features(b, b["tenure_today"].to_numpy())
    return ((f["contract_end_window"] + f["promo_end_window"]).to_numpy() > 0) | ((b["contract"] > 0) & (b["contract"] - b["tenure_today"]).between(0, 3)).to_numpy()


def plan_value(inc_by_seg: dict, cap: int, exempt_top_risk: float | None, exempt_window: bool, protect_value: bool) -> pd.DataFrame:
    b = base_today()
    P = price_grid()
    inc = np.minimum(b["segment"].map(inc_by_seg).fillna(0).to_numpy(int), cap)
    exempt = np.zeros(len(b), bool)
    if exempt_top_risk:
        exempt |= b["churn_6m"].to_numpy() >= np.quantile(b["churn_6m"], 1 - exempt_top_risk)
    if exempt_window:
        exempt |= window_flags()
    if protect_value:
        exempt |= (b["clv"] >= b["clv"].quantile(0.8)).to_numpy() & (b["churn_6m"] >= b["churn_6m"].median()).to_numpy()
    inc = np.where(exempt, 0, inc)
    rows = np.arange(len(b))
    extra = np.clip(P[rows, inc] - P[:, 0], 0, None)
    out = pd.DataFrame({"segment": b["segment"].to_numpy(), "increase": inc, "exempt": exempt, "extra_churn": extra,
                        "revenue_12m": inc * 12 * b["surv_12m"].to_numpy() * (1 - extra), "value_lost": extra * b["clv"].to_numpy()})
    out["net"] = out["revenue_12m"] - out["value_lost"]
    return out


@st.cache_data(show_spinner=False)
def best_increase_by_segment() -> pd.DataFrame:
    """Net 12-month value of a flat increase applied to each whole segment, with no guardrails."""
    b = base_today()
    P = price_grid()
    rows = []
    for i in range(MAX_INC + 1):
        extra = np.clip(P[:, i] - P[:, 0], 0, None)
        net = i * 12 * b["surv_12m"].to_numpy() * (1 - extra) - extra * b["clv"].to_numpy()
        for seg, v in pd.Series(net).groupby(b["segment"].to_numpy()).sum().items():
            rows.append({"segment": seg, "increase": i, "net": v})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def upgrade_value() -> pd.DataFrame:
    """Heavy users on the entry plan: lifetime value with and without a move to 1 Gig (+$15 a month, lower churn).
    The churn effect (−0.25 on the monthly log-odds) is assumed from a past upgrade pilot."""
    b = base_today()
    cand = b[(b["tier"] == "300 Mbps") & (b["usage_gb"] > 900)].copy()
    up = clv_for(cand.assign(tier="1 Gig"), fee=cand["fee"].to_numpy() + 15, shift=-0.25)
    cand["clv_upgraded"] = up["clv"].to_numpy()
    cand["gain"] = cand["clv_upgraded"] - cand["clv"]
    return cand


@st.cache_data(show_spinner=False)
def churn_psi() -> float:
    Xtr, ytr, Xte, yte = churn_sets()
    m = churn().model
    b = base_today()
    return psi(m.predict_proba(Xtr)[:, 1], b["churn_6m"].to_numpy())
