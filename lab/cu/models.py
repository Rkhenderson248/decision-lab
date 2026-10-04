"""Models for the lending lab. Each function is cached and depends only on the bank."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from lab.cu import data as d

# ---------------------------------------------------------------------------
# 1 · Segmentation
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "Behavior": ["log_deposits", "monthly_txn", "digital_share", "tenure_years"],
    "Credit": ["credit_score", "dti", "utilization", "inquiries_6m"],
    "Products": ["has_card", "has_auto", "has_personal", "has_mortgage", "has_heloc", "products_held", "high_rate_mortgage"],
    "Life stage": ["age", "log_income"],
}

# Business names, each with a profile it should match (z-scores of centroid features).
SEGMENT_NAMES = {
    "Digital starters": {"age": -1.2, "digital_share": 1.0, "tenure_years": -0.7, "log_deposits": -0.6},
    "Growing families": {"products_held": 0.9, "has_auto": 0.8, "has_mortgage": 0.5, "monthly_txn": 0.6},
    "Rate-sensitive refinancers": {"high_rate_mortgage": 2.0, "inquiries_6m": 0.6, "has_mortgage": 1.2},
    "Affluent savers": {"log_deposits": 1.3, "log_income": 1.2, "credit_score": 0.8, "dti": -0.8},
    "Credit builders": {"credit_score": -1.4, "utilization": 1.3, "inquiries_6m": 0.9, "has_personal": 0.9},
    "Retired loyalists": {"age": 1.6, "tenure_years": 1.4, "digital_share": -1.3},
}


def member_features(m: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=m.index)
    f["log_deposits"] = np.log1p(m["deposit_balance"])
    f["log_income"] = np.log(m["income"])
    f["high_rate_mortgage"] = (m["mortgage_rate"].fillna(0) >= 6.5).astype(float)
    for c in ["monthly_txn", "digital_share", "tenure_years", "credit_score", "dti", "utilization", "inquiries_6m",
              "products_held", "age"]:
        f[c] = m[c].astype(float)
    for c in ["has_card", "has_auto", "has_personal", "has_mortgage", "has_heloc"]:
        f[c] = m[c].astype(float)
    return f


@dataclass
class Segmentation:
    labels: np.ndarray
    names: list[str]
    profiles: pd.DataFrame       # one row per segment, business-readable
    centroids_z: pd.DataFrame
    ari: float                   # agreement with the planted archetypes
    silhouette: float
    features: list[str]


def _name_clusters(cz: pd.DataFrame) -> list[str]:
    names = list(SEGMENT_NAMES)
    cost = np.zeros((len(cz), len(names)))
    for j, n in enumerate(names):
        for feat, target in SEGMENT_NAMES[n].items():
            if feat in cz.columns:
                cost[:, j] -= cz[feat].to_numpy() * np.sign(target) * min(abs(target), 1.5)
    rows, cols = linear_sum_assignment(cost)
    out = [""] * len(cz)
    for r, c in zip(rows, cols):
        out[r] = names[c]
    for i in range(len(out)):
        if not out[i]:  # more clusters than names: describe the strongest trait
            top = cz.iloc[i].abs().sort_values(ascending=False).index[0]
            out[i] = f"Mixed · {'high' if cz.iloc[i][top] > 0 else 'low'} {top.replace('_', ' ')}"
    return out


@st.cache_data(show_spinner=False)
def segment(groups: tuple = tuple(FEATURE_GROUPS), k: int = 6, seed: int = 0) -> Segmentation:
    m = d.bank().members
    f = member_features(m)
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    X = StandardScaler().fit_transform(f[cols])
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
    labels = km.labels_
    zall = pd.DataFrame(StandardScaler().fit_transform(f), columns=f.columns)
    cz = zall.groupby(labels).mean()
    names = _name_clusters(cz)
    prof = m.assign(seg=labels).groupby("seg").agg(
        members=("member_id", "size"), age=("age", "median"), income=("income", "median"),
        score=("credit_score", "median"), deposits=("deposit_balance", "median"), products=("products_held", "mean"),
        digital=("digital_share", "mean"), mortgage=("has_mortgage", "mean"), default_rate=("default_12m", "mean"),
        churn=("churned_12m", "mean"))
    prof.insert(0, "segment", names)
    rs = np.random.default_rng(1).choice(len(X), 6000, replace=False)
    sil = float(silhouette_score(X[rs], labels[rs]))
    ari = float(adjusted_rand_score(m["archetype"], labels))
    return Segmentation(labels, names, prof.reset_index(drop=True), cz, ari, sil, cols)


@st.cache_data(show_spinner=False)
def segments() -> pd.Series:
    """The production segmentation every later module reports by."""
    s = segment()
    return pd.Series([s.names[i] for i in s.labels], index=d.bank().members["member_id"], name="segment")


def rfm(m: pd.DataFrame) -> pd.DataFrame:
    r = pd.qcut(-m["months_since_last_product"].rank(method="first"), 5, labels=False) + 1
    f_ = pd.qcut(m["monthly_txn"].rank(method="first"), 5, labels=False) + 1
    money = m["deposit_balance"] + m[["card_balance", "auto_balance", "personal_balance", "mortgage_balance", "heloc_balance"]].sum(axis=1)
    mo = pd.qcut(money.rank(method="first"), 5, labels=False) + 1
    return pd.DataFrame({"R": r, "F": f_, "M": mo})


# ---------------------------------------------------------------------------
# 3 · Underwriting (consumer loans)
# ---------------------------------------------------------------------------
UW_FEATURES = ["credit_score", "dti", "utilization", "log_income", "inquiries_6m", "tenure_years", "is_member"]
UW_LABELS = {
    "credit_score": "Credit score", "dti": "Debt-to-income ratio", "utilization": "Revolving utilization",
    "log_income": "Income", "inquiries_6m": "Recent credit inquiries", "tenure_years": "Length of membership",
    "is_member": "Existing membership",
}
REASONS = {
    "credit_score": "Credit score below the level required",
    "dti": "Debt obligations too high relative to income",
    "utilization": "Revolving balances too high relative to limits",
    "log_income": "Income insufficient for the amount requested",
    "inquiries_6m": "Too many recent requests for credit",
    "tenure_years": "Length of relationship with the credit union",
    "is_member": "No existing relationship with the credit union",
}
RI_INFLATION = 1.25  # rejects are assumed riskier than the approved-only model says
PDO, BASE_SCORE, BASE_ODDS = 20.0, 600.0, 50.0  # 20 points double the odds; 600 = 50:1 good:bad


def uw_frame(a: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=a.index)
    for c in ["credit_score", "dti", "utilization", "inquiries_6m", "tenure_years"]:
        x[c] = a[c].astype(float)
    x["log_income"] = np.log(a["income"].clip(lower=5_000))
    x["is_member"] = a["member_id"].notna().astype(float) if "member_id" in a else 1.0
    return x


@dataclass
class Scorecard:
    bins: dict          # feature -> bin edges
    woe: dict           # feature -> array of WOE per bin
    coef: np.ndarray
    intercept: float
    points: dict = field(default_factory=dict)  # feature -> points per bin

    def bin_index(self, feat: str, v: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.bins[feat][1:-1], v, side="right"), 0, len(self.woe[feat]) - 1)

    def woe_matrix(self, x: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.woe[f][self.bin_index(f, x[f].to_numpy())] for f in UW_FEATURES])

    def pd(self, x: pd.DataFrame) -> np.ndarray:
        z = self.intercept + self.woe_matrix(x) @ self.coef
        return 1 / (1 + np.exp(-z))

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return sum(self.points[f][self.bin_index(f, x[f].to_numpy())] for f in UW_FEATURES)

    def reasons(self, row: pd.DataFrame, n: int = 4) -> list[tuple[str, float]]:
        short = []
        for f in UW_FEATURES:
            got = self.points[f][self.bin_index(f, row[f].to_numpy())][0]
            short.append((f, float(self.points[f].max() - got)))
        short.sort(key=lambda t: -t[1])
        return [(REASONS[f], s) for f, s in short[:n] if s >= 3]


def _fit_scorecard(x: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None) -> Scorecard:
    w = np.ones(len(y)) if w is None else w
    bins, woe = {}, {}
    for f in UW_FEATURES:
        v = x[f].to_numpy()
        if f == "is_member":
            edges = np.array([-np.inf, 0.5, np.inf])
        else:
            q = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
            edges = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
        idx = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, len(edges) - 2)
        good = np.bincount(idx, weights=w * (1 - y), minlength=len(edges) - 1) + 0.5
        bad = np.bincount(idx, weights=w * y, minlength=len(edges) - 1) + 0.5
        bins[f], woe[f] = edges, np.log((bad / bad.sum()) / (good / good.sum()))
    sc = Scorecard(bins, woe, np.zeros(len(UW_FEATURES)), 0.0)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.woe_matrix(x), y, sample_weight=w)
    sc.coef, sc.intercept = lr.coef_[0], float(lr.intercept_[0])
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    nf = len(UW_FEATURES)
    for i, f in enumerate(UW_FEATURES):  # points = -(woe*beta + a/n)*factor + offset/n
        sc.points[f] = -(sc.woe[f] * sc.coef[i] + sc.intercept / nf) * factor + offset / nf
    return sc


@dataclass
class Underwriting:
    test: pd.DataFrame           # out-of-time, through-the-door, with scores
    kgb: Scorecard               # known good/bad: approved loans only
    ri: Scorecard                # with reject inference
    gbm: HistGradientBoostingClassifier
    auc: dict
    calib: pd.DataFrame          # predicted vs actual bad rate by score band


@st.cache_data(show_spinner=False)
def underwriting() -> Underwriting:
    a = d.bank().applications
    a = a[a["product"].isin(["card", "auto", "personal"])].copy()
    x = uw_frame(a)
    train = (a["month"] < 18).to_numpy()
    appr = a["approved"].to_numpy()
    y = a["default_12m"].astype(int).to_numpy()  # truth; the lender only sees it where approved

    kt = train & appr
    kgb = _fit_scorecard(x[kt], y[kt])
    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=24, random_state=0)
    gbm.fit(x[kt], y[kt])

    # Fuzzy augmentation: each reject enters twice, as bad and as good, weighted by its inflated KGB risk.
    rej = train & ~appr
    p_rej = np.clip(kgb.pd(x[rej]) * RI_INFLATION, 0, 0.95)
    xa = pd.concat([x[kt], x[rej], x[rej]], ignore_index=True)
    ya = np.concatenate([y[kt], np.ones(rej.sum()), np.zeros(rej.sum())])
    wa = np.concatenate([np.ones(kt.sum()), p_rej, 1 - p_rej])
    ri = _fit_scorecard(xa, ya.astype(int), wa)

    te = a[~train].copy()
    xt = x[~train]
    te["score"] = ri.score(xt).round(0)
    te["pd_kgb"] = kgb.pd(xt)
    te["pd_ri"] = ri.pd(xt)
    te["pd_gbm"] = gbm.predict_proba(xt)[:, 1]
    yt = te["default_12m"].astype(int).to_numpy()
    auc = {"Scorecard (approved only)": roc_auc_score(yt, te["pd_kgb"]),
           "Scorecard + reject inference": roc_auc_score(yt, te["pd_ri"]),
           "Gradient boosting (approved only)": roc_auc_score(yt, te["pd_gbm"]),
           "Legacy policy": roc_auc_score(yt, -(te["credit_score"] - 400 * te["dti"]))}
    band = pd.qcut(te["score"], 8, duplicates="drop")
    calib = te.groupby(band, observed=True).agg(actual=("default_12m", "mean"), kgb=("pd_kgb", "mean"), ri=("pd_ri", "mean"),
                                                 n=("app_id", "size"), score=("score", "median")).reset_index(drop=True)
    return Underwriting(te, kgb, ri, gbm, auc, calib)


def cutoff_table(te: pd.DataFrame, nim: float = 0.045, lgd: float = 0.6, opex: float = 120.0) -> pd.DataFrame:
    """Approval rate, losses and profit across score cut-offs (one-year view per loan)."""
    rows = []
    for c in range(480, 720, 5):
        ap = te["score"] >= c
        n = ap.sum()
        if n == 0:
            continue
        amt = te.loc[ap, "amount"]
        bad = te.loc[ap, "default_12m"]
        loss = (amt * bad).sum() * lgd
        income = (amt * (~bad)).sum() * nim
        rows.append({"cutoff": c, "approval_rate": n / len(te), "bad_rate": bad.mean(), "approved": n,
                     "loss": loss, "income": income, "profit": income - loss - opex * n,
                     "air": (te.loc[te["group_b"], "score"] >= c).mean() / max((te.loc[~te["group_b"], "score"] >= c).mean(), 1e-9)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4 · Fraud
# ---------------------------------------------------------------------------
def _components(keys_a: np.ndarray, keys_b: np.ndarray) -> np.ndarray:
    """Union-find over applications linked by a shared phone or address. Returns component size per application."""
    n = len(keys_a)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in (keys_a, keys_b):
        first = {}
        for i, k in enumerate(keys):
            if k in first:
                ra, rb = find(i), find(first[k])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[k] = i
    roots = np.array([find(i) for i in range(n)])
    _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return counts[inv]


@st.cache_data(show_spinner=False)
def fraud() -> pd.DataFrame:
    a = d.bank().applications.copy()
    # What income should look like for this person, from members' real incomes.
    ref = d.bank().members.groupby(pd.cut(d.bank().members["credit_score"], [0, 600, 660, 720, 780, 900]), observed=True)["income"].median()
    expected = pd.cut(a["credit_score"], [0, 600, 660, 720, 780, 900]).map(ref).astype(float)
    a["income_ratio"] = a["stated_income"] / expected
    a["link_size"] = _components(a["phone_id"].to_numpy(), a["address_id"].to_numpy())
    phone_counts = a.groupby("phone_id")["app_id"].transform("size")
    a["phone_reuse"] = phone_counts
    X = pd.DataFrame({
        "log_income_ratio": np.log(a["income_ratio"]), "log_email_age": np.log1p(a["email_age_days"]),
        "device_apps_24h": a["device_apps_24h"], "inquiries_6m": a["inquiries_6m"],
        "thin_file": a["member_id"].isna().astype(float), "link_size": np.log(a["link_size"]),
    })
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
    a["anomaly"] = -iso.score_samples(X)
    a["anomaly_pct"] = a["anomaly"].rank(pct=True)
    rules = {
        "Shares a phone or address with 3+ applications": a["link_size"] >= 3,
        "Stated income far above what this profile earns": a["income_ratio"] >= 1.6,
        "Email address created in the last 30 days": a["email_age_days"] <= 30,
        "Several applications from one device in 24 hours": a["device_apps_24h"] >= 3,
        "No existing relationship": a["member_id"].isna(),
    }
    hits = pd.DataFrame(rules)
    a["rules_hit"] = hits.sum(axis=1)
    a["reasons"] = hits.apply(lambda r: [k for k, v in r.items() if v and k != "No existing relationship"], axis=1)
    a["fraud_score"] = 0.55 * a["anomaly_pct"] + 0.45 * (a["rules_hit"] / 4).clip(0, 1)
    return a


# ---------------------------------------------------------------------------
# 5 · Pricing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def pricing_model() -> tuple[LogisticRegression, list[str], pd.DataFrame]:
    q = d.bank().quotes.copy()
    q["segment"] = q["member_id"].map(segments())
    q["gap"] = q["rate"] - q["competitor_rate"]
    X = _price_design(q)
    lr = LogisticRegression(max_iter=1000, C=5.0).fit(X, q["accepted"])
    return lr, list(X.columns), q


def _price_design(q: pd.DataFrame) -> pd.DataFrame:
    segs = list(SEGMENT_NAMES)
    X = pd.DataFrame(index=q.index)
    for s in segs:
        X[f"gap:{s}"] = q["gap"] * (q["segment"] == s)
        X[f"seg:{s}"] = (q["segment"] == s).astype(float)
    for t in ["A", "B", "C"]:
        X[f"tier:{t}"] = (q["tier"] == t).astype(float)
    X["gap:pd"] = q["gap"] * q["pd"]
    return X


def take_up(rate: np.ndarray, competitor: float, segment_name: str, tier: str, pd_: float) -> np.ndarray:
    lr, cols, _ = pricing_model()
    q = pd.DataFrame({"rate": rate, "competitor_rate": competitor, "segment": segment_name, "tier": tier, "pd": pd_})
    q["gap"] = q["rate"] - q["competitor_rate"]
    return lr.predict_proba(_price_design(q)[cols])[:, 1]


def segment_elasticities() -> pd.Series:
    lr, cols, _ = pricing_model()
    coef = dict(zip(cols, lr.coef_[0]))
    return pd.Series({s: coef[f"gap:{s}"] for s in SEGMENT_NAMES}).sort_values()


# ---------------------------------------------------------------------------
# 6 · Cross-sell
# ---------------------------------------------------------------------------
XS_FEATURES = ["age", "log_income", "credit_score", "utilization", "digital_share", "tenure_years", "products_held", "log_deposits"]


def _xs_frame(m: pd.DataFrame) -> pd.DataFrame:
    f = member_features(m)
    return f[[c for c in XS_FEATURES]]


@st.cache_data(show_spinner=False)
def uplift() -> pd.DataFrame:
    c = d.bank().campaign.copy()
    c["segment"] = c["member_id"].map(segments())
    X = _xs_frame(c)
    seg_d = pd.get_dummies(c["segment"], dtype=float)
    X = pd.concat([X.reset_index(drop=True), seg_d.reset_index(drop=True)], axis=1)
    rng = np.random.default_rng(0)
    test = rng.random(len(c)) < 0.4
    t = c["treated"].to_numpy()
    y = c["adopted"].to_numpy()
    params = dict(max_iter=150, learning_rate=0.06, max_leaf_nodes=16, random_state=0)
    m1 = HistGradientBoostingClassifier(**params).fit(X[~test & t], y[~test & t])
    m0 = HistGradientBoostingClassifier(**params).fit(X[~test & ~t], y[~test & ~t])
    c["p1"], c["p0"] = m1.predict_proba(X)[:, 1], m0.predict_proba(X)[:, 1]
    c["uplift"] = c["p1"] - c["p0"]
    c["test"] = test
    return c


def qini(c: pd.DataFrame, by: str) -> pd.DataFrame:
    """Cumulative incremental adopters when targeting in order of `by` (test set only)."""
    te = c[c["test"]].sort_values(by, ascending=False)
    t = te["treated"].to_numpy()
    y = te["adopted"].to_numpy()
    nt, nc = np.cumsum(t), np.cumsum(~t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * ~t)
    inc = yt - yc * np.divide(nt, np.maximum(nc, 1))
    share = np.arange(1, len(te) + 1) / len(te)
    step = max(1, len(te) // 100)
    return pd.DataFrame({"share": share[::step], "incremental": inc[::step]})


PRODUCT_VALUE = {"card": 210, "auto": 640, "personal": 380, "heloc": 900, "savings": 90, "mortgage": 2200}
PRODUCT_LABEL = {"card": "Rewards credit card", "auto": "Auto loan refinance", "personal": "Personal loan",
                 "heloc": "Home equity line", "savings": "High-yield savings", "mortgage": "Mortgage"}


@st.cache_data(show_spinner=False)
def next_best_product() -> tuple[dict, pd.DataFrame]:
    """One propensity model per product, trained on who already holds it."""
    m = d.bank().members
    X = _xs_frame(m).drop(columns=["products_held"])
    models, probs = {}, pd.DataFrame(index=m["member_id"])
    for p in ["card", "auto", "personal", "heloc", "savings"]:
        y = m[f"has_{p}"].to_numpy()
        lr = LogisticRegression(max_iter=600).fit(StandardScaler().fit(X).transform(X), y)
        sc = StandardScaler().fit(X)
        models[p] = (lr, sc)
        pr = lr.predict_proba(sc.transform(X))[:, 1]
        probs[p] = np.where(y, np.nan, pr)  # only offer what they do not hold
    return models, probs


# ---------------------------------------------------------------------------
# 7 · Retention
# ---------------------------------------------------------------------------
def _hazard_design(h: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=h.index)
    X["incentive"] = h["incentive"].clip(-2, 3)
    X["incentive_hi"] = (h["incentive"] - 0.75).clip(0, None)
    X["age"] = h["age"].clip(0, 24)
    for s in SEGMENT_NAMES:
        X[f"seg:{s}"] = (h["segment"] == s).astype(float)
        X[f"inc_hi:{s}"] = X["incentive_hi"] * (h["segment"] == s)
    return X


@st.cache_data(show_spinner=False)
def prepay_model() -> tuple[LogisticRegression, list[str], pd.DataFrame]:
    b = d.bank()
    seg = segments()
    mort = b.mortgages.copy()
    mort["segment"] = mort["member_id"].map(seg)
    h = b.mortgage_history.copy()
    h["segment"] = mort["segment"].to_numpy()[h["loan"]]
    # Keep every prepayment and a sample of the months without one, then re-weight.
    rng = np.random.default_rng(0)
    keep = h["prepaid"] | (rng.random(len(h)) < 0.35)
    hs = h[keep]
    w = np.where(hs["prepaid"], 1.0, 1 / 0.35)
    X = _hazard_design(hs)
    lr = LogisticRegression(max_iter=1000, C=2.0).fit(X, hs["prepaid"], sample_weight=w)
    return lr, list(X.columns), mort


def project_prepay(market_rate: float, months: int = 12) -> pd.DataFrame:
    lr, cols, mort = prepay_model()
    live = mort[~mort["prepaid"]].copy()
    h = pd.DataFrame({"incentive": live["mortgage_rate"] - market_rate, "age": live["months_observed"] + 6, "segment": live["segment"]})
    p_month = lr.predict_proba(_hazard_design(h)[cols])[:, 1]
    live["p_12m"] = 1 - (1 - p_month) ** months
    live["incentive"] = h["incentive"].to_numpy()
    return live


def kaplan_meier(mort: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    out = []
    for g, sub in mort.groupby(groups):
        t = sub["months_observed"].to_numpy()
        e = sub["prepaid"].to_numpy()
        s, rows = 1.0, [(0, 1.0)]
        for k in range(1, 37):
            at_risk = (t >= k).sum()
            ev = ((t == k) & e).sum()
            if at_risk:
                s *= 1 - ev / at_risk
            rows.append((k, s))
        out.append(pd.DataFrame(rows, columns=["month", "survival"]).assign(group=g))
    return pd.concat(out)


@st.cache_data(show_spinner=False)
def churn_model() -> tuple[pd.DataFrame, float]:
    m = d.bank().members.copy()
    m["segment"] = m["member_id"].map(segments())
    X = member_features(m)
    X["complaints_12m"] = m["complaints_12m"]
    X["months_since_last_product"] = m["months_since_last_product"]
    rng = np.random.default_rng(0)
    te = rng.random(len(m)) < 0.3
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=24, random_state=0)
    gb.fit(X[~te], m.loc[~te, "churned_12m"])
    m["churn_score"] = gb.predict_proba(X)[:, 1]
    auc = roc_auc_score(m.loc[te, "churned_12m"], m.loc[te, "churn_score"])
    return m, float(auc)


# ---------------------------------------------------------------------------
# 8 · Collections
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def roll_rates(segment_name: str | None = None) -> pd.DataFrame:
    dq = d.bank().delinquency
    if segment_name:
        dq = dq[dq["member_id"].map(segments()).eq(segment_name)]
    hist = dq[[f"m{t}" for t in range(24)]].to_numpy()
    counts = np.zeros((5, 5))
    for t in range(23):
        np.add.at(counts, (hist[:, t], hist[:, t + 1]), 1)
    counts[4] = [0, 0, 0, 0, 1]
    P = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(P, index=d.STATES, columns=d.STATES)


@st.cache_data(show_spinner=False)
def treatment_effects() -> pd.DataFrame:
    tr = d.bank().treatments.copy()
    tr["segment"] = tr["member_id"].map(segments())
    g = tr.groupby(["segment", "state", "treatment"])["cured"].agg(["mean", "size"]).reset_index()
    overall = tr.groupby(["state", "treatment"])["cured"].mean().rename("overall").reset_index()
    g = g.merge(overall, on=["state", "treatment"])
    k = 40  # shrink thin cells toward the overall rate
    g["cure"] = (g["mean"] * g["size"] + g["overall"] * k) / (g["size"] + k)
    return g


TREAT_COST = {"Text": (0.6, 0.0), "Call": (9.0, 1.0), "Hardship plan": (45.0, 3.0)}  # $ cost, collector conversations


def allocate(collectors: int, calls_per_day: int = 10) -> tuple[pd.DataFrame, float]:
    """Greedy: give each delinquent account the treatment with the best extra recovery per unit of collector time."""
    dq = d.bank().delinquency
    acct = dq[dq["state"].between(1, 3)].copy()
    acct["segment"] = acct["member_id"].map(segments())
    eff = treatment_effects().set_index(["segment", "state", "treatment"])["cure"]
    base = np.array([eff.get((s, st_, "None"), 0.15) for s, st_ in zip(acct["segment"], acct["state"])])
    options = {}
    for t in TREAT_COST:
        cure = np.array([eff.get((s, st_, t), np.nan) for s, st_ in zip(acct["segment"], acct["state"])])
        options[t] = (cure - base) * acct["balance"].to_numpy() * 0.6 - TREAT_COST[t][0]
    capacity = collectors * calls_per_day * 21
    plan = np.array(["Text"] * len(acct), dtype=object)
    gain = options["Text"].copy()
    cands = []
    for t in ("Call", "Hardship plan"):
        extra = options[t] - options["Text"]
        units = TREAT_COST[t][1]
        for i in np.flatnonzero(extra > 0):
            cands.append((extra[i] / units, i, t, units, extra[i]))
    cands.sort(key=lambda c: -c[0])
    used = 0.0
    for _, i, t, units, extra in cands:
        if plan[i] != "Text" or used + units > capacity:
            continue
        plan[i] = t
        gain[i] += extra
        used += units
    acct["treatment"] = plan
    acct["expected_recovery"] = gain + base * acct["balance"].to_numpy() * 0.6
    return acct, used / capacity if capacity else 0.0


# ---------------------------------------------------------------------------
# 2 · Acquisition
# ---------------------------------------------------------------------------
def lookalike(segment_name: str) -> pd.DataFrame:
    b = d.bank()
    m = b.members
    seg = m["member_id"].map(segments())
    pos = m[seg.eq(segment_name)]
    pros = b.prospects
    cols = ["age", "income", "credit_score", "dti", "utilization", "digital_share", "inquiries_6m"]
    train = pd.concat([pos[cols].assign(y=1), pros[cols].sample(min(len(pros), len(pos) * 3), random_state=0).assign(y=0)])
    train["income"] = np.log(train["income"])
    sc = StandardScaler().fit(train[cols])
    lr = LogisticRegression(max_iter=600).fit(sc.transform(train[cols]), train["y"])
    X = pros[cols].copy()
    X["income"] = np.log(X["income"])
    out = pros[["prospect_id", "metro", "age", "income", "credit_score"]].copy()
    out["similarity"] = lr.predict_proba(sc.transform(X))[:, 1]
    return out


# ---------------------------------------------------------------------------
# 9 · Governance helpers
# ---------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())
