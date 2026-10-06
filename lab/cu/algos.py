"""Algorithm additions to the lending lab: the model bench, a PCA view of the segments, a one-class SVM for fraud and a
random-forest churn challenger. New module (not models.py) so Streamlit Cloud loads it cleanly on redeploy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from lab.cu import data as d
from lab.cu import models as M

UW_APPROVE_SHARE = 0.65


@st.cache_data(show_spinner=False)
def uw_bench():
    """Five algorithms on the underwriting decision: trained on approved loans, tested on next year's applicants."""
    from lab import bench as B

    a = d.bank().applications
    a = a[a["product"].isin(["card", "auto", "personal"])].copy()
    x = M.uw_frame(a)
    train = (a["month"] < 18).to_numpy() & a["approved"].to_numpy()
    test = (a["month"] >= 18).to_numpy()
    y = a["default_12m"].astype(int).to_numpy()
    amt = a.loc[test, "amount"].to_numpy()

    def value(yt, p, nim=0.045, lgd=0.6, opex=120.0):
        ok = p <= np.quantile(p, UW_APPROVE_SHARE)
        return float((amt[ok] * (1 - yt[ok]) * nim).sum() - (amt[ok] * yt[ok] * lgd).sum() - opex * ok.sum())

    return B.run(x[train], y[train], x[test], y[test], value,
                 group_b=a.loc[test, "group_b"].to_numpy(), approve_share=UW_APPROVE_SHARE)


@st.cache_data(show_spinner=False)
def pca_view(groups: tuple = tuple(M.FEATURE_GROUPS), k: int = 6, n: int = 3500):
    """Project the clustering features onto principal components: a map of members, and how much each axis explains."""
    m = d.bank().members
    f = M.member_features(m)
    cols = [c for g in groups for c in M.FEATURE_GROUPS[g]]
    X = StandardScaler().fit_transform(f[cols])
    pca = PCA(random_state=0).fit(X)
    S = M.segment(groups, k)
    rs = np.random.default_rng(2).choice(len(X), min(n, len(X)), replace=False)
    xy = pca.transform(X[rs])[:, :2]
    pts = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "segment": [S.names[i] for i in S.labels[rs]]})
    load = pd.DataFrame(pca.components_[:2].T, index=cols, columns=["pc1", "pc2"])
    evr = pca.explained_variance_ratio_
    return pts, evr, load


def _axis_name(load: pd.Series) -> str:
    top = load.abs().sort_values(ascending=False).index[:2]
    nice = {"log_deposits": "deposits", "log_income": "income", "credit_score": "credit score", "dti": "debt-to-income",
            "utilization": "utilization", "inquiries_6m": "inquiries", "digital_share": "digital use", "tenure_years": "tenure",
            "monthly_txn": "transactions", "products_held": "products", "age": "age", "high_rate_mortgage": "7%+ mortgage",
            "has_mortgage": "mortgage", "has_auto": "auto loan", "has_card": "card", "has_personal": "personal loan", "has_heloc": "HELOC"}
    return " & ".join(nice.get(c, c) for c in top)


@st.cache_data(show_spinner=False)
def fraud_detectors() -> pd.DataFrame:
    """Fraud caught by queue size for each detector: isolation forest, one-class SVM and the rules on their own."""
    F = M.fraud()
    X = pd.DataFrame({
        "log_income_ratio": np.log(F["income_ratio"]), "log_email_age": np.log1p(F["email_age_days"]),
        "device_apps_24h": F["device_apps_24h"], "inquiries_6m": F["inquiries_6m"],
        "thin_file": F["member_id"].isna().astype(float), "link_size": np.log(F["link_size"]),
    })
    Z = StandardScaler().fit_transform(X)
    rs = np.random.default_rng(0).choice(len(Z), 6000, replace=False)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.03).fit(Z[rs])
    scores = {
        "Isolation forest": F["anomaly_pct"].to_numpy(),
        "One-class SVM": pd.Series(-oc.score_samples(Z)).rank(pct=True).to_numpy(),
        "Rules only": (F["rules_hit"] + 1e-3 * F["anomaly_pct"]).to_numpy(),
    }
    fraud = F["is_fraud"].to_numpy()
    ring = F["fraud_ring"].to_numpy()
    total = fraud.sum()
    rows = []
    for name, s in scores.items():
        order = np.argsort(-s)
        for q in range(50, 1001, 50):
            top = order[:q]
            rows.append({"detector": name, "queue": q, "caught": fraud[top].sum() / total, "precision": fraud[top].mean(),
                         "rings": len(set(ring[top][ring[top] >= 0]))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def churn_forest():
    """Random forest as challenger to the gradient-boosted churn model, plus the importance trap.

    A column of pure noise is added on purpose. Impurity importance (the default in most libraries) gives it real
    weight because it has many distinct values to split on; permutation importance on held-out data does not.
    """
    m = d.bank().members.copy()
    X = M.member_features(m)
    X["complaints_12m"] = m["complaints_12m"]
    X["months_since_last_product"] = m["months_since_last_product"]
    X["random_noise"] = np.random.default_rng(7).normal(size=len(X))
    y = m["churned_12m"].astype(int).to_numpy()
    te = np.random.default_rng(0).random(len(m)) < 0.3
    rf = RandomForestClassifier(n_estimators=120, min_samples_leaf=5, max_features="sqrt", n_jobs=2, random_state=0)
    rf.fit(X[~te], y[~te])
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=24, random_state=0).fit(X[~te], y[~te])
    auc = {"Random forest": roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]),
           "Gradient boosting (in use)": roc_auc_score(y[te], gb.predict_proba(X[te])[:, 1])}
    rs = np.random.default_rng(1).choice(np.flatnonzero(te), 3000, replace=False)
    perm = permutation_importance(rf, X.iloc[rs], y[rs], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": X.columns, "impurity": rf.feature_importances_, "permutation": perm.importances_mean})
    imp["impurity"] /= imp["impurity"].sum()
    imp["permutation"] = imp["permutation"].clip(lower=0) / imp["permutation"].clip(lower=0).sum()
    return auc, imp
