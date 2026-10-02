"""Synthetic lending pipeline and a transparent propensity model.

Nothing here is real customer data. The generator invents a pipeline whose
conversion depends on a handful of plausible signals, trains a logistic
regression on history, and scores a fresh "today" pipeline. Logistic
regression is deliberate: every score decomposes into per-feature
contributions, which become the reason codes shown beside each lead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CHANNELS = ["Past customer", "Referral", "Web inquiry", "Purchased list"]
CHANNEL_EFFECT = {"Past customer": 0.9, "Referral": 0.6, "Web inquiry": 0.0, "Purchased list": -0.8}
LEAD_TYPES = ["Refinance", "Purchase"]

FEATURES = [
    "days_since_inquiry",
    "rate_gap",
    "credit_score",
    "engagement",
    "prior_attempts",
    "ch_past_customer",
    "ch_referral",
    "ch_purchased_list",
    "is_refinance",
]

FEATURE_LABELS = {
    "days_since_inquiry": "recent inquiry",
    "rate_gap": "rate gap vs market",
    "credit_score": "credit profile",
    "engagement": "email engagement",
    "prior_attempts": "few prior attempts",
    "ch_past_customer": "past customer",
    "ch_referral": "referral source",
    "ch_purchased_list": "list source",
    "is_refinance": "refinance intent",
}


def _generate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    lead_type = rng.choice(LEAD_TYPES, size=n, p=[0.45, 0.55])
    channel = rng.choice(CHANNELS, size=n, p=[0.12, 0.18, 0.45, 0.25])
    days = np.clip(rng.gamma(shape=1.6, scale=11.0, size=n), 0, 120).round()
    credit = np.clip(rng.normal(712, 52, size=n), 560, 830).round()
    is_refi = lead_type == "Refinance"
    rate_gap = np.where(is_refi, np.clip(rng.normal(0.55, 0.6, size=n), -1.0, 2.5), 0.0).round(2)
    engagement = np.clip(rng.beta(1.6, 3.2, size=n) * 100, 0, 100).round()
    attempts = rng.poisson(1.4, size=n)
    balance = np.clip(rng.lognormal(mean=12.65, sigma=0.42, size=n), 60_000, 1_400_000).round(-3)

    ch = pd.Series(channel).map(CHANNEL_EFFECT).to_numpy()
    unobserved = rng.normal(0, 1.15, size=n)  # what no model sees: timing, life events, competing offers
    logit = (
        -2.35
        + unobserved
        - 0.055 * days
        + 0.70 * rate_gap * is_refi
        + 0.007 * (credit - 700)
        + 0.014 * (engagement - 30)
        - 0.25 * attempts
        + 0.7 * ch
        + 0.15 * is_refi
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    converted = rng.random(n) < p

    return pd.DataFrame(
        {
            "lead_type": lead_type,
            "channel": channel,
            "days_since_inquiry": days,
            "rate_gap": rate_gap,
            "credit_score": credit,
            "engagement": engagement,
            "prior_attempts": attempts,
            "loan_balance": balance,
            "true_p": p,
            "converted": converted,
        }
    )


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["days_since_inquiry"] = -df["days_since_inquiry"]  # sign flipped so "recent" is positive
    X["rate_gap"] = df["rate_gap"]
    X["credit_score"] = df["credit_score"]
    X["engagement"] = df["engagement"]
    X["prior_attempts"] = -df["prior_attempts"]
    X["ch_past_customer"] = (df["channel"] == "Past customer").astype(float)
    X["ch_referral"] = (df["channel"] == "Referral").astype(float)
    X["ch_purchased_list"] = (df["channel"] == "Purchased list").astype(float)
    X["is_refinance"] = (df["lead_type"] == "Refinance").astype(float)
    return X[FEATURES]


@dataclass
class Scored:
    today: pd.DataFrame
    auc: float
    coef: pd.Series


@st.cache_data(show_spinner=False)
def build(seed: int = 7, history: int = 24_000, pipeline: int = 6_000) -> Scored:
    rng = np.random.default_rng(seed)
    hist = _generate(history, rng)
    today = _generate(pipeline, rng)

    scaler = StandardScaler()
    Xh = scaler.fit_transform(_design(hist))
    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(Xh, hist["converted"])

    Xt = scaler.transform(_design(today))
    score = model.predict_proba(Xt)[:, 1]
    auc = float(roc_auc_score(today["converted"], score))

    contrib = Xt * model.coef_[0]
    order = np.argsort(-contrib, axis=1)
    reasons = []
    for row, idx in zip(contrib, order):
        parts = [FEATURE_LABELS[FEATURES[j]] for j in idx[:2] if row[j] > 0.05]
        reasons.append(" · ".join(parts) if parts else "no strong signal")

    today = today.copy()
    today["score"] = score
    today["reasons"] = reasons
    today["lead_id"] = [f"L-{40000 + i:05d}" for i in range(len(today))]
    today["action"] = _action(today)
    return Scored(today=today, auc=auc, coef=pd.Series(model.coef_[0], index=FEATURES))


def _action(df: pd.DataFrame) -> list[str]:
    out = []
    for r in df.itertuples():
        if r.credit_score < 620:
            out.append("Review eligibility")
        elif r.days_since_inquiry <= 2:
            out.append("Call today")
        elif r.lead_type == "Refinance" and r.rate_gap >= 1.0:
            out.append("Review refinance fit")
        elif r.channel == "Past customer":
            out.append("Personal outreach")
        elif r.engagement >= 55:
            out.append("Follow up")
        else:
            out.append("Nurture sequence")
    return out


def strategy_order(df: pd.DataFrame, strategy: str, seed: int = 11) -> pd.DataFrame:
    if strategy == "Model score":
        return df.sort_values("score", ascending=False)
    if strategy == "Newest first":
        return df.sort_values(["days_since_inquiry", "lead_id"], ascending=[True, True])
    rng = np.random.default_rng(seed)
    return df.iloc[rng.permutation(len(df))]


def gains(df: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of expected conversions captured vs share of pipeline worked."""
    ordered = strategy_order(df, strategy)
    expected = ordered["true_p"].to_numpy()
    x = np.arange(1, len(expected) + 1) / len(expected)
    y = np.cumsum(expected) / expected.sum()
    return x, y


def expected_conversions(df: pd.DataFrame, strategy: str, capacity: int) -> float:
    ordered = strategy_order(df, strategy)
    return float(ordered["true_p"].head(capacity).sum())
