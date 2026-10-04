"""Kestrel Valley Credit Union: a fictional lender and everything it knows.

One seeded generator creates every table the lending lab uses, so every
module works on the same members. Relationships are planted on purpose and
documented here, because a demo is only honest if the reader can see what the
models were supposed to find:

- Six latent member archetypes drive behavior, balances and product holdings.
  Segmentation should recover them without being told.
- Default risk rises with lower scores, higher debt-to-income and utilization,
  and many recent inquiries. Applications record it for everyone, but the
  lender only observes it for the loans it approved (so reject inference has
  something to fix).
- About 1.5% of applications are fraudulent, mostly synthetic-identity rings
  that share phones and addresses and overstate income.
- Riskier borrowers are less rate-sensitive (adverse selection), and
  rate-sensitive refinancers prepay quickly when market rates fall.
- A randomised campaign shows that some members respond to an offer, some
  would have bought anyway and some are put off by being contacted.
- A synthetic protected-class proxy ("group B") is correlated with income and
  geography, never used by any model, and present only for fair-lending tests.

Nothing here comes from a real institution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

NAME = "Kestrel Valley Credit Union"
SEED = 7
N_MEMBERS = 50_000

ARCHETYPES = {
    # key: (share, display name)
    "starter": (0.18, "Digital starters"),
    "family": (0.22, "Growing families"),
    "refi": (0.12, "Rate-sensitive refinancers"),
    "affluent": (0.13, "Affluent savers"),
    "builder": (0.17, "Credit builders"),
    "retired": (0.18, "Retired loyalists"),
}
PRODUCTS = ["checking", "savings", "card", "auto", "personal", "mortgage", "heloc"]
LOAN_PRODUCTS = ["card", "auto", "personal", "mortgage", "heloc"]
METROS = {
    # metro: (weight, share of group B, income index)
    "Dallas–Fort Worth": (0.20, 0.38, 1.05),
    "Houston": (0.14, 0.42, 1.00),
    "Austin": (0.09, 0.28, 1.18),
    "San Antonio": (0.09, 0.52, 0.88),
    "Oklahoma City": (0.07, 0.26, 0.86),
    "Tulsa": (0.05, 0.24, 0.84),
    "Denver": (0.09, 0.22, 1.15),
    "Albuquerque": (0.05, 0.48, 0.82),
    "Kansas City": (0.07, 0.24, 0.95),
    "Little Rock": (0.04, 0.34, 0.80),
    "Wichita": (0.04, 0.20, 0.83),
    "El Paso": (0.07, 0.70, 0.74),
}

# Per-archetype parameters. Means unless stated; draws add noise.
P = {
    #            age lo,hi  tenure  income  score sd  dti  util  deposits  txn  digital  inq   products: checking savings card auto personal mortgage heloc
    "starter":  (22, 33,   2.0,    46e3,   668, 55, .30, .48,   3_500,    62,  .88,    2.2, (.95, .55, .70, .30, .18, .03, .00)),
    "family":   (31, 48,   7.0,    96e3,   716, 45, .36, .30,   12_000,   78,  .66,    1.2, (.97, .80, .72, .58, .16, .55, .10)),
    "refi":     (34, 56,   6.0,    122e3,  742, 38, .33, .22,   18_000,   54,  .60,    2.6, (.95, .78, .62, .28, .06, 1.0, .12)),
    "affluent": (48, 70,   14.0,   178e3,  786, 30, .18, .10,   96_000,   40,  .52,    0.5, (.98, .97, .55, .20, .03, .30, .26)),
    "builder":  (24, 46,   3.0,    41e3,   618, 46, .41, .62,   1_400,    48,  .64,    3.4, (.92, .35, .62, .34, .42, .04, .00)),
    "retired":  (65, 86,   24.0,   52e3,   762, 34, .14, .12,   41_000,   22,  .22,    0.3, (.99, .92, .42, .12, .04, .12, .08)),
}


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_pd(score, dti, util, income, inquiries) -> np.ndarray:
    """The planted 12-month default probability."""
    z = (-4.0 - 0.0125 * (score - 700) + 2.4 * (dti - 0.35) + 2.3 * (util - 0.30)
         - 0.30 * np.log(income / 70_000) + 0.26 * inquiries)
    return _logistic(z)


def _population(rng: np.random.Generator, n: int, prospects: bool = False) -> pd.DataFrame:
    """Draw people from the archetype mixture. Prospects have no relationship history."""
    keys = list(ARCHETYPES)
    shares = np.array([ARCHETYPES[k][0] for k in keys])
    arch = rng.choice(keys, n, p=shares / shares.sum())
    metro_names = list(METROS)
    mw = np.array([METROS[m][0] for m in metro_names])
    metro = rng.choice(metro_names, n, p=mw / mw.sum())
    m_b = np.array([METROS[m][1] for m in metro])
    m_inc = np.array([METROS[m][2] for m in metro])

    cols = {k: np.empty(n) for k in ("age", "tenure", "income", "score", "dti", "util", "deposits", "txn", "digital", "inq")}
    held = np.zeros((n, len(PRODUCTS)), dtype=bool)
    for k in keys:
        idx = np.flatnonzero(arch == k)
        if not len(idx):
            continue
        a_lo, a_hi, ten, inc, sc, sd, dti, util, dep, txn, dig, inq, prod = P[k]
        m = len(idx)
        cols["age"][idx] = rng.uniform(a_lo, a_hi, m)
        cols["tenure"][idx] = np.minimum(rng.gamma(2.0, ten / 2.0, m), cols["age"][idx] - 18)
        cols["income"][idx] = inc * m_inc[idx] * rng.lognormal(0, 0.28, m)
        cols["score"][idx] = rng.normal(sc, sd, m)
        cols["dti"][idx] = rng.normal(dti, 0.08, m)
        cols["util"][idx] = rng.beta(2, 2, m) * 0.5 + util - 0.25
        cols["deposits"][idx] = dep * rng.lognormal(0, 0.8, m)
        cols["txn"][idx] = rng.poisson(txn, m)
        cols["digital"][idx] = np.clip(rng.normal(dig, 0.12, m), 0, 1)
        cols["inq"][idx] = rng.poisson(inq, m)
        held[idx] = rng.random((m, len(PRODUCTS))) < np.array(prod)

    # Synthetic protected-class proxy: more common in lower-income metros, never a model input.
    group_b = rng.random(n) < np.clip(m_b + 0.10 * (arch == "builder") - 0.08 * (arch == "affluent"), 0.02, 0.95)
    # A small, deliberate structural gap that flows through income and score, as in real data.
    cols["income"] *= np.where(group_b, 0.90, 1.0)
    cols["score"] -= np.where(group_b, 12, 0)

    df = pd.DataFrame({
        "archetype": arch,
        "metro": metro,
        "group_b": group_b,
        "age": cols["age"].round(0).astype(int),
        "tenure_years": np.clip(cols["tenure"], 0, None).round(1),
        "income": cols["income"].round(-2),
        "credit_score": np.clip(cols["score"], 300, 850).round(0).astype(int),
        "dti": np.clip(cols["dti"], 0.02, 0.70).round(3),
        "utilization": np.clip(cols["util"], 0, 1).round(3),
        "deposit_balance": cols["deposits"].round(0),
        "monthly_txn": cols["txn"].astype(int),
        "digital_share": cols["digital"].round(2),
        "inquiries_6m": cols["inq"].astype(int),
    })
    if prospects:
        df["tenure_years"] = 0.0
        df["deposit_balance"] = 0.0
        df["monthly_txn"] = 0
        held[:] = False
    for j, p in enumerate(PRODUCTS):
        df[f"has_{p}"] = held[:, j]
    return df


@dataclass
class Bank:
    members: pd.DataFrame
    applications: pd.DataFrame
    quotes: pd.DataFrame
    campaign: pd.DataFrame
    mortgages: pd.DataFrame
    mortgage_history: pd.DataFrame
    delinquency: pd.DataFrame
    treatments: pd.DataFrame
    prospects: pd.DataFrame
    market_rates: pd.Series


def _members(rng) -> pd.DataFrame:
    m = _population(rng, N_MEMBERS)
    m.insert(0, "member_id", [f"KV{100000 + i}" for i in range(len(m))])
    n = len(m)
    a = m["archetype"].to_numpy()
    # Balances for held products.
    m["card_balance"] = np.where(m["has_card"], m["income"] * m["utilization"] * rng.uniform(0.04, 0.10, n), 0).round(0)
    m["auto_balance"] = np.where(m["has_auto"], rng.uniform(8e3, 42e3, n), 0).round(0)
    m["personal_balance"] = np.where(m["has_personal"], rng.uniform(2e3, 18e3, n), 0).round(0)
    m["mortgage_balance"] = np.where(m["has_mortgage"], m["income"] * rng.uniform(1.8, 3.6, n), 0).round(-2)
    m["heloc_balance"] = np.where(m["has_heloc"], rng.uniform(10e3, 80e3, n), 0).round(0)
    # Mortgage note rates: refinancers locked in when rates were high.
    note = np.where(a == "refi", rng.uniform(6.6, 7.9, n), rng.choice([3.1, 3.6, 4.4, 5.6, 6.4, 7.1], n, p=[.22, .20, .16, .16, .14, .12]) + rng.normal(0, .15, n))
    m["mortgage_rate"] = np.where(m["has_mortgage"], note, np.nan).round(3)
    m["products_held"] = m[[f"has_{p}" for p in PRODUCTS]].sum(axis=1)
    m["months_since_last_product"] = np.clip(rng.gamma(2, 10 + 6 * (a == "retired"), n), 0, 240).round(0).astype(int)
    m["complaints_12m"] = rng.poisson(0.10 + 0.22 * (a == "starter") + 0.15 * (a == "builder"), n)
    m["pd_true"] = true_pd(m["credit_score"], m["dti"], m["utilization"], m["income"], m["inquiries_6m"])
    m["default_12m"] = rng.random(n) < m["pd_true"]
    churn_z = (-2.6 + 0.9 * (m["tenure_years"] < 2) - 0.35 * (m["products_held"] - 3) + 0.55 * m["complaints_12m"]
               + 0.5 * (a == "starter") + 0.6 * (a == "refi") - 0.8 * (a == "retired") - 0.6 * m["digital_share"] * (a == "retired"))
    m["churn_true"] = _logistic(churn_z)
    m["churned_12m"] = rng.random(n) < m["churn_true"]
    return m


def _applications(rng, members: pd.DataFrame, n: int = 24_000) -> pd.DataFrame:
    """Loan applications from members and walk-in applicants, with fraud rings."""
    n_member = int(n * 0.68)
    from_members = members.sample(n_member, random_state=int(rng.integers(1e9)), replace=True).reset_index(drop=True)
    walk = _population(rng, n - n_member, prospects=True)
    walk["member_id"] = None
    cols = ["member_id", "archetype", "metro", "group_b", "age", "tenure_years", "income", "credit_score", "dti",
            "utilization", "deposit_balance", "monthly_txn", "digital_share", "inquiries_6m"]
    apps = pd.concat([from_members[cols], walk[cols]], ignore_index=True)
    apps = apps.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    k = len(apps)
    apps.insert(0, "app_id", [f"A{300000 + i}" for i in range(k)])
    product_p = {"starter": [.40, .30, .25, .03, .02], "family": [.18, .42, .10, .18, .12], "refi": [.10, .20, .05, .45, .20],
                 "affluent": [.20, .25, .03, .17, .35], "builder": [.38, .30, .30, .01, .01], "retired": [.30, .30, .10, .10, .20]}
    prods = ["card", "auto", "personal", "mortgage", "heloc"]
    product = np.empty(k, dtype=object)
    arch_arr = apps["archetype"].to_numpy()
    for a_key, probs in product_p.items():
        idx = np.flatnonzero(arch_arr == a_key)
        product[idx] = rng.choice(prods, len(idx), p=probs)
    apps["product"] = product
    size = {"card": (2e3, 15e3), "auto": (12e3, 48e3), "personal": (2e3, 25e3), "mortgage": (150e3, 520e3), "heloc": (20e3, 120e3)}
    lo = apps["product"].map({p_: v[0] for p_, v in size.items()}).to_numpy()
    hi = apps["product"].map({p_: v[1] for p_, v in size.items()}).to_numpy()
    apps["amount"] = (lo + rng.random(k) * (hi - lo)).round(-2)
    apps["month"] = rng.integers(0, 24, k)  # months since start of the window
    apps["email_age_days"] = np.clip(rng.gamma(2.5, 900, k), 1, 9000).round(0)
    apps["stated_income"] = (apps["income"] * rng.lognormal(0.0, 0.08, k)).round(-2)
    apps["phone_id"] = [f"P{x}" for x in rng.integers(0, 10_000_000, k)]
    apps["address_id"] = [f"H{x}" for x in rng.integers(0, 10_000_000, k)]
    apps["device_apps_24h"] = rng.poisson(0.15, k) + 1

    # Fraud: synthetic-identity rings sharing phones and addresses, plus a few lone fabrications.
    fraud = np.zeros(k, dtype=bool)
    ring_id = np.full(k, -1)
    candidates = np.flatnonzero(apps["member_id"].isna().to_numpy())
    rng.shuffle(candidates)
    pos, r = 0, 0
    while fraud.sum() < int(k * 0.013) and pos < len(candidates) - 10:
        size_r = int(rng.integers(3, 9))
        idx = candidates[pos:pos + size_r]
        pos += size_r
        fraud[idx] = True
        ring_id[idx] = r
        phones = [f"PR{r}_{j}" for j in range(max(1, size_r // 3))]
        apps.loc[idx, "phone_id"] = rng.choice(phones, len(idx))
        apps.loc[idx, "address_id"] = f"HR{r}"
        apps.loc[idx, "month"] = int(rng.integers(0, 23)) + rng.integers(0, 2, len(idx))
        r += 1
    lone = rng.choice(np.setdiff1d(np.arange(k), np.flatnonzero(fraud)), int(k * 0.003), replace=False)
    fraud[lone] = True
    f_idx = np.flatnonzero(fraud)
    apps.loc[f_idx, "stated_income"] = (apps.loc[f_idx, "income"] * rng.uniform(1.5, 2.6, len(f_idx))).round(-2)
    apps.loc[f_idx, "email_age_days"] = rng.integers(1, 60, len(f_idx))
    apps.loc[f_idx, "device_apps_24h"] = rng.integers(2, 7, len(f_idx))
    apps.loc[f_idx, "credit_score"] = np.clip(rng.normal(690, 35, len(f_idx)), 560, 800).round(0)
    apps.loc[f_idx, "inquiries_6m"] = rng.integers(3, 9, len(f_idx))
    # Honest applicants trip the same signals sometimes: households share an
    # address, people open new email accounts, some round their income up.
    legit = np.flatnonzero(~fraud)
    house = rng.choice(legit, int(k * 0.06), replace=False)
    for h_i, start in enumerate(range(0, len(house) - 3, 3)):
        apps.loc[house[start:start + int(rng.integers(2, 5))], "address_id"] = f"HH{h_i}"
    apps.loc[rng.choice(legit, int(k * 0.05), replace=False), "email_age_days"] = rng.integers(1, 45, int(k * 0.05))
    over = rng.choice(legit, int(k * 0.05), replace=False)
    apps.loc[over, "stated_income"] = (apps.loc[over, "income"] * rng.uniform(1.4, 2.0, len(over))).round(-2)
    apps.loc[rng.choice(legit, int(k * 0.03), replace=False), "device_apps_24h"] = rng.integers(2, 5, int(k * 0.03))
    # A third of the fraud is quieter: an aged email, a believable income, one application per device.
    quiet = rng.choice(f_idx, len(f_idx) // 3, replace=False)
    apps.loc[quiet, "email_age_days"] = rng.integers(200, 3000, len(quiet))
    apps.loc[quiet, "stated_income"] = (apps.loc[quiet, "income"] * rng.uniform(1.0, 1.3, len(quiet))).round(-2)
    apps.loc[quiet, "device_apps_24h"] = 1
    apps["is_fraud"] = fraud
    apps["fraud_ring"] = ring_id

    # The market shifts in the final six months: more credit shopping, slightly weaker files.
    recent = (apps["month"] >= 18).to_numpy()
    apps.loc[recent, "inquiries_6m"] += rng.poisson(0.35, recent.sum())
    apps.loc[recent, "credit_score"] = (apps.loc[recent, "credit_score"] - rng.normal(9, 5, recent.sum())).clip(300, 850).round(0)
    apps.loc[recent, "utilization"] = (apps.loc[recent, "utilization"] + 0.03).clip(0, 1)
    apps["pd_true"] = true_pd(apps["credit_score"], apps["dti"], apps["utilization"], apps["income"], apps["inquiries_6m"])
    apps.loc[fraud, "pd_true"] = 0.85  # a fraudulent loan is a loss
    apps["default_12m"] = rng.random(k) < apps["pd_true"]
    # The legacy policy: score and DTI cut-offs with a few judgemental overrides.
    legacy = (apps["credit_score"] >= 640) & (apps["dti"] <= 0.45)
    override = rng.random(k) < 0.04
    apps["approved"] = legacy ^ override
    apps["observed_default"] = np.where(apps["approved"], apps["default_12m"], np.nan)
    return apps


def _quotes(rng, members: pd.DataFrame, n: int = 18_000) -> pd.DataFrame:
    """Auto-loan quotes with a little randomised rate variation, and whether each was taken."""
    q = members.sample(n, random_state=int(rng.integers(1e9)), replace=True).reset_index(drop=True)
    q = q[["member_id", "archetype", "credit_score", "dti", "utilization", "income", "inquiries_6m"]].copy()
    q["pd"] = true_pd(q["credit_score"], q["dti"], q["utilization"], q["income"], q["inquiries_6m"])
    q["tier"] = pd.cut(q["credit_score"], [0, 619, 679, 739, 900], labels=["D", "C", "B", "A"]).astype(str)
    base = {"A": 5.4, "B": 6.6, "C": 8.9, "D": 12.4}
    q["competitor_rate"] = q["tier"].map(base) + rng.normal(0, 0.25, n)
    q["rate"] = (q["competitor_rate"] + rng.normal(0.15, 0.55, n)).round(2)  # rate tests and branch discretion
    sens = {"starter": 1.15, "family": 0.95, "refi": 1.35, "affluent": 0.75, "builder": 0.55, "retired": 0.70}
    # Adverse selection: the riskier the borrower, the less the rate matters to them.
    beta = q["archetype"].map(sens) * (1.0 - 2.2 * q["pd"].clip(0, 0.35))
    z = 0.35 - 1.05 * beta * (q["rate"] - q["competitor_rate"]) + 0.4 * (q["tier"] == "D")
    q["accepted"] = rng.random(n) < _logistic(z)
    q["default_12m"] = rng.random(n) < q["pd"] * np.where(q["accepted"], 1.0, 0.0) * (1 + 0.6 * (q["rate"] - q["competitor_rate"]).clip(0, None))
    return q


def _campaign(rng, members: pd.DataFrame) -> pd.DataFrame:
    """Last quarter's randomised credit-card campaign: half the eligible members were contacted."""
    elig = members[~members["has_card"]].copy()
    n = len(elig)
    treated = rng.random(n) < 0.5
    a = elig["archetype"].to_numpy()
    base_z = (-3.1 + 0.9 * (a == "starter") + 0.5 * (a == "family") + 0.3 * (a == "builder") - 0.4 * (a == "retired")
              + 1.5 * (a == "affluent") + 0.6 * (a == "refi")
              + 0.010 * (elig["credit_score"].to_numpy() - 700) + 0.25 * np.log(elig["income"].to_numpy() / 60_000))
    uplift = {"starter": 1.10, "family": 0.75, "refi": 0.05, "affluent": 0.0, "builder": 0.70, "retired": -0.65}
    lift = np.array([uplift[x] for x in a])
    p = _logistic(base_z + treated * lift)
    elig["treated"] = treated
    elig["adopted"] = rng.random(n) < p
    elig["p0_true"] = _logistic(base_z)
    elig["p1_true"] = _logistic(base_z + lift)
    return elig.reset_index(drop=True)


def _market_rates() -> pd.Series:
    """Monthly 30-year mortgage rate over the four-year history (illustrative)."""
    months = pd.period_range("2022-07", periods=48, freq="M")
    x = np.arange(48)
    rate = 5.4 + 1.9 * np.sin(np.clip(x, 0, 16) / 16 * np.pi / 2) - 0.035 * np.clip(x - 18, 0, None) + 0.08 * np.sin(x / 2.5)
    return pd.Series(rate.round(2), index=months, name="rate_30y")


def _mortgages(rng, members: pd.DataFrame, rates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The mortgage book and its monthly prepayment history (person-period form)."""
    mort = members[members["has_mortgage"]][["member_id", "archetype", "mortgage_rate", "mortgage_balance", "credit_score", "income", "tenure_years"]].copy()
    mort = mort.reset_index(drop=True)
    n = len(mort)
    mort["start"] = rng.integers(0, 30, n)  # month index the loan entered the window
    seg_eff = mort["archetype"].map({"refi": 0.9, "family": 0.15, "starter": 0.2, "affluent": -0.1, "builder": -0.6, "retired": -0.7}).to_numpy()
    r = rates.to_numpy()
    rows = []
    alive = np.ones(n, dtype=bool)
    event_month = np.full(n, -1)
    for t in range(48):
        active = alive & (mort["start"].to_numpy() <= t)
        if not active.any():
            continue
        idx = np.flatnonzero(active)
        age = t - mort["start"].to_numpy()[idx]
        incentive = mort["mortgage_rate"].to_numpy()[idx] - r[t]
        z = -5.6 + 1.25 * np.clip(incentive, -2, 3) + 0.55 * np.clip(incentive - 0.75, 0, None) * (seg_eff[idx] > 0.5) + seg_eff[idx] + 0.02 * np.minimum(age, 24)
        p = _logistic(z)
        ev = rng.random(len(idx)) < p
        rows.append(pd.DataFrame({"loan": idx, "month": t, "age": age, "incentive": incentive, "prepaid": ev}))
        alive[idx[ev]] = False
        event_month[idx[ev]] = t
    hist = pd.concat(rows, ignore_index=True)
    hist["archetype"] = mort["archetype"].to_numpy()[hist["loan"]]
    mort["prepaid"] = event_month >= 0
    mort["event_month"] = np.where(event_month >= 0, event_month, 47)
    mort["months_observed"] = mort["event_month"] - mort["start"] + 1
    return mort, hist


STATES = ["Current", "30 days", "60 days", "90+ days", "Charged off"]


def _delinquency(rng, members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """24 months of delinquency states for consumer loans, and randomised past collection treatments."""
    loans = members[members["has_auto"] | members["has_personal"] | members["has_card"]][
        ["member_id", "archetype", "credit_score", "income", "auto_balance", "personal_balance", "card_balance"]].copy().reset_index(drop=True)
    loans["balance"] = loans[["auto_balance", "personal_balance", "card_balance"]].sum(axis=1)
    n = len(loans)
    risk = true_pd(loans["credit_score"], 0.35, 0.35, loans["income"], 1).to_numpy()
    state = np.zeros(n, dtype=int)
    hist = np.zeros((n, 24), dtype=int)
    for t in range(24):
        u = rng.random(n)
        roll_c = np.clip(0.004 + 0.18 * risk, 0, 0.3)
        new = state.copy()
        new[(state == 0) & (u < roll_c)] = 1
        cure1 = (state == 1) & (u < 0.45)
        new[cure1] = 0
        new[(state == 1) & ~cure1 & (u < 0.45 + 0.30 + 0.4 * risk)] = 2
        cure2 = (state == 2) & (u < 0.22)
        new[cure2] = 0
        new[(state == 2) & ~cure2 & (u < 0.22 + 0.45 + 0.3 * risk)] = 3
        cure3 = (state == 3) & (u < 0.08)
        new[cure3] = 0
        new[(state == 3) & ~cure3 & (u < 0.08 + 0.35)] = 4
        new[state == 4] = 4
        state = new
        hist[:, t] = state
    loans["state"] = state
    for t in range(24):
        loans[f"m{t}"] = hist[:, t]
    # Past treatments, randomised among delinquent accounts: response = cured within 30 days.
    dq = np.flatnonzero((hist[:, :-1] >= 1).any(axis=1) & (hist[:, :-1] <= 3).any(axis=1))
    k = min(len(dq), 9000)
    pick = rng.choice(dq, k, replace=False)
    st_ = np.clip(rng.integers(1, 4, k), 1, 3)
    treat = rng.choice(["None", "Text", "Call", "Hardship plan"], k, p=[.25, .25, .30, .20])
    a = loans["archetype"].to_numpy()[pick]
    base = np.array([0.30, 0.16, 0.06])[st_ - 1]
    eff = {"None": 0.0, "Text": 0.07, "Call": 0.13, "Hardship plan": 0.22}
    seg_mod = {"starter": {"Text": 1.5, "Call": 0.6}, "retired": {"Text": 0.3, "Call": 1.4}, "builder": {"Hardship plan": 1.5}}
    lift = np.array([eff[t] * seg_mod.get(s, {}).get(t, 1.0) for t, s in zip(treat, a)]) * np.array([1.0, 0.8, 0.5])[st_ - 1]
    cured = rng.random(k) < np.clip(base + lift, 0, 0.95)
    tr = pd.DataFrame({"member_id": loans["member_id"].to_numpy()[pick], "archetype": a, "state": st_, "treatment": treat, "cured": cured})
    return loans, tr


@st.cache_resource(show_spinner="Building Kestrel Valley Credit Union…")
def bank(seed: int = SEED) -> Bank:
    rng = np.random.default_rng(seed)
    members = _members(rng)
    apps = _applications(rng, members)
    quotes = _quotes(rng, members)
    campaign = _campaign(rng, members)
    rates = _market_rates()
    mort, mhist = _mortgages(rng, members, rates)
    dq, tr = _delinquency(rng, members)
    prospects = _population(rng, 30_000, prospects=True)
    prospects.insert(0, "prospect_id", [f"PR{i}" for i in range(len(prospects))])
    return Bank(members, apps, quotes, campaign, mort, mhist, dq, tr, prospects, rates)
