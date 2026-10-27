"""Calder Lane: a fictional property manager with a resident-lending arm, simulated month by month.

Calder Lane Residential manages about 120 apartment properties (classes A, B and C). Its finance arm, Calder Lane
Credit, makes small installment loans to residents for move-in costs, security deposits and furnishing packages
($1,500 to $9,000, 12 to 36 months). Every loan is simulated from origination to payoff, charge-off or the data
cutoff, through the delinquency states a servicing system records (current, 30, 60 and 90+ days past due).

The data is synthetic, with known relationships, so every model can be checked against the truth. No real resident,
client or employer data. Deterministic: the same seed always gives the same book.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2021-01-01")
N_ORIG_MONTHS = 48          # originations Jan 2021 to Dec 2024
CUTOFF = 53                 # performance observed through Jun 2025 (month index from START)
PER_MONTH = 400
SEED = 20210101
STATES = ["Current", "30 DPD", "60 DPD", "Charged off", "Prepaid", "Paid off"]
CLASSES = ["A", "B", "C"]


def month_label(m: int) -> str:
    return (START + pd.DateOffset(months=int(m))).strftime("%Y-%m")


def unemployment() -> np.ndarray:
    """Regional unemployment by month (synthetic): recovery in 2021, a 2023 softening, then a plateau."""
    m = np.arange(0, 72)
    u = 6.2 - 2.4 * (1 - np.exp(-m / 9.0))                       # recovery to ~3.8
    u += 0.95 * np.exp(-((m - 30) / 6.5) ** 2)                    # 2023 softening, peak mid-2023
    u += np.where(m > 36, 0.35 * (1 - np.exp(-(m - 36) / 8.0)), 0)  # plateau a little higher
    return np.round(u, 2)


def annuity_balance(amount: np.ndarray, apr: np.ndarray, term: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Scheduled balance after k payments of a level-payment loan."""
    r = apr / 12.0
    k = np.clip(k, 0, term)
    g = (1 + r) ** term
    return amount * (g - (1 + r) ** k) / (g - 1)


@dataclass
class Book:
    loans: pd.DataFrame
    perf: pd.DataFrame
    macro: pd.DataFrame
    hash: str


def simulate(seed: int = SEED, per_month: int = PER_MONTH) -> Book:
    rng = np.random.default_rng(seed)
    n = per_month * N_ORIG_MONTHS
    orig = np.repeat(np.arange(N_ORIG_MONTHS), per_month)
    u = unemployment()

    # ---------------- applicant and property characteristics
    pclass = rng.choice(3, size=n, p=[0.30, 0.45, 0.25])
    occupancy = np.clip(np.array([0.965, 0.935, 0.885])[pclass] + rng.normal(0, 0.02, n), 0.75, 0.995)
    score = np.clip(rng.normal(662, 62, n) + np.array([22, 0, -24])[pclass], 500, 820).round()
    rti = np.clip(rng.beta(5, 10, n) * 0.85 + 0.05 + np.array([-0.02, 0, 0.03])[pclass], 0.12, 0.70)
    tenure = np.clip(rng.exponential(26, n), 0, 180).round()
    evictions = rng.choice(3, size=n, p=[0.86, 0.11, 0.03])
    verified = (rng.random(n) < 0.72 + 0.0006 * (score - 660)).astype(int)
    product = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])          # move-in, deposit, furnishing
    amount = np.clip(rng.lognormal(np.log(np.array([3200, 2400, 5600])[product]), 0.32), 1500, 9000).round(-1)
    term = np.where(amount > 5200, 36, np.where(amount > 3000, 24, 12))
    deposit = np.clip(rng.normal(0.18, 0.12, n) * amount + np.where(product == 1, 0, 300), 0, None).round(-1)
    dep_cov = np.clip(deposit / amount, 0, 0.9)
    apr = np.clip(0.11 + 0.0011 * (700 - score) + 0.04 * (evictions > 0) + rng.normal(0, 0.01, n), 0.08, 0.36).round(4)
    u_orig = u[orig]

    # ---------------- monthly simulation of the delinquency process
    # Credit sense with thresholds and interactions a straight line cannot draw: risk climbs steeply below a 600
    # score, when rent passes ~40% of income for thin-file applicants, at short tenure and at poorly occupied
    # properties (a sign of a struggling building), and a deposit cushion matters most when rent is stretched.
    base = (-4.25 - 0.008 * (score - 660) + 0.030 * np.maximum(0, 600 - score) + 2.2 * (rti - 0.33)
            + 7.5 * np.maximum(0, rti - 0.40) * (score < 650) - 1.0 * dep_cov - 2.2 * dep_cov * (rti > 0.38)
            + 0.50 * evictions + np.array([-0.15, 0.0, 0.30])[pclass] - 2.0 * (occupancy - 0.93)
            + 14.0 * np.maximum(0, 0.90 - occupancy) - 0.35 * verified + 0.9 * (tenure < 6) - 0.004 * np.minimum(tenure, 60)
            + np.array([0.0, 0.10, -0.05])[product])
    frail = rng.normal(0, 0.35, n)                                   # unobserved heterogeneity
    state = np.full(n, -1)                                           # -1 not yet originated
    default_m = np.full(n, -1)
    end_m = np.full(n, -1)
    end_state = np.full(n, -1)
    rows = []
    for m in range(CUTOFF + 1):
        new = orig == m
        state[new] = 0
        act = np.where((state >= 0) & (state <= 2))[0]
        if len(act) == 0:
            continue
        mob = m - orig[act]
        bal = annuity_balance(amount[act], apr[act], term[act], mob)
        rows.append(np.column_stack([act, np.full(len(act), m), mob, state[act], bal.round(2)]))
        if m == CUTOFF:
            break
        s = state[act]
        season = 0.55 * np.exp(-((mob - 6) / 5.0) ** 2) - 0.25 * (mob > 18)
        h = 1 / (1 + np.exp(-(base[act] + frail[act] + season + 0.65 * (u[m] - 4.5))))
        r = rng.random(len(act))
        nxt = s.copy()
        cur = s == 0
        pre = 0.010 + 0.004 * (score[act] > 700)
        nxt[cur & (r < h)] = 1
        nxt[cur & (r >= h) & (r < h + pre)] = 4
        d30 = s == 1
        risk = 1 / (1 + np.exp(-(base[act] + frail[act] + 3.4)))      # roll severity rises with risk
        nxt[d30] = np.select([r[d30] < 0.42 - 0.15 * risk[d30], r[d30] < 0.80], [0, 2], 1)
        d60 = s == 2
        p_co = 0.48 + 0.25 * risk[d60]
        nxt[d60] = np.select([r[d60] < p_co, r[d60] < p_co + 0.17, r[d60] < p_co + 0.27], [3, 0, 1], 2)
        matured = (mob + 1 >= term[act]) & np.isin(nxt, [0, 1])
        nxt[matured] = 5
        state[act] = nxt
        done = act[np.isin(nxt, [3, 4, 5])]
        end_m[done] = m + 1
        end_state[done] = nxt[np.isin(nxt, [3, 4, 5])]
        default_m[act[nxt == 3]] = m + 1
    perf = pd.DataFrame(np.vstack(rows), columns=["loan_id", "month", "mob", "status", "balance"])
    perf[["loan_id", "month", "mob", "status"]] = perf[["loan_id", "month", "mob", "status"]].astype(int)

    # ---------------- loss given default: deposit first, then a workout recovery
    defaulted = default_m >= 0
    ead = np.where(defaulted, annuity_balance(amount, apr, term, np.maximum(default_m - orig - 3, 0)), 0.0)
    rec_mean = np.array([0.34, 0.27, 0.17])[pclass] + 0.05 * verified + 0.0006 * (score - 660)
    rec_rate = rng.beta(np.clip(rec_mean, 0.03, 0.9) * 6, (1 - np.clip(rec_mean, 0.03, 0.9)) * 6)
    recovery = np.minimum(ead, deposit + rec_rate * np.maximum(ead - deposit, 0))
    lgd = np.where(defaulted & (ead > 0), np.clip(1 - recovery / np.maximum(ead, 1), 0, 1), np.nan)

    loans = pd.DataFrame({
        "loan_id": np.arange(n), "orig_month": orig, "orig_date": [month_label(x) for x in orig],
        "vintage": 2021 + orig // 12, "product": np.array(["Move-in", "Deposit", "Furnishing"])[product],
        "property_class": np.array(CLASSES)[pclass], "occupancy": occupancy.round(3), "credit_score": score.astype(int),
        "rent_to_income": rti.round(3), "months_at_address": tenure.astype(int), "prior_evictions": evictions,
        "income_verified": verified, "amount": amount, "term": term, "apr": apr, "deposit": deposit,
        "deposit_coverage": dep_cov.round(3), "unemployment_at_orig": u_orig,
        "default_month": default_m, "end_month": end_m,
        "end_state": np.where(end_state >= 0, np.array(STATES)[np.maximum(end_state, 0)], "Active"),
        "ead": ead.round(2), "recovery": recovery.round(2), "lgd": np.round(lgd, 4),
    })
    mob_def = np.where(defaulted, default_m - orig, 999)
    observed12 = orig + 12 <= CUTOFF
    loans["observed_12m"] = observed12.astype(int)
    loans["default_12m"] = np.where(observed12, (mob_def <= 12).astype(int), -1)
    macro = pd.DataFrame({"month": np.arange(len(u)), "date": [month_label(x) for x in range(len(u))], "unemployment": u})
    h = hashlib.sha256(pd.util.hash_pandas_object(loans, index=False).values.tobytes()).hexdigest()[:12]
    return Book(loans, perf, macro, h)


def warehouse(book: Book) -> sqlite3.Connection:
    """The book as a small SQL warehouse: loans, monthly performance and macro."""
    con = sqlite3.connect(":memory:", check_same_thread=False)
    book.loans.to_sql("loans", con, index=False)
    p = book.perf.copy()
    p["status"] = np.array(STATES)[p["status"]]
    p.to_sql("performance", con, index=False)
    book.macro.to_sql("macro", con, index=False)
    con.execute("CREATE INDEX ix_perf ON performance(loan_id, month)")
    return con


SQL_VINTAGE = """-- Cumulative charge-off rate by vintage and months on book
WITH first_co AS (
  SELECT loan_id, MIN(mob) + 1 AS co_mob
  FROM performance WHERE status = '60 DPD'
    AND loan_id IN (SELECT loan_id FROM loans WHERE end_state = 'Charged off')
  GROUP BY loan_id
), grid AS (
  SELECT l.vintage, p.mob, COUNT(DISTINCT l.loan_id) AS at_risk
  FROM loans l JOIN performance p ON p.loan_id = l.loan_id
  GROUP BY l.vintage, p.mob
), events AS (
  SELECT l.vintage, f.co_mob AS mob, COUNT(*) AS charged_off
  FROM first_co f JOIN loans l USING (loan_id)
  GROUP BY l.vintage, f.co_mob
)
SELECT g.vintage, g.mob,
       SUM(COALESCE(e.charged_off, 0)) OVER w * 1.0
         / (SELECT COUNT(*) FROM loans x WHERE x.vintage = g.vintage) AS cum_co_rate
FROM grid g LEFT JOIN events e ON e.vintage = g.vintage AND e.mob = g.mob
WINDOW w AS (PARTITION BY g.vintage ORDER BY g.mob ROWS UNBOUNDED PRECEDING)
ORDER BY g.vintage, g.mob;"""

SQL_ROLL = """-- Roll rates: where each delinquency state goes next month
WITH pairs AS (
  SELECT a.status AS from_status,
         COALESCE(b.status, CASE l.end_state WHEN 'Active' THEN NULL ELSE l.end_state END) AS to_status
  FROM performance a
  JOIN loans l ON l.loan_id = a.loan_id
  LEFT JOIN performance b ON b.loan_id = a.loan_id AND b.month = a.month + 1
  WHERE a.month < (SELECT MAX(month) FROM performance)
)
SELECT from_status, to_status, COUNT(*) AS n,
       ROUND(1.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY from_status), 4) AS share
FROM pairs WHERE to_status IS NOT NULL
GROUP BY from_status, to_status
ORDER BY from_status, n DESC;"""

SQL_DQ = """-- Delinquency and balance by month: the servicing dashboard
SELECT p.month, m.date,
       SUM(p.balance) AS outstanding,
       SUM(CASE WHEN p.status <> 'Current' THEN p.balance ELSE 0 END) * 1.0 / SUM(p.balance) AS dq_rate,
       m.unemployment
FROM performance p JOIN macro m USING (month)
GROUP BY p.month ORDER BY p.month;"""
