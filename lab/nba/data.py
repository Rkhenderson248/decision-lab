"""Fernwood Outfitters: a fictional outdoor-gear retailer with 60,000 customers and six things it can do each month.

The true response to every action is known (that is the point of simulating it): each customer has a chance of
buying with no contact and a different lift for each offer, and offers cost money and margin. The company's past
decisions come from a rules-based policy with 30% random exploration, and the probability of every logged action
(the propensity) was recorded. That log is all a model is allowed to learn from; the truth is used only to grade.
Deterministic: the same seed gives the same customers and the same log.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 4040
N = 60_000
EPS = .30                                    # share of the log chosen uniformly at random (exploration)
LIFT = 2.0                                   # scale of the true offer effects on the log-odds of buying
ACTIONS = ["No contact", "10% coupon", "Free shipping", "New-arrivals email", "Loyalty points ×3", "Gear bundle"]
# action: (cost per contact $, margin given up per purchase as a share of basket)
ACOST = {"No contact": (0.00, .00), "10% coupon": (0.15, .10), "Free shipping": (0.15, .06), "New-arrivals email": (0.02, .00),
         "Loyalty points ×3": (0.10, .03), "Gear bundle": (0.25, .05)}
MARGIN = .42                                  # gross margin on a basket
AFFINITY = ["Camping", "Climbing", "Apparel", "Footwear"]


@dataclass
class World:
    customers: pd.DataFrame
    p_true: np.ndarray        # N x A purchase probability under each action
    basket: np.ndarray        # N expected basket value $
    value_true: np.ndarray    # N x A expected profit under each action
    log: pd.DataFrame         # the logged month: action, propensity, purchased, profit
    hash: str


def _features(rng, n):
    tenure = rng.gamma(2.0, 14, n).clip(1, 120).round()                      # months
    freq = rng.poisson(1 + tenure / 18).clip(0, 25)                           # orders in last 12 months
    recency = np.where(freq > 0, rng.exponential(60 / (1 + freq / 3), n), rng.uniform(200, 720, n)).round()
    spend = np.exp(rng.normal(4.6, .55, n)) * (1 + freq / 6)                  # avg basket $
    price_sens = rng.beta(2, 3, n)
    email_eng = rng.beta(2, 2.5, n)
    app_user = (rng.random(n) < .35 + .2 * email_eng).astype(int)
    aff = rng.choice(len(AFFINITY), n, p=[.32, .14, .34, .20])
    loyalty = (rng.random(n) < .25 + .03 * np.minimum(freq, 10)).astype(int)
    return pd.DataFrame({"tenure_months": tenure, "orders_12m": freq, "days_since_order": recency, "avg_basket": spend.round(2),
                         "price_sensitivity": price_sens.round(3), "email_engagement": email_eng.round(3), "app_user": app_user,
                         "affinity": np.array(AFFINITY)[aff], "loyalty_member": loyalty})


def simulate(seed: int = SEED, n: int = N) -> World:
    rng = np.random.default_rng(seed)
    X = _features(rng, n)
    lg = lambda z: 1 / (1 + np.exp(-z))                                       # noqa: E731
    f, r, ps, ee = X.orders_12m.values, X.days_since_order.values, X.price_sensitivity.values, X.email_engagement.values
    base_z = -3.0 + .22 * np.minimum(f, 10) - .006 * np.minimum(r, 365) + .5 * ee + .2 * X.loyalty_member.values
    p0 = lg(base_z)
    lifts = {
        "No contact": np.zeros(n),
        "10% coupon": 1.6 * ps ** 1.5 * (1 + (r > 120)) * .55,                     # works on price-sensitive and lapsing
        "Free shipping": .25 * ps + .10 * (X.avg_basket.values < 80),
        "New-arrivals email": .9 * ee * (X.affinity.isin(["Climbing", "Apparel"]).values * .8 + .3) - .15 * (ee < .2),
        "Loyalty points ×3": .65 * X.loyalty_member.values * (f >= 3) + .1,
        "Gear bundle": .8 * (X.affinity.values == "Camping") * (X.avg_basket.values > 110) + .05,
    }
    p = np.column_stack([lg(base_z + LIFT * lifts[a]) for a in ACTIONS])
    p[:, 2] = np.maximum(p[:, 2], p0)                                          # free shipping never hurts
    basket = np.repeat(X.avg_basket.values[:, None], len(ACTIONS), axis=1)
    basket[:, ACTIONS.index("Gear bundle")] *= 1.15
    cost = np.array([ACOST[a][0] for a in ACTIONS])
    give = np.array([ACOST[a][1] for a in ACTIONS])
    value = p * basket * (MARGIN - give) - cost
    # logging policy: old rules, then 20% uniform exploration; propensities recorded
    rules = np.where(r > 150, 1, np.where(X.loyalty_member.values == 1, 4, np.where(ee > .55, 3, 2)))
    eps = EPS
    prop = np.full((n, len(ACTIONS)), eps / len(ACTIONS))
    prop[np.arange(n), rules] += 1 - eps
    a = (rng.random(n)[:, None] > np.cumsum(prop, axis=1)).sum(axis=1).clip(0, len(ACTIONS) - 1)
    bought = rng.random(n) < p[np.arange(n), a]
    profit = np.where(bought, basket[np.arange(n), a] * (MARGIN - give[a]), 0) - cost[a]
    log = pd.DataFrame({"action": np.array(ACTIONS)[a], "action_id": a, "propensity": prop[np.arange(n), a], "rule_action": rules,
                        "purchased": bought.astype(int), "profit": profit.round(4)})
    h = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([X, log], axis=1), index=False).values.tobytes()).hexdigest()[:12]
    return World(X, p, basket[:, 0], value, log, h)
