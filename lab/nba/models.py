"""Learn from the log, choose under constraints, and grade the new policy before it ships.

1. Response model: one gradient-boosted classifier over customer features plus the logged action, trained on half of
   the log. It predicts the chance of purchase under every action for every customer (the direct method).
2. Expected profit per action = p(buy | action) × basket × (margin − discount give-up) − contact cost.
3. Policy: the action with the highest predicted profit, subject to capacity (bundles in stock, coupon budget) and a
   contact cap. Capacities are met with shadow prices: each scarce action gets a per-slot price found by bisection,
   so a slot goes to the customers who gain the most from it.
4. Off-policy evaluation on the other half: IPS, self-normalised IPS and doubly robust, with bootstrap intervals.
   Because this is a simulation, the true value of every policy is also known, so each estimator can be graded.
Deterministic throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from lab.nba import data as D

A = len(D.ACTIONS)
COST = np.array([D.ACOST[a][0] for a in D.ACTIONS])
GIVE = np.array([D.ACOST[a][1] for a in D.ACTIONS])
BASKET_MULT = np.array([1.15 if a == "Gear bundle" else 1.0 for a in D.ACTIONS])
CAPACITY = {"Gear bundle": 5_000, "10% coupon": 8_000}       # bundles in stock; coupon budget in redemptions
CONTACT_CAP = .85                                           # at most 85% of customers contacted this month
FEATURES = ["tenure_months", "orders_12m", "days_since_order", "avg_basket", "price_sensitivity", "email_engagement",
            "app_user", "affinity_id", "loyalty_member"]
B = 400                                                     # bootstrap resamples
MIN_GAIN = 1.0                                              # $ per customer a segment must gain (DR) to change policy
SEGMENTS = ["Lapsing", "Loyalty", "Price-sensitive", "Core"]


def segment_of(cust: pd.DataFrame) -> np.ndarray:
    return np.select([cust.days_since_order > 150, cust.loyalty_member == 1, cust.price_sensitivity > .55], SEGMENTS[:3], SEGMENTS[3])


def _x(cust: pd.DataFrame, action: np.ndarray) -> np.ndarray:
    c = cust.copy()
    c["affinity_id"] = c["affinity"].map({k: i for i, k in enumerate(D.AFFINITY)})
    return np.column_stack([c[FEATURES].values, action])


@dataclass
class Policy:
    name: str
    action: np.ndarray                                      # N chosen action ids
    note: str = ""


@dataclass
class Study:
    world: D.World
    train: np.ndarray
    test: np.ndarray
    model: HistGradientBoostingClassifier
    p_hat: np.ndarray                                       # N x A
    v_hat: np.ndarray                                       # N x A predicted profit
    auc: float
    calib: pd.DataFrame
    policies: dict[str, Policy]
    shadow: dict[str, float]
    ope: pd.DataFrame                                       # policy x estimator: est, lo, hi, truth
    uplift: pd.DataFrame                                    # targeting curve
    segments: pd.DataFrame                                  # guardrail table
    mix: pd.DataFrame
    profit_sd: float
    gain: tuple                                             # DR gain of shipped over rules: est, lo, hi (paired)


def fit(world: D.World, seed: int = 7):
    n = len(world.customers)
    idx = np.random.default_rng(seed).permutation(n)
    train, test = np.sort(idx[: int(.5 * n)]), np.sort(idx[int(.5 * n):])
    log = world.log
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=60,
                                       categorical_features=[7, 9], l2_regularization=1.0, random_state=0)
    m.fit(_x(world.customers.iloc[train], log.action_id.values[train]), log.purchased.values[train])
    p_hat = np.column_stack([m.predict_proba(_x(world.customers, np.full(n, k)))[:, 1] for k in range(A)])
    auc = roc_auc_score(log.purchased.values[test], p_hat[test, log.action_id.values[test]])
    pl = p_hat[test, log.action_id.values[test]]
    bins = pd.qcut(pl, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": pl, "obs": log.purchased.values[test], "true": world.p_true[test, log.action_id.values[test]], "bin": bins}) \
        .groupby("bin").mean().reset_index()
    return train, test, m, p_hat, auc, calib


def value_of(p: np.ndarray, basket: np.ndarray) -> np.ndarray:
    return p * basket[:, None] * BASKET_MULT * (D.MARGIN - GIVE) - COST


def constrained(v: np.ndarray, capacity: dict[str, int] = CAPACITY, contact_cap: float = CONTACT_CAP, iters: int = 40):
    """Argmax with shadow prices: raise each scarce action's price until demand fits its capacity."""
    n = len(v)
    price = np.zeros(A)
    caps = {D.ACTIONS.index(k): c for k, c in capacity.items()}
    contact_price = 0.0
    for _ in range(6):                                       # a few rounds so the prices settle jointly
        for k, cap in caps.items():
            lo, hi = 0.0, float(v[:, k].max() - v.min())
            for _ in range(iters):
                mid = (lo + hi) / 2
                pr = price.copy(); pr[k] = mid
                adj = v - pr - contact_price * (np.arange(A) > 0)
                if (adj.argmax(1) == k).sum() > cap:
                    lo = mid
                else:
                    hi = mid
            price[k] = hi
        lo, hi = 0.0, float(np.abs(v).max())
        for _ in range(iters):
            mid = (lo + hi) / 2
            if ((v - price - mid * (np.arange(A) > 0)).argmax(1) > 0).mean() > contact_cap:
                lo = mid
            else:
                hi = mid
        contact_price = hi
    act = (v - price - contact_price * (np.arange(A) > 0)).argmax(1)
    return act, {**{D.ACTIONS[k]: float(price[k]) for k in caps}, "Contact slot": contact_price}


def _ope(world: D.World, test: np.ndarray, v_hat: np.ndarray, act: np.ndarray, rng: np.random.Generator) -> dict:
    log = world.log.iloc[test]
    a_log, r, prop = log.action_id.values, log.profit.values, log.propensity.values
    pi = act[test]
    match = (pi == a_log).astype(float)
    w = match / prop
    q_pi = v_hat[test, pi]
    q_log = v_hat[test, a_log]
    ips_i = w * r
    dr_i = q_pi + w * (r - q_log)
    dm_i = q_pi

    def boot(fn):
        k = len(test)
        vals = np.array([fn(rng.integers(0, k, k)) for _ in range(B)])
        return np.percentile(vals, [2.5, 97.5])

    est = {
        "Direct method": (dm_i.mean(), *boot(lambda s: dm_i[s].mean())),
        "IPS": (ips_i.mean(), *boot(lambda s: ips_i[s].mean())),
        "SNIPS": ((w * r).sum() / w.sum(), *boot(lambda s: (w[s] * r[s]).sum() / max(w[s].sum(), 1e-9))),
        "Doubly robust": (dr_i.mean(), *boot(lambda s: dr_i[s].mean())),
    }
    truth = world.value_true[test, pi].mean()
    return {"est": est, "truth": truth, "match": match.mean(), "ess": w.sum() ** 2 / max((w ** 2).sum(), 1e-9)}


def run(world: D.World | None = None) -> Study:
    world = world or D.simulate()
    rng = np.random.default_rng(11)
    train, test, m, p_hat, auc, calib = fit(world)
    basket = world.customers.avg_basket.values
    v_hat = value_of(p_hat, basket)
    n = len(basket)
    rules = world.log.rule_action.values
    blanket_k = int(v_hat[train].mean(0)[1:].argmax()) + 1
    greedy = v_hat.argmax(1)
    best, shadow = constrained(v_hat)
    pols = {
        "Current rules": Policy("Current rules", rules, "Lapsing → coupon, loyalty → points, engaged → email, else free shipping"),
        "One offer for all": Policy("One offer for all", np.full(n, blanket_k), f"Everyone gets {D.ACTIONS[blanket_k]}"),
        "Model, no limits": Policy("Model, no limits", greedy, "Best predicted profit, ignoring stock and contact limits"),
        "Model, within limits": Policy("Model, within limits", best, "Best predicted profit after shadow prices for stock, coupons and contacts"),
    }
    rows = []
    for name, pol in pols.items():
        o = _ope(world, test, v_hat, pol.action, rng)
        for est, (val, lo, hi) in o["est"].items():
            rows.append({"policy": name, "estimator": est, "est": val, "lo": lo, "hi": hi, "truth": o["truth"], "match": o["match"], "ess": o["ess"]})
    ope = pd.DataFrame(rows)

    # targeting curve on the test set: rank by predicted gain of chosen action over no contact
    gain_hat = v_hat[test, best[test]] - v_hat[test, 0]
    gain_true = world.value_true[test, best[test]] - world.value_true[test, 0]
    order = np.argsort(-gain_hat)
    rand = np.random.default_rng(3).permutation(len(test))
    up = pd.DataFrame({"share": np.arange(1, len(test) + 1) / len(test),
                       "model": np.cumsum(gain_true[order]) / len(test) * n,
                       "random": np.cumsum(gain_true[rand]) / len(test) * n})

    # guardrails: by segment, doubly robust estimate of (model - rules) on the held-out half, with a bootstrap interval.
    # A segment must gain at least MIN_GAIN per customer by that estimate to change; otherwise it stays on the current rules. Truth shown for grading.
    cust = world.customers
    seg = segment_of(cust)
    log = world.log
    a_log, r, prop = log.action_id.values, log.profit.values, log.propensity.values

    def dr_terms(act, idx):
        w = (act[idx] == a_log[idx]) / prop[idx]
        return v_hat[idx, act[idx]] + w * (r[idx] - v_hat[idx, a_log[idx]])

    segrows, shipped = [], best.copy()
    for s in SEGMENTS:
        mk = seg == s
        idx = test[mk[test]]
        d = dr_terms(best, idx) - dr_terms(rules, idx)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
        fallback = d.mean() < MIN_GAIN
        segrows.append({"segment": s, "customers": int(mk.sum()), "dr_diff": d.mean(), "dr_lo": np.percentile(bs, 2.5),
                        "dr_hi": np.percentile(bs, 97.5), "true_diff": world.value_true[mk, best[mk]].mean() - world.value_true[mk, rules[mk]].mean(),
                        "contact_rules": (rules[mk] > 0).mean(), "contact_model": (best[mk] > 0).mean(), "fallback": bool(fallback)})
    segments = pd.DataFrame(segrows)
    keep = np.isin(seg, segments.segment[segments.fallback].tolist())
    shipped[keep] = rules[keep]
    if keep.any():                                           # re-solve the rest with what the fallback segments use up
        caps = {k: c - int((rules[keep] == D.ACTIONS.index(k)).sum()) for k, c in CAPACITY.items()}
        share = (CONTACT_CAP * n - (rules[keep] > 0).sum()) / (~keep).sum()
        shipped[~keep], _ = constrained(v_hat[~keep], caps, share)
    pols["Shipped (with guardrails)"] = Policy("Shipped (with guardrails)", shipped, "Model within limits; segments the DR check flags stay on the current rules")
    o = _ope(world, test, v_hat, shipped, rng)
    for est, (val, lo, hi) in o["est"].items():
        ope.loc[len(ope)] = {"policy": "Shipped (with guardrails)", "estimator": est, "est": val, "lo": lo, "hi": hi, "truth": o["truth"], "match": o["match"], "ess": o["ess"]}
    d = dr_terms(shipped, test) - dr_terms(rules, test)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    gain = (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    mix = pd.DataFrame({name: np.bincount(p.action, minlength=A) for name, p in pols.items()}, index=D.ACTIONS)
    sd = log.profit.values[test][a_log[test] == rules[test]].std()
    return Study(world, train, test, m, p_hat, v_hat, auc, calib, pols, shadow, ope, up, segments, mix, sd, gain)


def sample_size(sd: float, lift: float, alpha_z: float = 1.96, power_z: float = .84) -> int:
    """Customers per arm to detect a profit lift with a two-sided 5% test at 80% power."""
    return int(np.ceil(2 * ((alpha_z + power_z) * sd / max(lift, 1e-9)) ** 2))
