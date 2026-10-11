"""Mortgage product strategy engine.

Turns the public market table (see lab.market_engine) into a decision problem:
which market is the strongest place for a hypothetical lender to expand one of
three product lines, under adjustable strategic weights and hard screens.

Everything here is a transparent calculation on public, aggregated data plus
clearly labelled illustrative assumptions. Nothing models a real lender's
eligibility rules, pricing or results.

Evidence classes used throughout the page:
  observed     public figures, as published (Census, HMDA, FHFA)
  modeled      scores and estimates derived from observed figures
  illustrative hypothetical assumptions chosen to show a method
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab import market_engine as me

PRODUCTS = {
    "purchase": "Conventional purchase",
    "refi": "Refinance, including cash-out",
    "gov": "Government-backed purchase",
}

# key -> (label, one-line meaning)
DIMENSIONS = {
    "demand": ("Demand", "How much purchase or refinance activity the market shows, and how fast it is growing."),
    "access": ("Access", "How well the product fits the borrowers who are there: capacity, equity or affordability need."),
    "risk": ("Credit risk", "Lower is better. Affordability pressure, price instability, valuation strain, denial and DTI."),
    "economics": ("Economics", "Size of the observed lending pool for the product, scaled by loan size."),
    "competition": ("Competition", "How open the market is: lender concentration and lender depth."),
}
DIM_KEYS = list(DIMENSIONS)
DEFAULT_WEIGHTS = {"demand": 30, "access": 20, "risk": 20, "economics": 20, "competition": 10}

# What each dimension is built from, for the Frame step and the methodology note.
DIM_DETAIL = {
    "demand": {
        "data": "HMDA purchase originations per 1,000 residents, origination and application growth, origination rate. Refinance and government-backed products blend in the market's refinance or government-backed share.",
        "approach": "Percentile scores within metro or micro areas, combined with fixed block weights.",
        "impact": "A high score means there is activity to compete for. It does not mean unmet demand: low lending can reflect low demand, tight credit or strong incumbents.",
    },
    "access": {
        "data": "HMDA approval and denial rates, high-DTI and high-CLTV shares, applicant income against property value, and the area's affordability pressure.",
        "approach": "Purchase uses borrower capacity. Refinance adds the share of loans with low combined loan-to-value, a proxy for equity. Government-backed leans on affordability pressure, where those programs matter most.",
        "impact": "Separates markets where the product fits the people there from markets that are only large.",
    },
    "risk": {
        "data": "FHFA house-price momentum, stability and drawdown, affordability pressure, valuation strain, HMDA denial and DTI.",
        "approach": "A two-basis composite (full FHFA basis or HMDA-only basis), ranked only against markets measured the same way. Shown here as 100 minus the risk percentile.",
        "impact": "Keeps a growth story from hiding a fragile market. Hard risk limits are applied separately as screens.",
    },
    "economics": {
        "data": "Observed purchase originations and average loan amount. Refinance and government-backed pools are scaled from those using each market's refinance or government-backed share.",
        "approach": "Percentile of the estimated lending pool in dollars, blended with average loan size. This is relative market size, not profit.",
        "impact": "A bigger pool can carry a launch cost. The Economics step tests that with an illustrative model.",
    },
    "competition": {
        "data": "HMDA lender concentration (HHI), top-five lender share and number of active lenders.",
        "approach": "Percentile of openness within metro or micro areas.",
        "impact": "Stands in for how hard it would be to win share. It is a proxy for execution difficulty, not a measure of any lender's capability.",
    },
}


# ---------------------------------------------------------------------------
# Dimension scores
# ---------------------------------------------------------------------------
def _pct(frame: pd.DataFrame, column: str, higher: bool = True) -> pd.Series:
    return me.within_type_pct(frame, column, higher)


def _blend(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    """Weighted mean that tolerates a missing part by renormalising over the rest."""
    values = pd.concat([p for p, _ in parts], axis=1)
    weights = np.array([w for _, w in parts], dtype=float)
    observed = values.notna().to_numpy()
    num = np.nansum(values.to_numpy() * weights, axis=1)
    den = (observed * weights).sum(axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return pd.Series(out, index=values.index)


def lending_pool(frame: pd.DataFrame, product: str) -> pd.Series:
    """Estimated annual lending pool in dollars for the product, scaled from observed purchase originations."""
    orig = frame["purchase_originations_latest"]
    loan = frame["avg_loan_amount"]
    if product == "refi":
        share = frame["refinance_share_pct"].clip(upper=90)
        orig = orig * share / (100 - share)
    elif product == "gov":
        orig = orig * frame["government_purchase_share_pct"] / 100
    return orig * loan


def market_volume(row: pd.Series, product: str) -> float:
    """Estimated annual funded loans in the whole market for the product."""
    orig = float(row["purchase_originations_latest"])
    if product == "refi":
        share = min(float(row["refinance_share_pct"]), 90.0)
        return orig * share / (100 - share)
    if product == "gov":
        return orig * float(row["government_purchase_share_pct"]) / 100
    return orig


def dimension_scores(prepared: pd.DataFrame, product: str) -> pd.DataFrame:
    """Five 0-100 scores per market for one product. Higher is better on every dimension."""
    f = prepared
    demand = f["mortgage_demand_score"]
    capacity = f["borrower_capacity_score"]
    if product == "refi":
        demand = _blend([(_pct(f, "refinance_share_pct"), 0.6), (demand, 0.4)])
        access = _blend([(capacity, 0.5), (f["score_low_high_cltv"], 0.5)])
    elif product == "gov":
        demand = _blend([(demand, 0.6), (_pct(f, "government_purchase_share_pct"), 0.4)])
        access = _blend([(f["affordability_pressure_score"], 0.6), (capacity, 0.4)])
    else:
        access = capacity
    pool = lending_pool(f, product)
    econ_frame = f.assign(_pool=pool)
    economics = _blend([(_pct(econ_frame, "_pool"), 0.7), (_pct(f, "avg_loan_amount"), 0.3)])
    out = pd.DataFrame({
        "demand": demand,
        "access": access,
        "risk": 100 - f["mortgage_risk_score"],
        "economics": economics,
        "competition": f["market_openness_score"],
    })
    out.index = f["market_key"].to_numpy()
    return out.clip(0, 100)


# ---------------------------------------------------------------------------
# Screens (hard constraints, kept apart from preference weights)
# ---------------------------------------------------------------------------
READINESS_ORDER = ["Insufficient", "Partial", "Strong", "Complete"]


def screen(prepared: pd.DataFrame, scores: pd.DataFrame, *, area_type: str, states: list[str], min_population: int,
           min_readiness: str, max_risk_pct: int) -> pd.DataFrame:
    """One row per market with a pass flag and the reason it was screened out, if any."""
    f = prepared.set_index("market_key", drop=False)
    f = f.loc[scores.index]
    reasons = pd.Series("", index=f.index, dtype=object)

    def mark(mask: pd.Series, text: str) -> None:
        nonlocal reasons
        mask = mask.fillna(True)
        reasons = reasons.where(~(mask & (reasons == "")), text)

    mark(f["area_type"] != area_type, "Other market type")
    if states:
        mark(~f["primary_state"].isin(states), "Outside selected states")
    mark(f["population_latest"].fillna(0) < min_population, "Below population floor")
    floor = READINESS_ORDER.index(min_readiness)
    mark(f["data_readiness"].map(READINESS_ORDER.index) < floor, "Evidence below required level")
    mark(f["mortgage_risk_score"] > max_risk_pct, "Above risk limit")
    mark(scores.notna().sum(axis=1) < 4, "Too few measured dimensions")
    return pd.DataFrame({"passes": reasons == "", "reason": reasons}, index=f.index)


# ---------------------------------------------------------------------------
# Composite, stress test, explanation
# ---------------------------------------------------------------------------
def normalise(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(v, 0) for v in weights.values()))
    if total <= 0:
        weights = DEFAULT_WEIGHTS
        total = float(sum(weights.values()))
    return {k: max(weights.get(k, 0), 0) / total for k in DIM_KEYS}


def composite(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = normalise(weights)
    vec = np.array([w[k] for k in DIM_KEYS])
    vals = scores[DIM_KEYS].to_numpy(dtype=float)
    observed = ~np.isnan(vals)
    num = np.nansum(vals * vec, axis=1)
    den = (observed * vec).sum(axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return pd.Series(out, index=scores.index)


def contributions(scores: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Points each dimension adds to the composite (weight x score), renormalised for missing scores."""
    w = normalise(weights)
    vec = np.array([w[k] for k in DIM_KEYS])
    vals = scores[DIM_KEYS].to_numpy(dtype=float)
    observed = ~np.isnan(vals)
    den = (observed * vec).sum(axis=1, keepdims=True)
    parts = np.nan_to_num(vals) * vec / np.where(den > 0, den, 1)
    return pd.DataFrame(parts, index=scores.index, columns=DIM_KEYS)


@dataclass
class Stress:
    top1: pd.Series
    top3: pd.Series
    top10: pd.Series
    median_rank: pd.Series
    draws: int


def stress_test(scores: pd.DataFrame, weights: dict[str, float], draws: int = 600, concentration: float = 25.0,
                seed: int = 7) -> Stress:
    """Re-rank under many nearby weight sets and report how often each market stays on top.

    Weights are drawn from a Dirichlet centred on the chosen weights. A market that
    holds a top-three place across most draws does not depend on the exact weights;
    one that appears only at the chosen weights is fragile. This tests weight
    sensitivity only, not uncertainty in the underlying data.
    """
    rng = np.random.default_rng(seed)
    base = np.array([normalise(weights)[k] for k in DIM_KEYS])
    alpha = np.maximum(base * concentration, 0.35)
    sample = rng.dirichlet(alpha, size=draws)  # draws x 5
    vals = scores[DIM_KEYS].to_numpy(dtype=float)
    observed = ~np.isnan(vals)
    filled = np.nan_to_num(vals)
    totals = filled @ sample.T  # markets x draws
    den = observed.astype(float) @ sample.T
    comp = np.where(den > 0, totals / np.where(den > 0, den, 1), -np.inf)
    ranks = (-comp).argsort(axis=0).argsort(axis=0) + 1  # 1 = best
    idx = scores.index
    return Stress(
        top1=pd.Series((ranks == 1).mean(axis=1), index=idx),
        top3=pd.Series((ranks <= 3).mean(axis=1), index=idx),
        top10=pd.Series((ranks <= 10).mean(axis=1), index=idx),
        median_rank=pd.Series(np.median(ranks, axis=1), index=idx),
        draws=draws,
    )


def flip_point(scores: pd.DataFrame, weights: dict[str, float], leader: str, runner: str) -> dict | None:
    """The smallest single-weight move that would put the runner-up ahead of the leader.

    For each dimension d, the weight on d is raised or lowered to t while the others keep
    their relative proportions. The gap between the two markets is linear in t, so the
    tie point has a closed form. Returns the move needing the least change, or None.
    """
    w = normalise(weights)
    a, b = scores.loc[leader, DIM_KEYS].astype(float), scores.loc[runner, DIM_KEYS].astype(float)
    if a.isna().any() or b.isna().any():
        return None
    best = None
    for d in DIM_KEYS:
        rest = 1 - w[d]
        if rest <= 1e-9:
            continue
        a_rest = sum(w[k] * a[k] for k in DIM_KEYS if k != d) / rest
        b_rest = sum(w[k] * b[k] for k in DIM_KEYS if k != d) / rest
        # gap(t) = (b - a) composite = t*(b_d - a_d) + (1 - t)*(b_rest - a_rest)
        slope = (b[d] - a[d]) - (b_rest - a_rest)
        if abs(slope) < 1e-9:
            continue
        t_star = -(b_rest - a_rest) / slope
        if not 0 <= t_star <= 1:
            continue
        move = abs(t_star - w[d])
        if best is None or move < best["move"]:
            best = {"dimension": d, "from": w[d], "to": t_star, "move": move}
    return best


# ---------------------------------------------------------------------------
# Illustrative economics
# ---------------------------------------------------------------------------
# All values are hypothetical and exist only to show the method. They are not quotes,
# benchmarks or any lender's actual pricing or cost.
ECON_DEFAULTS = {
    "purchase": {"revenue_bps": 340, "cost_per_loan": 7000, "reserve_bps": 10, "launch_cost": 300_000, "capture_pct": 2.0},
    "refi": {"revenue_bps": 280, "cost_per_loan": 5500, "reserve_bps": 8, "launch_cost": 250_000, "capture_pct": 2.0},
    "gov": {"revenue_bps": 350, "cost_per_loan": 7500, "reserve_bps": 15, "launch_cost": 350_000, "capture_pct": 1.5},
}
RAMP = [0.5, 1.0, 1.0, 1.0, 1.0]
SCENARIOS = {
    "Downside": {"volume": 0.70, "revenue": 0.90, "cost": 1.15},
    "Base": {"volume": 1.00, "revenue": 1.00, "cost": 1.00},
    "Upside": {"volume": 1.25, "revenue": 1.10, "cost": 0.95},
}


def economics(market_loans: float, avg_loan: float, capture_pct: float, revenue_bps: float, cost_per_loan: float,
              reserve_bps: float, launch_cost: float, horizon: int, scenario: str = "Base") -> dict:
    s = SCENARIOS[scenario]
    ramp = RAMP[:horizon]
    loans_full = market_loans * capture_pct / 100 * s["volume"]
    margin = avg_loan * (revenue_bps * s["revenue"] - reserve_bps) / 10_000 - cost_per_loan * s["cost"]
    yearly = [loans_full * r * margin for r in ramp]
    cumulative = np.cumsum([-launch_cost] + yearly)
    loan_years = loans_full * sum(ramp)
    breakeven_loans = launch_cost / margin if margin > 0 else float("inf")
    breakeven_share = (breakeven_loans / (market_loans * s["volume"] * sum(ramp)) * 100
                       if margin > 0 and market_loans > 0 else float("inf"))
    return {
        "loans_full": loans_full,
        "margin": margin,
        "yearly": yearly,
        "cumulative": cumulative.tolist(),
        "net": float(cumulative[-1]),
        "breakeven_loans": breakeven_loans,
        "breakeven_share": breakeven_share,
        "loan_years": loan_years,
    }


def tornado(base: dict, swing: float = 0.20) -> pd.DataFrame:
    """Change in net result when each input moves up or down by `swing`, one at a time."""
    keys = [
        ("market_loans", "Market volume"),
        ("capture_pct", "Share captured"),
        ("avg_loan", "Average loan"),
        ("revenue_bps", "Revenue per loan"),
        ("cost_per_loan", "Cost per loan"),
        ("launch_cost", "Launch cost"),
    ]
    center = economics(**base)["net"]
    rows = []
    for key, label in keys:
        lo = economics(**{**base, key: base[key] * (1 - swing)})["net"] - center
        hi = economics(**{**base, key: base[key] * (1 + swing)})["net"] - center
        rows.append({"input": label, "down": lo, "up": hi, "span": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("span")


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------
CLASSES = [
    ("Prioritize for diligence",
     "Top three at the chosen weights, a top-three place in at least 60% of nearby weight sets, and a positive downside case."),
    ("Pilot candidate",
     "Top ten at the chosen weights and in the top ten in at least half of nearby weight sets."),
    ("Optimize current offering",
     "Does not rank high enough to expand into, but the illustrative economics stay positive, so existing activity is worth tuning."),
    ("Monitor",
     "Economics do not clear at the base case, break-even needs more than a 10% share, or risk sits in the top quarter."),
    ("Defer pending evidence",
     "Public data for this market is only partial or insufficient, so no conclusion is drawn."),
]


def classify(*, readiness: str, rank: int, top3: float, top10: float, base_net: float, downside_net: float,
             breakeven_share: float, risk_pct: float) -> tuple[str, str]:
    if readiness not in ("Complete", "Strong"):
        return CLASSES[4][0], "Public data for this market is only partial, so the analysis stops short of a conclusion."
    if base_net <= 0 or breakeven_share > 10:
        return CLASSES[3][0], "At the base case the illustrative economics do not clear, or break-even needs an unrealistic share."
    if risk_pct >= 75:
        return CLASSES[3][0], "Risk sits in the top quarter of comparable markets."
    if rank <= 3 and top3 >= 0.60 and downside_net > 0:
        return CLASSES[0][0], "Ranks in the top three, holds that place across most weight sets, and stays positive in the downside case."
    if rank <= 10 and top10 >= 0.50:
        return CLASSES[1][0], "Ranks in the top ten and stays there across at least half of nearby weight sets."
    return CLASSES[2][0], "Not a leading expansion market, but the illustrative economics stay positive."
