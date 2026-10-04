"""Mortgage market intelligence engine.

A compact port of the scoring method from the Mortgage Market Intelligence
notebook, generalised for a public demo. It starts from one row per CBSA of
public metrics (Census population estimates, the HMDA loan application
register, the FHFA House Price Index) and rebuilds everything else:

* within-area-type percentile components
* five opportunity blocks (demographic, demand, capacity, collateral, openness)
* a two-basis mortgage risk composite (full FHFA basis vs HMDA-only basis)
* a strategic opportunity score with caller-supplied block weights
* evidence labels, signal bands and rule-based archetypes
* unsupervised archetypes (RobustScaler -> PCA -> KMeans, k by silhouette)
* anomaly scores (IsolationForest) and population-vs-mortgage divergence

Nothing here is specific to any lender. Lender names are never needed; market
structure is described only through concentration statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

SEED = 42
AREA_TYPES = ["Metropolitan", "Micropolitan"]

DEFAULT_WEIGHTS = {
    "demographic": 0.25,
    "demand": 0.25,
    "capacity": 0.20,
    "collateral": 0.20,
    "openness": 0.10,
}

BLOCKS = {
    "demographic": "demographic_opportunity_score",
    "demand": "mortgage_demand_score",
    "capacity": "borrower_capacity_score",
    "collateral": "collateral_momentum_score",
    "openness": "market_openness_score",
}

BLOCK_LABELS = {
    "demographic_opportunity_score": "Demographic strength",
    "mortgage_demand_score": "Mortgage demand",
    "borrower_capacity_score": "Borrower capacity",
    "collateral_momentum_score": "Collateral momentum",
    "market_openness_score": "Competitive openness",
    "affordability_pressure_score": "Affordability pressure",
}

# The public data contract: what the notebook export (or any other builder)
# must provide. Columns beyond the identifiers are optional; a missing input
# simply leaves its component unobserved, exactly as in the notebook.
IDENTITY = ["market_key", "market_name", "area_type", "primary_state"]
INPUTS = [
    # Census
    "population_latest", "population_cagr_pct", "domestic_migration_rate_pct",
    "international_migration_rate_pct", "natural_change_rate_pct", "demographic_opportunity_score",
    # HMDA
    "hmda_first_year", "hmda_latest_year", "purchase_applications_latest", "purchase_originations_latest",
    "purchase_application_cagr_pct", "purchase_origination_cagr_pct", "purchase_origination_rate_pct",
    "purchase_approval_rate_pct", "purchase_denial_rate_pct", "refinance_share_pct", "cashout_share_pct",
    "government_purchase_share_pct", "avg_loan_amount", "avg_property_value", "avg_applicant_income_k",
    "high_cltv_share_pct", "avg_dti_pct", "high_dti_share_pct", "avg_interest_rate_pct",
    "active_lenders", "lender_hhi", "top5_lender_share_pct", "applicant_income_growth_pct",
    # FHFA
    "fhfa_latest_year", "hpi_1y_pct", "hpi_3y_cagr_pct", "hpi_5y_cagr_pct", "hpi_volatility_pct",
    "hpi_drawdown_pct", "hpi_acceleration_pp",
]

COMPONENTS = [
    ("score_purchase_density", "purchase_originations_per_1000_residents", True),
    ("score_purchase_growth", "purchase_origination_cagr_pct", True),
    ("score_application_growth", "purchase_application_cagr_pct", True),
    ("score_origination_rate", "purchase_origination_rate_pct", True),
    ("score_approval", "purchase_approval_rate_pct", True),
    ("score_low_denial", "purchase_denial_rate_pct", False),
    ("score_low_high_dti", "high_dti_share_pct", False),
    ("score_low_high_cltv", "high_cltv_share_pct", False),
    ("score_income_to_property", "applicant_income_to_property_pct", True),
    ("score_hpi_1y", "hpi_1y_pct", True),
    ("score_hpi_3y", "hpi_3y_cagr_pct", True),
    ("score_hpi_stability", "hpi_volatility_pct", False),
    ("score_hpi_drawdown", "hpi_drawdown_pct", True),
    ("score_low_hhi", "lender_hhi", False),
    ("score_low_top5", "top5_lender_share_pct", False),
    ("score_lender_depth", "active_lenders", True),
    ("affordability_pressure_score", "affordability_pressure_raw", True),
]

BLOCK_RECIPES = {
    "mortgage_demand_score": {"score_purchase_density": 0.35, "score_purchase_growth": 0.30,
                              "score_application_growth": 0.15, "score_origination_rate": 0.20},
    "borrower_capacity_score": {"score_approval": 0.25, "score_low_denial": 0.25, "score_low_high_dti": 0.20,
                                "score_low_high_cltv": 0.15, "score_income_to_property": 0.15},
    "collateral_momentum_score": {"score_hpi_1y": 0.30, "score_hpi_3y": 0.35,
                                  "score_hpi_stability": 0.20, "score_hpi_drawdown": 0.15},
    "market_openness_score": {"score_low_hhi": 0.45, "score_low_top5": 0.35, "score_lender_depth": 0.20},
}

RISK_WEIGHTS = {
    "affordability_pressure_score": 0.28,
    "score_hpi_instability": 0.16,
    "score_hpi_decline": 0.11,
    "score_valuation_strain": 0.15,
    "score_high_dti": 0.10,
    "score_high_denial": 0.10,
    "score_weak_demand": 0.10,
}
RISK_COLLATERAL = ("affordability_pressure_score", "score_hpi_instability", "score_hpi_decline")
RISK_MIN_COMPONENTS, RISK_MIN_WEIGHT, RISK_MIN_GROUP = 2, 0.35, 10

CLUSTER_FEATURES = [
    "demographic_opportunity_score", "mortgage_demand_score", "borrower_capacity_score",
    "collateral_momentum_score", "market_openness_score", "affordability_pressure_score",
]
DIVERGENCE_PP = 30


# ---------------------------------------------------------------------------
# Helpers (same arithmetic as the notebook)
# ---------------------------------------------------------------------------
def safe_divide(num: pd.Series, den: pd.Series, mult: float = 1.0) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return (num / den.where(den != 0)) * mult


def weighted_score(frame: pd.DataFrame, components: Mapping[str, float], min_components: int = 2) -> pd.Series:
    cols = [c for c in components if c in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    weights = pd.Series({c: float(components[c]) for c in cols})
    num = values.mul(weights, axis=1).sum(axis=1, min_count=1)
    den = values.notna().mul(weights, axis=1).sum(axis=1)
    out = num / den.replace(0, np.nan)
    out[values.notna().sum(axis=1) < min_components] = np.nan
    return out


def within_type_pct(frame: pd.DataFrame, column: str, higher: bool = True) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    ranked = values.groupby(frame["area_type"]).rank(pct=True, method="average") * 100
    return ranked if higher else 100 - ranked


def within_group_pct(frame: pd.DataFrame, column: str, groups: list[str]) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.groupby([frame[g] for g in groups]).rank(pct=True, method="average") * 100


# ---------------------------------------------------------------------------
# Stage 1: everything that does not depend on the user's block weights
# ---------------------------------------------------------------------------
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Components, blocks, risk and rule inputs. Cached by the page."""
    m = raw.copy()
    for col in INPUTS:
        if col not in m.columns:
            m[col] = np.nan
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m["market_key"] = m["market_key"].astype(str).str.zfill(5)
    m["area_type"] = m["area_type"].where(m["area_type"].isin(AREA_TYPES), "Micropolitan")

    m["purchase_originations_per_1000_residents"] = safe_divide(
        m["purchase_originations_latest"], m["population_latest"], 1_000)
    m["applicant_income_to_property_pct"] = safe_divide(
        m["avg_applicant_income_k"] * 1_000, m["avg_property_value"], 100)
    m["property_value_to_income_multiple"] = safe_divide(
        m["avg_property_value"], m["avg_applicant_income_k"] * 1_000)
    # Price growth against income growth. Where income growth is unobserved, the
    # area-type median stands in, so the measure falls back to relative price pressure.
    income_growth = m["applicant_income_growth_pct"].fillna(
        m.groupby("area_type")["applicant_income_growth_pct"].transform("median")).fillna(0)
    m["affordability_pressure_raw"] = m["hpi_3y_cagr_pct"] * 3 - income_growth

    for name, column, higher in COMPONENTS:
        m[name] = within_type_pct(m, column, higher)

    m["score_hpi_instability"] = 100 - m["score_hpi_stability"]
    m["score_hpi_decline"] = 100 - m["score_hpi_drawdown"]
    m["score_high_denial"] = 100 - m["score_low_denial"]
    m["score_high_dti"] = 100 - m["score_low_high_dti"]
    m["score_valuation_strain"] = 100 - m["score_income_to_property"]

    for block, recipe in BLOCK_RECIPES.items():
        m[block] = weighted_score(m, recipe, min_components=2)
    m["score_weak_demand"] = 100 - m["mortgage_demand_score"]

    # Two-basis risk: markets are ranked only against others measured on the same basis.
    inputs = [c for c in RISK_WEIGHTS if c in m.columns]
    values = m[inputs].apply(pd.to_numeric, errors="coerce")
    weights = pd.Series({c: RISK_WEIGHTS[c] for c in inputs})
    observed = values.notna()
    seen = observed.mul(weights, axis=1).sum(axis=1)
    raw_risk = values.mul(weights, axis=1).sum(axis=1, min_count=1) / seen.replace(0, np.nan)
    publishable = observed.sum(axis=1).ge(RISK_MIN_COMPONENTS) & seen.ge(RISK_MIN_WEIGHT)
    raw_risk = raw_risk.where(publishable)
    collateral_seen = observed[[c for c in RISK_COLLATERAL if c in observed.columns]].any(axis=1)
    m["mortgage_risk_score_raw"] = raw_risk
    m["mortgage_risk_basis"] = np.select(
        [raw_risk.isna(), collateral_seen], ["Not available", "Full"], default="HMDA-only")
    basis_size = raw_risk.notna().groupby([m["area_type"], m["mortgage_risk_basis"]]).transform("sum")
    m["mortgage_risk_score"] = within_group_pct(m, "mortgage_risk_score_raw", ["area_type", "mortgage_risk_basis"]) \
        .where(basis_size.ge(RISK_MIN_GROUP), within_type_pct(m, "mortgage_risk_score_raw"))
    m.loc[m["mortgage_risk_basis"].eq("Not available"), "mortgage_risk_score"] = np.nan

    blocks = list(BLOCKS.values())
    m["score_blocks_available"] = m[blocks].notna().sum(axis=1)
    m["data_readiness"] = np.select(
        [m["score_blocks_available"].ge(5), m["score_blocks_available"].eq(4), m["score_blocks_available"].eq(3)],
        ["Complete", "Strong", "Partial"], default="Insufficient")
    m["market_label"] = m["market_name"].astype(str)
    return m


# ---------------------------------------------------------------------------
# Stage 2: weight-dependent scores, archetypes, divergence
# ---------------------------------------------------------------------------
def score(prepared: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total = sum(max(v, 0.0) for v in w.values()) or 1.0
    recipe = {BLOCKS[k]: max(v, 0.0) / total for k, v in w.items() if k in BLOCKS and v > 0}

    m = prepared.copy()
    m["strategic_mortgage_opportunity_score"] = weighted_score(m, recipe, min_components=min(3, len(recipe)))
    m["opportunity_percentile"] = m.groupby("area_type")["strategic_mortgage_opportunity_score"].rank(pct=True) * 100
    m["mortgage_signal"] = np.select(
        [m["opportunity_percentile"].ge(66.667), m["opportunity_percentile"].le(33.333)],
        ["High", "Low"], default="Medium")
    m.loc[m["strategic_mortgage_opportunity_score"].isna(), "mortgage_signal"] = "Not scored"

    g = m.groupby("area_type")
    q = {}
    for col, quant, key in [
        ("demographic_opportunity_score", .70, "demo_hi"), ("mortgage_demand_score", .70, "demand_hi"),
        ("mortgage_demand_score", .30, "demand_lo"), ("collateral_momentum_score", .70, "collateral_hi"),
        ("affordability_pressure_score", .70, "pressure_hi"), ("market_openness_score", .70, "open_hi"),
        ("mortgage_risk_score", .75, "risk_hi"), ("refinance_share_pct", .75, "refi_hi"),
    ]:
        q[key] = g[col].transform(lambda s, qq=quant: s.quantile(qq))
    m["mortgage_archetype"] = np.select([
        (m["demographic_opportunity_score"] >= q["demo_hi"]) & (m["mortgage_demand_score"] >= q["demand_hi"])
        & (m["collateral_momentum_score"] >= q["collateral_hi"]),
        (m["mortgage_demand_score"] >= q["demand_hi"]) & (m["affordability_pressure_score"] >= q["pressure_hi"]),
        (m["demographic_opportunity_score"] >= q["demo_hi"]) & (m["mortgage_demand_score"] <= q["demand_lo"]),
        (m["mortgage_demand_score"] >= q["demand_hi"]) & (m["market_openness_score"] >= q["open_hi"]),
        m["refinance_share_pct"] >= q["refi_hi"],
        m["mortgage_risk_score"] >= q["risk_hi"],
    ], [
        "Compound growth",
        "Demand under affordability pressure",
        "Population growth, lending lag",
        "Open, competitive growth",
        "Refinance-dependent",
        "Structural pressure",
    ], default="Stable or mixed")

    m["demographic_percentile"] = g["demographic_opportunity_score"].rank(pct=True) * 100
    m["mortgage_fundamentals_percentile"] = m.groupby("area_type")["strategic_mortgage_opportunity_score"].rank(pct=True) * 100
    m["population_mortgage_gap_pp"] = m["demographic_percentile"] - m["mortgage_fundamentals_percentile"]
    return m


@dataclass
class Clusters:
    assignments: pd.DataFrame  # market_key, cluster_id, cluster_label, margin, membership, anomaly_score
    profiles: pd.DataFrame     # area_type, cluster_id, label, markets, population, z_* per feature
    selection: pd.DataFrame    # area_type, k, silhouette, min_share, viable


def _cluster_label(profile: pd.Series) -> str:
    short = {
        "demographic_opportunity_score": "demographic strength", "mortgage_demand_score": "mortgage demand",
        "borrower_capacity_score": "borrower capacity", "collateral_momentum_score": "collateral momentum",
        "market_openness_score": "competitive openness", "affordability_pressure_score": "affordability pressure",
    }
    top = profile[CLUSTER_FEATURES].abs().sort_values(ascending=False).head(2).index
    return " + ".join(("High " if profile[f] >= 0 else "Low ") + short[f] for f in top)


def cluster(prepared: pd.DataFrame, k_min: int = 3, k_max: int = 6, min_markets: int = 35,
            min_share: float = 0.05, n_init: int = 20) -> Clusters:
    assignments, profiles, selection = [], [], []
    for area in AREA_TYPES:
        pool = prepared[prepared["area_type"].eq(area)].copy()
        # Markets missing up to two features are clustered on area-type medians for
        # those features (micros rarely carry a house-price index); sparser rows are left out.
        enough = pool[CLUSTER_FEATURES].notna().sum(axis=1) >= len(CLUSTER_FEATURES) - 2
        usable = pool[enough].copy()
        usable[CLUSTER_FEATURES] = usable[CLUSTER_FEATURES].fillna(usable[CLUSTER_FEATURES].median())
        usable = usable.dropna(subset=CLUSTER_FEATURES)
        if len(usable) < min_markets:
            continue
        scaled = RobustScaler().fit_transform(usable[CLUSTER_FEATURES])
        full = PCA(random_state=SEED).fit(scaled)
        n_comp = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.85) + 1)
        n_comp = max(2, min(n_comp, len(CLUSTER_FEATURES)))
        X = PCA(n_components=n_comp, random_state=SEED).fit_transform(scaled)

        best = None
        for k in range(k_min, k_max + 1):
            if k >= len(usable):
                continue
            km = KMeans(n_clusters=k, n_init=n_init, random_state=SEED)
            labels = km.fit_predict(X)
            shares = pd.Series(labels).value_counts(normalize=True)
            sil = float(silhouette_score(X, labels))
            viable = float(shares.min()) >= min_share
            selection.append({"area_type": area, "k": k, "silhouette": sil,
                              "min_share": float(shares.min()), "viable": viable})
            if viable and (best is None or sil > best[0]):
                best = (sil, k, km, labels)
        if best is None:
            continue
        _, k, km, labels = best

        d = np.sqrt(((X[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2))
        o = np.sort(d, axis=1)
        margin = np.where(o[:, 1] > 0, (o[:, 1] - o[:, 0]) / o[:, 1], 0)

        iso = IsolationForest(contamination="auto", random_state=SEED, n_estimators=300)
        anomaly = pd.Series(-iso.fit(X).score_samples(X)).rank(pct=True).to_numpy() * 100

        usable["cluster_id"] = labels
        z = pd.DataFrame(scaled, columns=CLUSTER_FEATURES, index=usable.index)
        z["cluster_id"] = labels
        prof = z.groupby("cluster_id")[CLUSTER_FEATURES].mean()
        names = {int(cid): _cluster_label(row) for cid, row in prof.iterrows()}

        assignments.append(pd.DataFrame({
            "market_key": usable["market_key"].to_numpy(),
            "cluster_id": [f"{area[:5]}-{int(c) + 1}" for c in labels],
            "cluster_label": [names[int(c)] for c in labels],
            "cluster_margin": margin,
            "cluster_membership": np.where(margin >= 0.20, "Core", "Boundary"),
            "anomaly_score": anomaly,
        }))
        for cid, row in prof.iterrows():
            members = usable[usable["cluster_id"].eq(cid)]
            rec = {"area_type": area, "cluster_id": f"{area[:5]}-{int(cid) + 1}", "cluster_label": names[int(cid)],
                   "markets": int(len(members)), "population": float(members["population_latest"].sum())}
            rec.update({c: float(row[c]) for c in CLUSTER_FEATURES})
            profiles.append(rec)

    empty = pd.DataFrame(columns=["market_key", "cluster_id", "cluster_label", "cluster_margin",
                                  "cluster_membership", "anomaly_score"])
    return Clusters(
        assignments=pd.concat(assignments, ignore_index=True) if assignments else empty,
        profiles=pd.DataFrame(profiles),
        selection=pd.DataFrame(selection),
    )


# ---------------------------------------------------------------------------
# Plain-language reads for the market brief
# ---------------------------------------------------------------------------
def driver_read(row: pd.Series) -> str:
    observed = [(BLOCK_LABELS[c], float(row[c])) for c in BLOCKS.values() if pd.notna(row.get(c))]
    missing = [BLOCK_LABELS[c].lower() for c in BLOCKS.values() if pd.isna(row.get(c))]
    if not observed:
        return "No strategic blocks are observed for this market."
    observed.sort(key=lambda x: x[1], reverse=True)
    text = (f"Strongest support comes from {observed[0][0].lower()} ({observed[0][1]:.0f})"
            + (f" and {observed[1][0].lower()} ({observed[1][1]:.0f})" if len(observed) > 1 else "")
            + f". The weakest observed block is {observed[-1][0].lower()} ({observed[-1][1]:.0f}).")
    if missing:
        text += f" Not observed here: {', '.join(missing)}."
    return text


def divergence_read(gap: float) -> str:
    if pd.isna(gap):
        return "Population-to-mortgage divergence is not available."
    if gap >= DIVERGENCE_PP:
        return ("Demographic strength runs well ahead of mortgage fundamentals. That is a possible "
                "unconverted opportunity to investigate, not proof of one.")
    if gap <= -DIVERGENCE_PP:
        return ("Mortgage fundamentals run well ahead of the demographic backdrop. Lending strength may be "
                "outpacing population; test how durable it is.")
    return "Demographic and mortgage signals are broadly aligned."


def evidence_read(row: pd.Series) -> str:
    base = {
        "Complete": "All five blocks are observed.",
        "Strong": "Four of five blocks are observed.",
        "Partial": "Only three blocks are observed; read the ranking as directional.",
    }.get(str(row.get("data_readiness")), "Evidence is limited; avoid strong ranking claims.")
    basis = str(row.get("mortgage_risk_basis"))
    if basis == "Full":
        return base + " Risk uses the full basis, including house-price inputs."
    if basis == "HMDA-only":
        return base + " Risk uses the HMDA-only basis and is ranked only against other HMDA-only markets."
    return base + " Mortgage risk is not available."


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIVE_FILE = DATA_DIR / "markets.csv"
META_FILE = DATA_DIR / "markets_meta.json"


def load_raw() -> tuple[pd.DataFrame, bool]:
    """Return (frame, is_live). Falls back to the labeled synthetic sample."""
    if LIVE_FILE.exists():
        frame = pd.read_csv(LIVE_FILE, dtype={"market_key": str})
        if "market_name" not in frame.columns and "NAME" in frame.columns:
            frame = frame.rename(columns={"NAME": "market_name"})
        return frame, True
    from lab.market_sample import sample_markets
    return sample_markets(), False


YEARS_FILE = DATA_DIR / "market_years.csv"
LENDERS_FILE = DATA_DIR / "lenders.csv"
HISTORY_DIR = DATA_DIR / "history"


def load_years() -> pd.DataFrame | None:
    if not YEARS_FILE.exists():
        return None
    return pd.read_csv(YEARS_FILE, dtype={"market_key": str})


def load_lenders() -> pd.DataFrame | None:
    if not LENDERS_FILE.exists():
        return None
    return pd.read_csv(LENDERS_FILE, dtype={"market_key": str, "lei": str})


def load_history() -> list[tuple[str, pd.DataFrame]]:
    """Monthly snapshots, oldest first, as (YYYY-MM, frame)."""
    if not HISTORY_DIR.exists():
        return []
    out = []
    for f in sorted(HISTORY_DIR.glob("markets_*.csv")):
        out.append((f.stem.split("_", 1)[1], pd.read_csv(f, dtype={"market_key": str})))
    return out
