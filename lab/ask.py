"""Grounded answers about one market.

With an Anthropic API key in Streamlit secrets (ANTHROPIC_API_KEY), free-text
questions are answered by Claude using only the market's record as context.
Without a key, a fixed set of questions is answered deterministically from the
same record, so the feature always works and never invents numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from lab import market_engine as me

QUESTIONS = [
    "Why does it rank where it does?",
    "What are the main risks?",
    "How does it compare with its peers?",
    "Who leads lending here, and is it concentrated?",
    "What is changing over time?",
]

SYSTEM = (
    "You are a market analyst. Answer ONLY from the JSON context about one U.S. metro or micro area. "
    "Cite specific numbers from the context. If the context does not contain the answer, say so plainly. "
    "Scores are within-type percentiles (0-100, higher is stronger; for risk, higher is riskier). "
    "Never give lending, investment or legal advice. Answer in at most 140 words, plain prose, no headings."
)


def llm_available() -> bool:
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:  # noqa: BLE001  (no secrets file)
        return False


def context(row: pd.Series, peers: pd.DataFrame, years: pd.DataFrame | None, lenders: pd.DataFrame | None) -> dict:
    def val(c):
        v = row.get(c)
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else str(v))

    fields = ["market_name", "area_type", "primary_state", "population_latest", "population_cagr_pct",
              "strategic_mortgage_opportunity_score", "mortgage_signal", "mortgage_archetype", "cluster_label",
              "demographic_opportunity_score", "mortgage_demand_score", "borrower_capacity_score",
              "collateral_momentum_score", "market_openness_score", "mortgage_risk_score", "mortgage_risk_basis",
              "affordability_pressure_score", "purchase_originations_latest", "purchase_origination_cagr_pct",
              "purchase_origination_rate_pct", "purchase_denial_rate_pct", "refinance_share_pct",
              "government_purchase_share_pct", "avg_loan_amount", "hpi_1y_pct", "hpi_3y_cagr_pct",
              "hpi_volatility_pct", "active_lenders", "lender_hhi", "top5_lender_share_pct",
              "population_mortgage_gap_pp", "data_readiness"]
    ctx = {f: val(f) for f in fields if f in row.index}
    ctx["peer_medians"] = {f: round(float(peers[f].median()), 2) for f in fields
                           if f in peers.columns and pd.api.types.is_numeric_dtype(peers[f]) and peers[f].notna().any()}
    ctx["rank_in_view"] = int((peers["strategic_mortgage_opportunity_score"] > row["strategic_mortgage_opportunity_score"]).sum() + 1)
    ctx["markets_in_view"] = int(len(peers))
    if years is not None and not years.empty:
        y = years[years["market_key"].eq(row["market_key"])].sort_values("year")
        ctx["yearly"] = y[["year", "purchase_originations", "purchase_denial_rate_pct", "refinance_share_pct"]].round(1).to_dict("records")
    if lenders is not None and not lenders.empty:
        lt = lenders[lenders["market_key"].eq(row["market_key"])].nsmallest(5, "rank")
        ctx["top_lenders"] = [{"lender": r.lender, "share_pct": round(r.share_pct, 1),
                               "share_change_pp": None if pd.isna(r.share_first_pct) else round(r.share_pct - r.share_first_pct, 1)}
                              for r in lt.itertuples()]
    return ctx


def ask_llm(question: str, ctx: dict) -> str:
    import anthropic  # imported lazily so the app runs without the package configured

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    model = st.secrets.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    msg = client.messages.create(
        model=model, max_tokens=400, system=SYSTEM,
        messages=[{"role": "user", "content": f"Context:\n{json.dumps(ctx)}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def _p(v, digits=0, suffix=""):
    return "n/a" if v is None else f"{v:,.{digits}f}{suffix}"


def ask_rules(question: str, row: pd.Series, ctx: dict) -> str:
    med = ctx.get("peer_medians", {})
    name = ctx.get("market_name", "This market")
    if question == QUESTIONS[0]:
        return (f"{name} ranks {ctx['rank_in_view']} of {ctx['markets_in_view']} markets in view with an opportunity score "
                f"of {_p(ctx.get('strategic_mortgage_opportunity_score'))}. " + me.driver_read(row))
    if question == QUESTIONS[1]:
        parts = [f"Mortgage risk is {_p(ctx.get('mortgage_risk_score'))} on the {ctx.get('mortgage_risk_basis', 'n/a')} basis "
                 f"(peer median {_p(med.get('mortgage_risk_score'))})."]
        if ctx.get("affordability_pressure_score") is not None:
            parts.append(f"Affordability pressure sits at the {_p(ctx['affordability_pressure_score'])}th percentile.")
        if ctx.get("purchase_denial_rate_pct") is not None:
            parts.append(f"The purchase denial rate is {_p(ctx['purchase_denial_rate_pct'], 1, '%')} against a peer median of "
                         f"{_p(med.get('purchase_denial_rate_pct'), 1, '%')}.")
        if ctx.get("hpi_volatility_pct") is not None:
            parts.append(f"House-price volatility is {_p(ctx['hpi_volatility_pct'], 1, '%')} a year.")
        return " ".join(parts)
    if question == QUESTIONS[2]:
        lines = []
        for f, label, d in (("purchase_origination_cagr_pct", "purchase-origination growth", 1),
                            ("purchase_origination_rate_pct", "origination rate", 1),
                            ("government_purchase_share_pct", "government share of purchases", 1),
                            ("hpi_3y_cagr_pct", "three-year house-price growth", 1),
                            ("active_lenders", "active lenders", 0)):
            if ctx.get(f) is not None and med.get(f) is not None:
                direction = "above" if ctx[f] > med[f] else "below"
                lines.append(f"{label} of {_p(ctx[f], d)} is {direction} the peer median of {_p(med[f], d)}")
        return f"Against other {str(ctx.get('area_type', '')).lower()} markets: " + "; ".join(lines) + "."
    if question == QUESTIONS[3]:
        top = ctx.get("top_lenders") or []
        if not top:
            return "Lender-level data is not available for this market in the current build."
        hhi = ctx.get("lender_hhi")
        conc = "unconcentrated" if hhi is not None and hhi < 1000 else "moderately concentrated" if hhi is not None and hhi < 1800 else "concentrated"
        names = ", ".join(f"{l['lender'].title()} ({l['share_pct']:.1f}%)" for l in top[:3])
        return (f"The largest lenders by HMDA records are {names}. With {_p(ctx.get('active_lenders'))} active lenders and an "
                f"HHI of {_p(hhi)}, the market is {conc}; the top five hold {_p(ctx.get('top5_lender_share_pct'), 1, '%')}.")
    yearly = ctx.get("yearly") or []
    if len(yearly) >= 2:
        a, b = yearly[0], yearly[-1]
        chg = (b["purchase_originations"] / a["purchase_originations"] - 1) * 100 if a["purchase_originations"] else float("nan")
        return (f"Purchase originations went from {a['purchase_originations']:,.0f} in {a['year']} to "
                f"{b['purchase_originations']:,.0f} in {b['year']} ({chg:+.0f}%). The denial rate moved from "
                f"{a['purchase_denial_rate_pct']:.1f}% to {b['purchase_denial_rate_pct']:.1f}%, and the refinance share "
                f"from {a['refinance_share_pct']:.1f}% to {b['refinance_share_pct']:.1f}%.")
    return "Year-by-year data is not available for this market in the current build."
