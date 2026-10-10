"""Shared rendering for the model bench (used by the lending and subscriber labs)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import bench as B
from lab import theme as t

ALGO_COLORS = dict(zip(B.ALGOS, ["#5CC3EE", "#CBFA7C", "#D9C24A", "#EF936F", "#8FA2F0"]))


def render(bench: B.Bench, *, key: str, regulated_default: bool, value_label: str, decision: str,
           regulated_help: str) -> str:
    """Draw the bench and return the champion's name."""
    tb = bench.table.copy()
    regulated = st.toggle("Regulated decision: every outcome must be explained to the customer", value=regulated_default,
                          key=f"{key}_reg", help=regulated_help)
    champ, why = B.recommend(tb, regulated)
    c = tb.set_index("algorithm").loc[champ]
    best = tb.loc[tb["value"].idxmax()]
    t.tiles([
        {"label": "Champion", "value": champ, "accent": True, "note": "regulated rules" if regulated else "unregulated rules"},
        {"label": "Champion AUC", "value": f"{c['auc']:.3f}", "note": f"best on the bench {tb['auc'].max():.3f}"},
        {"label": value_label, "value": t.money(c["value"]), "accent": True,
         "note": "the best on the bench" if champ == best["algorithm"] else f"{t.money(best['value'] - c['value'])} below {best['algorithm']}"},
        {"label": "Scoring cost", "value": f"{c['score_ms']:.1f} ms", "note": "per 1,000 decisions on this server"},
    ])
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        order = tb.sort_values("value")
        colors = [ALGO_COLORS[a] if (a == champ or not regulated or e) else "#3A4F47" for a, e in zip(order["algorithm"], order["exact"])]
        fig = go.Figure(go.Bar(y=order["algorithm"], x=order["value"], orientation="h", marker=dict(color=colors, cornerradius=3),
                               text=[("★ " if a == champ else "") + t.money(v) for a, v in zip(order["algorithm"], order["value"])],
                               textposition="outside", cliponaxis=False,
                               customdata=order[["auc", "reasons"]], hovertemplate="%{y}: %{x:$,.0f}<br>AUC %{customdata[0]:.3f} · reasons: %{customdata[1]}<extra></extra>"))
        fig.update_layout(title=value_label + (" · grey: no exact reasons" if regulated else ""),
                          xaxis=dict(tickprefix="$", tickformat="~s", range=[0, tb["value"].max() * 1.25]), margin=dict(l=8, t=70))
        t.chart(fig, height=300)
    with right:
        fig2 = go.Figure()
        top = 0
        for a in B.ALGOS:
            cal = B.calibration_curve(bench.y, bench.preds[a])
            top = max(top, cal["actual"].max(), cal["pred"].max())
            fig2.add_trace(go.Scatter(x=cal["pred"] * 100, y=cal["actual"] * 100, name=a, mode="lines+markers",
                                      line=dict(color=ALGO_COLORS[a], width=2.6 if a == champ else 1.6), marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=[0, top * 100], y=[0, top * 100], mode="lines", line=dict(color=t.INK, width=1, dash="dot"),
                                  name="Perfect calibration", hoverinfo="skip"))
        fig2.update_layout(title="Calibration: predicted vs actual rate, by decile", xaxis_title="Predicted (%)", yaxis_title="Actual (%)",
                           legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=11)), margin=dict(t=120))
        t.chart(fig2, height=400)
    show = pd.DataFrame({
        "": tb["algorithm"].eq(champ).map({True: "★", False: ""}),
        "Algorithm": tb["algorithm"],
        "AUC": tb["auc"].map(lambda v: f"{v:.3f}"),
        "Calib. error": tb["ece"].map(lambda v: f"{v * 100:.1f} pts"),
        "Top 10%": tb["top_decile_capture"].map(lambda v: f"{v:.0%}"),
        value_label: tb["value"].map(t.money),
        "Reasons": tb["reasons"],
        "ms / 1,000": tb["score_ms"].map(lambda v: f"{v:.1f}"),
    })
    if regulated:
        show.insert(7, "Eligible", ["Yes" if e else "No" for e in tb["exact"]])
    if "air" in tb:
        show["AIR"] = tb["air"].map(lambda v: f"{v:.2f}")
    st.dataframe(show, hide_index=True, width="stretch")
    lin = tb.set_index("algorithm").loc["Linear regression", "out_of_range"]
    if lin > 0:
        st.caption(f"Linear regression predicted a probability below 0% or above 100% for {lin:.0%} of cases before clipping.")
    t.insight(why)
    with st.expander("How the bench is run"):
        st.markdown(f"""
- **Same rows, same test.** Every algorithm trains on the same data and is scored on the same later period, so the comparison is fair and out of time.
- **Five algorithms.** Linear regression (a straight-line probability, the baseline), logistic regression (the scorecard family), random forest, XGBoost and a support vector machine with an RBF kernel (Nyström approximation, Platt-calibrated, trained on at most {max(tb['trained_on'].min(), 1):,} rows because kernel methods do not scale).
- **Judged on the decision, not the leaderboard.** Ranking (AUC), calibration, how much of the outcome lands in the top tenth, the money the decision makes, scoring cost and whether a single decision can be explained.
- **Champion rule.** In a regulated decision only models with exact reasons are eligible. Within 1% of the best value, the simpler model wins, because it is cheaper to run, explain and monitor.
""")
    return champ


def subhead(title: str, text: str) -> None:
    st.markdown(f'<div style="margin:2.2rem 0 .6rem;border-top:1px solid #263E36;padding-top:1.4rem">'
                f'<div style="font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#CBFA7C">{t.esc(title)}</div>'
                f'<p style="margin:.35rem 0 0;color:#C9D6CE;max-width:62rem">{text}</p></div>', unsafe_allow_html=True)
