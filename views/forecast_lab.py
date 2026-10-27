"""Forecasting at scale: 72 weekly series for a fictional grocery chain, forecast, reconciled, turned into orders."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.cu import ui
from lab.forecast import data as D
from lab.forecast import models as F

ui.css()
STAGES = ["Demand", "Model bake-off", "Reconcile", "Uncertainty", "Order", "Monitor"]

t.header(
    "Product · Larkspur Market (fictional)",
    "Forecasting at scale: 72 series, one model, orders that add up",
    "Twelve grocery stores in three regions, six categories, three years of weekly sales. The question is the one a "
    "planner faces every Monday: how much of each category will each store sell over the next quarter, how sure are "
    "we, and how much should we order? Synthetic data with known structure, backtested on four rolling quarters.",
)


@st.cache_resource(show_spinner="Simulating the chain and backtesting four quarters…")
def study() -> F.Study:
    return F.run()


S = study()
chain, Y = S.chain, S.chain.wide
dates = D.week_dates()

q = str(st.query_params.get("stage", "")).lower().replace("-", "").replace(" ", "")
lookup = {s.lower().replace(" ", "").replace("-", ""): s for s in STAGES}
lookup.update({"bakeoff": "Model bake-off", "models": "Model bake-off", "orders": "Order"})
if "fc_stage" not in st.session_state:
    st.session_state.fc_stage = lookup.get(q, STAGES[0])
    st.session_state.fc_stage__last = st.session_state.fc_stage
stage = w.segmented("Stage", STAGES, key="fc_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace(" ", "-")
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")

scores = S.scores.groupby("model")[["WAPE", "Bias", "MASE"]].mean()
gbm, sn = scores.loc["Global gradient boosting"], scores.loc["Seasonal naive"]

# ===========================================================================
if stage == "Demand":
    ui.stage_head(1, "Know the demand", "What does a year of grocery demand look like?",
                  "Seventy-two store-category series share a few patterns: a yearly cycle that differs by category, "
                  "holiday spikes, promotions planned in advance, slow growth, and noise. A forecast at this scale has "
                  "to learn the shared patterns without flattening what is different.", "diagnostic")
    tot = Y.sum(axis=1)
    t.tiles([
        {"label": "Series", "value": f"{Y.shape[1]}", "note": "12 stores × 6 categories"},
        {"label": "Weeks of history", "value": f"{D.WEEKS}", "note": f"{dates[0]:%b %Y} – {dates[-1]:%b %Y}"},
        {"label": "Units a week, chain", "value": f"{tot.mean():,.0f}", "note": f"peak {tot.max():,.0f} at the holidays"},
        {"label": "Promoted series-weeks", "value": f"{chain.promo.values.mean():.0%}", "accent": True, "note": "known before the week starts"},
    ])
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.markdown("##### Chain demand by category")
        cats = {c: Y[[k for k in Y.columns if k.endswith(c)]].sum(axis=1) for c in D.CATEGORIES}
        fig = go.Figure()
        pal = [t.S1, t.S2, t.S3, "#E8D26B", "#7FD0BE", t.BASE]
        for (c, v), col in zip(cats.items(), pal):
            fig.add_scatter(x=dates, y=v, name=c, stackgroup="one", line=dict(width=.5, color=col))
        fig.update_layout(yaxis_title="Units a week", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 340)
    with b:
        st.markdown("##### The yearly shape differs by category")
        woy = np.arange(D.WEEKS) % 52
        fig = go.Figure()
        for (c, v), col in zip(cats.items(), pal):
            prof = pd.Series(v.values).groupby(woy).mean()
            fig.add_scatter(x=prof.index, y=prof / prof.mean(), name=c, line=dict(color=col, width=2))
        fig.update_layout(yaxis_title="Index (average week = 1)", xaxis_title="Week of year", legend=dict(orientation="h", y=1.18, font=dict(size=11)))
        t.chart(fig, 340)
    st.markdown("##### One series, close up")
    pick = st.selectbox("Series", list(Y.columns), index=list(Y.columns).index("S07|Beverages"), key="fc_series")
    fig = go.Figure()
    fig.add_scatter(x=dates, y=Y[pick], name="Units", line=dict(color=t.S1, width=2))
    pw = chain.promo[pick].values == 1
    fig.add_scatter(x=dates[pw], y=Y[pick][pw], mode="markers", name="Promotion week", marker=dict(color=t.S2, size=8))
    hw = chain.holiday == 1
    fig.add_scatter(x=dates[hw], y=Y[pick][hw], mode="markers", name="Holiday week", marker=dict(color=t.S3, size=8, symbol="diamond"))
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    t.chart(fig, 300)

# ===========================================================================
elif stage == "Model bake-off":
    ui.stage_head(2, "Model bake-off", "Which forecasting approach should run every week?",
                  "Three candidates, judged on four quarters the model never saw. Seasonal naive repeats last year; "
                  "damped Holt-Winters fits each series alone; a single gradient-boosting model learns from all 72 "
                  "series at once, with promotions, holidays and recent growth as features.", "build")
    t.tiles([
        {"label": "Global model · WAPE", "value": f"{gbm.WAPE:.1%}", "accent": True, "note": "weighted absolute % error, 13 weeks ahead"},
        {"label": "Seasonal naive · WAPE", "value": f"{sn.WAPE:.1%}", "note": "the baseline to beat"},
        {"label": "Error cut", "value": f"{1 - gbm.WAPE / sn.WAPE:.0%}", "note": "vs repeating last year"},
        {"label": "Bias · global model", "value": f"{gbm.Bias:+.1%}", "note": f"seasonal naive {sn.Bias:+.1%}"},
    ])
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown("##### Error by quarter, out of sample")
        fig = go.Figure()
        for m, col in [("Seasonal naive", t.BASE), ("Holt-Winters (damped)", t.S3), ("Global gradient boosting", t.S1)]:
            d = S.scores[S.scores.model == m]
            fig.add_bar(x=[str(dates[o + 1].date()) for o in d.origin], y=d.WAPE, name=m, marker_color=col)
        fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="WAPE", xaxis_title="Forecast from week starting", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 330)
    with b:
        st.markdown("##### What the global model leans on")
        imp = S.importance.head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["increase in error"], y=imp.feature, orientation="h", marker_color=t.S1))
        fig.update_layout(xaxis_title="Error increase when shuffled")
        t.chart(fig, 330)
    st.markdown("##### Last fold, one series")
    pick = st.selectbox("Series", list(Y.columns), index=list(Y.columns).index("S03|Produce"), key="fc_series2")
    j = list(Y.columns).index(pick)
    fd = S.folds[-1]
    o = fd["o"]
    fig = go.Figure()
    fig.add_scatter(x=dates[o - 40:o + 1], y=Y[pick].values[o - 40:o + 1], name="History", line=dict(color=t.MUTED, width=2))
    fig.add_scatter(x=dates[o + 1:o + 1 + F.H], y=fd["y"][:, j], name="Actual", line=dict(color=t.INK, width=2))
    for k, nm, col in [("sn", "Seasonal naive", t.BASE), ("ets", "Holt-Winters", t.S3), ("gbm", "Global model", t.S1)]:
        fig.add_scatter(x=dates[o + 1:o + 1 + F.H], y=fd[k][:, j], name=nm, line=dict(color=col, width=2.5 if k == "gbm" else 1.5, dash=None if k == "gbm" else "dot"))
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    t.chart(fig, 300)
    ui.call(f"One model for all {Y.shape[1]} series beats one model per series: it borrows strength across stores, knows the "
            f"promotion calendar, and cuts error by <b>{1 - gbm.WAPE / sn.WAPE:.0%}</b> against last year's numbers. It "
            f"trains on {S.train_rows:,} examples built only from weeks before each forecast origin.")

# ===========================================================================
elif stage == "Reconcile":
    ui.stage_head(3, "Reconcile", "Do the store, region and chain forecasts add up?",
                  "Planners read the chain total, finance reads regions, buyers read categories, stores read their own "
                  "lines. Forecast each level separately and the numbers disagree. Reconciliation makes every level add "
                  "up; the backtest decides which way to do it.", "build")
    bl = S.by_level.pivot(index="level", columns="method", values="WAPE")
    order = ["Total", "Region", "Category", "Store", "Region × category", "Store × category"]
    bl = bl.loc[order]
    fd = S.folds[-1]
    gap = float(abs(fd["base"][0].sum() - fd["base"][-len(S.columns):].sum()) / fd["base"][0].sum())
    t.tiles([
        {"label": "Nodes in the hierarchy", "value": f"{len(S.nodes)}", "note": "total, 3 regions, 12 stores, 6 categories, 18 region × category, 72 lines"},
        {"label": "Incoherence before", "value": f"{gap:.1%}", "note": "chain forecast vs the sum of the lines"},
        {"label": "Chain error · bottom-up", "value": f"{bl.loc['Total', 'Bottom-up']:.1%}", "accent": True, "note": f"forecast the chain directly: {bl.loc['Total', 'Base (incoherent)']:.1%}"},
        {"label": "Incoherence after", "value": "0.0%", "note": "every level sums exactly"},
    ])
    fig = go.Figure()
    for m, col in [("Base (incoherent)", t.BASE), ("Reconciled (WLS)", t.S3), ("Bottom-up", t.S1)]:
        fig.add_bar(x=order, y=bl[m], name=m, marker_color=col)
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="WAPE, averaged over four quarters", legend=dict(orientation="h", y=1.12))
    t.chart(fig, 340)
    ui.call("The textbook answer is often a weighted reconciliation. Here the backtest says otherwise: the global model's "
            "line forecasts are strong enough that adding them up (bottom-up) beats forecasting each aggregate on its "
            "own at every level above the line. The method is chosen by the backtest, not by habit.")

# ===========================================================================
elif stage == "Uncertainty":
    ui.stage_head(4, "Uncertainty", "How sure is the forecast, and is that claim honest?",
                  "Intervals come from the model's own past errors (split conformal): the spread of actual ÷ forecast in "
                  "earlier quarters, by weeks ahead. They are then checked on a quarter they were not built from.", "advisory")
    c80, c95 = S.coverage[.8], S.coverage[.95]
    t.tiles([
        {"label": "80% interval covers", "value": f"{c80['coverage']:.1%}", "accent": True, "note": "of actuals, held-out quarter"},
        {"label": "95% interval covers", "value": f"{c95['coverage']:.1%}", "accent": True, "note": "of actuals, held-out quarter"},
        {"label": "80% width, 1 week ahead", "value": f"±{(c80['hi'][0] - c80['lo'][0]) / 2:.0%}", "note": "of the forecast"},
        {"label": "80% width, 13 weeks ahead", "value": f"±{(c80['hi'][-1] - c80['lo'][-1]) / 2:.0%}", "note": "wider further out"},
    ])
    pick = st.selectbox("Series", list(Y.columns), index=list(Y.columns).index("S10|Frozen"), key="fc_series3")
    j = list(Y.columns).index(pick)
    fd = S.folds[-1]; o = fd["o"]; x = dates[o + 1:o + 1 + F.H]
    fig = go.Figure()
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(c95["U"][:, j]) + list(c95["L"][::-1, j]), fill="toself", fillcolor="rgba(143,162,240,.18)", line=dict(width=0), name="95%")
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(c80["U"][:, j]) + list(c80["L"][::-1, j]), fill="toself", fillcolor="rgba(203,250,124,.22)", line=dict(width=0), name="80%")
    fig.add_scatter(x=x, y=fd["gbm"][:, j], name="Forecast", line=dict(color=t.S1, width=2.5))
    fig.add_scatter(x=x, y=fd["y"][:, j], name="Actual", mode="lines+markers", line=dict(color=t.INK, width=1.5))
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    t.chart(fig, 320)
    ui.call("An interval is a promise. These keep it: the 80% band holds about 80% of what actually happened, on data it "
            "did not see. That is what lets the next stage set orders from it.")

# ===========================================================================
elif stage == "Order":
    ui.stage_head(5, "Order", "How much should each store order?",
                  "A forecast is only worth the decision it improves. Each store-category line orders at the quantile "
                  "that balances a lost sale (margin) against a leftover unit (waste or holding), the newsvendor rule. "
                  "Perishables lean lower; shelf-stable lines lean higher.", "build")
    inv = S.inventory.set_index("method")
    saving = inv.loc["Seasonal naive", "cost"] - inv.loc["Global gradient boosting", "cost"]
    t.tiles([
        {"label": "Cost of mismatch · global model", "value": t.money(inv.loc['Global gradient boosting', 'cost']), "accent": True, "note": "lost margin + leftovers, one quarter"},
        {"label": "Same rule, last-year forecast", "value": t.money(inv.loc['Seasonal naive', 'cost']), "note": "seasonal naive"},
        {"label": "Saved", "value": t.money(saving * 4), "accent": True, "note": f"a year · {saving / inv.loc['Seasonal naive', 'cost']:.0%} less"},
        {"label": "Fill rate", "value": f"{inv.loc['Global gradient boosting', 'fill_rate']:.1%}", "note": f"last-year forecast {inv.loc['Seasonal naive', 'fill_rate']:.1%}"},
    ])
    cats = [c.split("|")[1] for c in S.columns]
    crs = pd.DataFrame({"category": cats, "critical ratio": S.cr}).groupby("category")["critical ratio"].mean().reindex(D.CATEGORIES)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Where each category orders")
        fig = go.Figure(go.Bar(x=crs.index, y=crs.values, marker_color=[t.S2 if v < .5 else t.S1 for v in crs.values], text=[f"{v:.0%}" for v in crs.values], textposition="outside"))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Order-to quantile (critical ratio)", yaxis_range=[0, 1])
        t.chart(fig, 300)
    with b:
        st.markdown("##### Cost of getting it wrong, one quarter")
        fig = go.Figure(go.Bar(x=inv.index, y=inv.cost, marker_color=[t.BASE, t.S1, t.S3], text=[t.money(v) for v in inv.cost], textposition="outside"))
        fig.update_layout(yaxis_title="$")
        t.chart(fig, 300)
    ui.call(f"Same ordering rule, better forecast: about <b>{t.money(saving * 4)}</b> a year less lost margin and waste "
            "across twelve stores, mostly in perishables where a leftover unit is thrown away.")

# ===========================================================================
elif stage == "Monitor":
    ui.stage_head(6, "Monitor", "How do we know when the world has changed?",
                  "Every store is compared each week with what last year and recent growth predict. Eight weeks of "
                  "accumulated error, scaled by its normal spread, becomes a drift score; past ±4 it raises an alert. "
                  "Planted in the data: a competitor opens next to store S07.", "advisory")
    al = F.alerts(S)
    hit = al.dropna()
    t.tiles([
        {"label": "Stores alerted", "value": f"{len(hit)} of {len(al)}", "accent": True, "note": "over three years"},
        {"label": "First alert", "value": f"S07 · week {int(hit.iloc[0]['first alert week'])}" if len(hit) else "none", "note": f"{dates[int(hit.iloc[0]['first alert week'])]:%b %d, %Y}" if len(hit) else ""},
        {"label": "Detection delay", "value": f"{int(hit.iloc[0]['first alert week']) - D.BREAK_WEEK} weeks" if len(hit) else "—", "note": f"competitor opened week {D.BREAK_WEEK}"},
        {"label": "False alarms", "value": f"{max(0, len(hit) - 1)}", "note": "other stores past ±4"},
    ])
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        pick = st.selectbox("Store", list(S.tracking.columns), index=6, key="fc_store")
        fig = go.Figure()
        fig.add_scatter(x=dates, y=S.store_y[pick], name="Actual", line=dict(color=t.INK, width=1.5))
        fig.add_scatter(x=dates, y=S.store_f[pick], name="Expected", line=dict(color=t.S1, width=2, dash="dot"))
        if pick == D.BREAK_STORE:
            fig.add_vline(x=dates[D.BREAK_WEEK], line=dict(color=t.S2, dash="dash"))
        fig.update_layout(legend=dict(orientation="h", y=1.12), yaxis_title="Units a week")
        t.chart(fig, 320)
    with b:
        fig = go.Figure()
        for s in S.tracking.columns:
            fig.add_scatter(x=dates, y=S.tracking[s], name=s, line=dict(color=t.S2 if s == pick else "rgba(166,184,174,.35)", width=2.5 if s == pick else 1), showlegend=False)
        fig.add_hrect(y0=-4, y1=4, fillcolor="rgba(203,250,124,.06)", line_width=0)
        fig.update_layout(yaxis_title="Drift score (z)", title=dict(text=f"{pick} against the other stores", font=dict(size=13)))
        t.chart(fig, 360)
    ui.call("The alert fired four weeks after the competitor opened, on one store only. The response is a re-forecast "
            "of that store with the new level and a note to the buyer, not a retrain of the whole chain.")

st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.fc_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Larkspur Market is fictional. All stores, sales and costs are synthetic; forecasts and savings are "
           "illustrative.")
