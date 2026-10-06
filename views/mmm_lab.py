"""Marketing mix and budget optimizer: what each channel really returns, and where the next dollar should go."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG

from lab import theme as t
from lab import widgets as w
from lab.cu import ui as cui
from lab.mmm import engine as E

cui.css()
STAGES = ["Data", "Model", "Calibrate", "ROI", "Optimize"]
KEYS = {s.lower(): s for s in STAGES}
COL = dict(zip(E.CHANNELS, ["#008A73", "#D9772B", "#5B6CB8", "#B8486E", "#8C7A1F"]))

t.header(
    "Project · marketing mix · Corvane Connect (fictional)",
    "Where should the next marketing dollar go?",
    "Three years of weekly spend across five acquisition channels and the new subscribers that followed. A marketing mix "
    "model separates each channel's effect, including its carryover and diminishing returns, is calibrated to a geo "
    "experiment, and an optimizer moves budget to where it earns the most lifetime value, inside guardrails.",
)

q = str(st.query_params.get("stage", "")).lower()
if "mm_stage" not in st.session_state:
    st.session_state.mm_stage = KEYS.get(q, STAGES[0])
    st.session_state.mm_stage__last = st.session_state.mm_stage
stage = w.segmented("Stage", STAGES, key="mm_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower()
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")
D = E.data()
df = D.df


# ===========================================================================
if stage == "Data":
    cui.stage_head(1, "Data", "What happened to spend and sign-ups?",
                   "Weekly spend by channel and weekly new subscribers. Spend rose when demand rose, because that is how budgets "
                   "get set, which is exactly why a simple correlation overstates what media does.", "")
    tot = df[E.CHANNELS].sum(1)
    t.tiles([
        {"label": "Weeks of history", "value": f"{E.WEEKS}", "note": f"{df['date'].iloc[0]:%b %Y} – {df['date'].iloc[-1]:%b %Y}"},
        {"label": "Weekly media spend", "value": t.money(tot.tail(52).mean() * 1000), "accent": True, "note": "average, last 52 weeks"},
        {"label": "New subscribers a week", "value": f"{df['new_subs'].tail(52).mean():,.0f}", "note": "average, last 52 weeks"},
        {"label": "Value of a new subscriber", "value": t.money(E.CLV_NEW), "note": "lifetime value, from the subscriber lab"},
    ])
    fig = go.Figure()
    for c in E.CHANNELS:
        fig.add_trace(go.Bar(x=df["date"], y=df[c], name=c, marker=dict(color=COL[c])))
    fig.add_trace(go.Scatter(x=df["date"], y=df["new_subs"], name="New subscribers (right)", yaxis="y2", line=dict(color=t.INK, width=2)))
    fig.update_layout(title="Weekly spend by channel ($K) and new subscribers", barmode="stack", bargap=0.1,
                      yaxis=dict(title="Spend ($K)"), yaxis2=dict(title="New subscribers", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.02), margin=dict(t=110))
    t.chart(fig, height=420)
    left, right = st.columns(2, gap="large")
    with left:
        corr = pd.Series({c: np.corrcoef(df[c], df["new_subs"])[0, 1] for c in E.CHANNELS})
        fig2 = go.Figure(go.Bar(y=corr.index, x=corr.values, orientation="h", marker=dict(color=[COL[c] for c in corr.index], cornerradius=3),
                                hovertemplate="%{y}: r = %{x:.2f}<extra></extra>"))
        fig2.update_layout(title="Raw correlation of spend with sign-ups", xaxis=dict(range=[-0.3, 1], title="Correlation"), yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=300)
    with right:
        tr = pd.Series({c: E.roi_table(E.fit(False)).set_index("channel").loc[c, "true_roi"] for c in E.CHANNELS})
        fig3 = go.Figure(go.Bar(y=tr.index, x=tr.values, orientation="h", marker=dict(color=[COL[c] for c in tr.index], cornerradius=3),
                                hovertemplate="%{y}: $%{x:.2f} per $1<extra></extra>"))
        fig3.update_layout(title="True return per $1 (known only because the data is synthetic)", xaxis_title="Lifetime value per $1 spent",
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig3, height=300)
    t.insight(f"{corr.idxmax()} and paid social correlate most with sign-ups because their budgets rise in peak season, when sign-ups rise "
              f"anyway. Paid social's true return is the lowest of the five, and partners, with almost no correlation, return among the most. "
              "Correlation ranks channels by when they were spent, not by what they did.")
    cui.call("Agree the decision before the model: how much budget is movable, over what horizon, and which channels have contracts "
             "that cannot change this quarter. Those become the optimizer's guardrails.")

# ===========================================================================
elif stage == "Model":
    cal = st.toggle("Use the model calibrated to the geo experiment", value=True, key="mm_cal_m")
    f = E.fit(cal)
    dec = E.decompose(f)
    cui.stage_head(2, "Model", "How much of each week came from each channel?",
                   "Regression on transformed spend: carryover (adstock) spreads a week's spend across later weeks, saturation "
                   "bends returns downward, and trend, seasonality, holidays and promotions form the base. Carryover and "
                   "saturation are searched channel by channel and chosen on fit.", "")
    media = dec[E.CHANNELS].sum(1).sum() / dec["fitted"].sum()
    t.tiles([
        {"label": "Holdout error", "value": t.pct(f.mape, 1), "accent": True, "note": "last 26 weeks, model fitted without them"},
        {"label": "Sign-ups driven by media", "value": t.pct(media), "note": "the rest is base demand"},
        {"label": "Longest carryover", "value": max(f.decay, key=f.decay.get), "note": f"{max(f.decay.values()):.0%} of a week's effect carries to the next"},
        {"label": "Channels", "value": f"{len(E.CHANNELS)}", "note": "search, social, TV, mail, partners"},
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec["date"], y=dec["base"], name="Base demand", stackgroup="one", line=dict(width=0), fillcolor="rgba(163,171,165,.55)"))
    for c in E.CHANNELS:
        fig.add_trace(go.Scatter(x=dec["date"], y=dec[c], name=c, stackgroup="one", line=dict(width=0, color=COL[c])))
    fig.add_trace(go.Scatter(x=dec["date"], y=dec["actual"], name="Actual", line=dict(color=t.INK, width=1.6)))
    fig.add_vrect(x0=dec["date"].iloc[-26], x1=dec["date"].iloc[-1], fillcolor="rgba(127,208,190,.12)", line_width=0)
    fig.update_layout(title="New subscribers each week, decomposed (shaded = holdout period)", yaxis_title="New subscribers",
                      legend=dict(orientation="h", y=1.02), margin=dict(t=110))
    t.chart(fig, height=420)
    tab = pd.DataFrame({"Channel": E.CHANNELS,
                        "Carryover (model)": [f"{f.decay[c]:.0%}" for c in E.CHANNELS],
                        "Carryover (true)": [f"{E.TRUE[c]['decay']:.0%}" for c in E.CHANNELS],
                        "Half-saturation spend (model)": [t.money(f.half[c] * (1 - f.decay[c]) * 1000) for c in E.CHANNELS],
                        "Weekly spend today": [t.money(df[c].tail(52).mean() * 1000) for c in E.CHANNELS]})
    st.dataframe(tab, hide_index=True, width="stretch")
    t.insight("Connected TV carries over the longest: a flight keeps producing sign-ups for weeks after it ends, which a week-by-week "
              "attribution report would miss entirely. Search and partners act almost immediately.")
    cui.call("A mix model is a planning tool, not a weekly attribution report. It is refreshed quarterly, checked against experiments "
             "and used for budget decisions; day-to-day optimization stays inside each channel's own platform.")

# ===========================================================================
elif stage == "Calibrate":
    cui.stage_head(3, "Calibrate", "Does the model agree with an experiment?",
                   "For eight weeks, paid social was switched off in a random fifth of regions. The sign-ups those regions lost "
                   "measure social's true incremental effect. The model is then refitted with that measurement as an anchor.", "")
    exp = D.experiment
    f0, f1 = E.fit(False), E.fit(True)
    r0 = E.roi_table(f0).set_index("channel")
    r1 = E.roi_table(f1).set_index("channel")
    test = (df["week"] >= exp["weeks"][0]) & (df["week"] < exp["weeks"][1])
    implied0 = float(E.decompose(f0).loc[test, "Paid social"].mean())
    implied1 = float(E.decompose(f1).loc[test, "Paid social"].mean())
    t.tiles([
        {"label": "Experiment: social's weekly effect", "value": f"{exp['weekly_incremental']:,.0f}", "accent": True,
         "note": f"new subscribers a week, ± {1.96 * exp['se']:,.0f}"},
        {"label": "Uncalibrated model said", "value": f"{implied0:,.0f}", "note": f"{implied0 / exp['weekly_incremental'] - 1:+.0%} vs the experiment"},
        {"label": "Calibrated model says", "value": f"{implied1:,.0f}", "accent": True, "note": f"{implied1 / exp['weekly_incremental'] - 1:+.0%} vs the experiment"},
        {"label": "Social ROI, before → after", "value": f"{r0.loc['Paid social', 'roi']:.2f} → {r1.loc['Paid social', 'roi']:.2f}", "note": f"true {r1.loc['Paid social', 'true_roi']:.2f}"},
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=E.CHANNELS, y=r0["roi"], name="Uncalibrated model", marker=dict(color=t.S2)))
    fig.add_trace(go.Bar(x=E.CHANNELS, y=r1["roi"], name="Calibrated to the experiment", marker=dict(color=t.S1)))
    fig.add_trace(go.Scatter(x=E.CHANNELS, y=r1["true_roi"], name="True (known here)", mode="markers", marker=dict(symbol="line-ew-open", size=34, color=t.INK, line=dict(width=2.5))))
    fig.update_layout(title="Lifetime value returned per $1, by channel", yaxis_title="$ per $1", barmode="group", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
    t.chart(fig, height=400)
    t.insight(f"Without the experiment, the model gives paid social credit for seasonal demand it happened to coincide with and "
              f"overstates its return ({r0.loc['Paid social', 'roi']:.2f} per $1). Anchored to the geo test, it falls to "
              f"{r1.loc['Paid social', 'roi']:.2f}, close to the true {r1.loc['Paid social', 'true_roi']:.2f}, and the other channels barely move.")
    cui.call("Run one geo experiment a quarter on the channel the model is least sure about, and feed it back. Over a year the model "
             "stops being an opinion and becomes a summary of experiments.")

# ===========================================================================
elif stage == "ROI":
    cal = st.toggle("Use the model calibrated to the geo experiment", value=True, key="mm_cal_r")
    f = E.fit(cal)
    cui.stage_head(4, "Return on spend", "Which channels pay back, and which are saturated?",
                   "Average return says whether a channel has paid back so far; marginal return says what the next dollar will "
                   "do. Budget decisions run on the second. Intervals come from a block bootstrap of the weekly data.", "")
    r = E.roi_table(f).set_index("channel")
    bs = E.bootstrap(cal)
    lo, hi = bs.quantile(0.05), bs.quantile(0.95)
    best_m = r["mroi"].idxmax()
    worst_m = r["mroi"].idxmin()
    t.tiles([
        {"label": "Best average return", "value": r["roi"].idxmax(), "note": f"${r['roi'].max():.2f} per $1"},
        {"label": "Best next dollar", "value": best_m, "accent": True, "note": f"${r.loc[best_m, 'mroi']:.2f} per extra $1"},
        {"label": "Most saturated", "value": worst_m, "note": f"${r.loc[worst_m, 'mroi']:.2f} per extra $1"},
        {"label": "Cost per new subscriber", "value": t.money(r["spend"].sum() / r["subs"].sum()), "note": "all media, blended"},
    ])
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=E.CHANNELS, x=r["roi"], name="Average return", orientation="h", marker=dict(color=[COL[c] for c in E.CHANNELS], cornerradius=3),
                             error_x=dict(type="data", symmetric=False, array=hi - r["roi"], arrayminus=r["roi"] - lo, color=t.INK, thickness=1.2, width=0)))
        fig.add_trace(go.Scatter(y=E.CHANNELS, x=r["mroi"], name="Next-dollar return", mode="markers", marker=dict(size=11, color=t.INK, symbol="diamond")))
        fig.add_vline(x=1, line=dict(color=t.S2, width=1, dash="dot"))
        fig.update_layout(title="Return per $1: average and next dollar", xaxis_title="Lifetime value per $1",
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.02), margin=dict(l=8, t=110))
        t.chart(fig, height=400)
    with right:
        fig2 = go.Figure()
        for c in E.CHANNELS:
            cur = df[c].tail(52).mean()
            xs = np.linspace(0, cur * 2.2, 80)
            fig2.add_trace(go.Scatter(x=xs, y=E.steady_response(f, c, xs), name=c, line=dict(color=COL[c], width=2)))
            fig2.add_trace(go.Scatter(x=[cur], y=E.steady_response(f, c, [cur]), mode="markers", marker=dict(size=9, color=COL[c], line=dict(color="#fff", width=1.5)),
                                      showlegend=False, hovertemplate=c + " today: $%{x:.0f}K → %{y:.0f} a week<extra></extra>"))
        cur_w = df[worst_m].tail(52).mean()
        fig2.add_annotation(x=cur_w * 2.2, y=float(E.steady_response(f, worst_m, [cur_w * 2.2])[0]), text=f"{worst_m}: nearly flat past today's spend",
                            showarrow=True, ax=-10, ay=-40, xanchor="right", font=dict(size=12, color="#0E1311"), bgcolor="rgba(255,255,255,.9)", bordercolor="#E1E4DE", borderwidth=1, borderpad=4, arrowcolor="#4A534D", arrowwidth=1, arrowhead=0)
        fig2.update_layout(title="Response curves: weekly sign-ups by weekly spend (● = today)", xaxis_title="Weekly spend ($K)", yaxis_title="New subscribers a week",
                           legend=dict(orientation="h", y=1.02), margin=dict(t=110))
        t.chart(fig2, height=400)
    t.insight(f"Every channel pays back on average, but they sit at very different points on their curves. {worst_m} is close to "
              f"flat: the next dollar there returns ${r.loc[worst_m, 'mroi']:.2f}. {best_m} still has room, at ${r.loc[best_m, 'mroi']:.2f} per extra dollar.")
    cui.call("Average return justifies a channel's existence; marginal return decides its next budget. Reporting only the first is how "
             "saturated channels keep growing.")

# ===========================================================================
elif stage == "Optimize":
    cui.stage_head(5, "Optimize", "With the same budget, what mix earns the most?",
                   "The optimizer reallocates weekly spend across channels to maximize new subscribers, with every channel held "
                   "within a guardrail of today's level so the plan is one a marketing team can actually run.", "")
    f = E.fit(True)
    cur_total = float(sum(df[c].tail(52).mean() for c in E.CHANNELS))
    with st.container(border=True):
        a, b = st.columns(2)
        budget_pct = a.slider("Weekly budget vs today", 60, 140, 100, 5, format="%d%%", key="mm_budget")
        guard = b.slider("Guardrail: most any channel can move", 10, 60, 30, 5, format="±%d%%", key="mm_guard") / 100
    budget = cur_total * budget_pct / 100
    o = E.optimize(f, budget, guard)
    base_subs = o["subs_current"].sum()
    gain = o["subs_optimal"].sum() - base_subs * (1 if budget_pct == 100 else 1)
    same_mix = float(sum(E.steady_response(f, c, df[c].tail(52).mean() * budget_pct / 100) for c in E.CHANNELS))
    gain_vs_mix = o["subs_optimal"].sum() - same_mix
    true_gain = o["true_optimal"].sum() - sum(E.true_response(c, df[c].tail(52).mean() * budget_pct / 100) for c in E.CHANNELS)
    t.tiles([
        {"label": "Weekly budget", "value": t.money(budget * 1000), "note": f"{budget_pct}% of today's {t.money(cur_total * 1000)}"},
        {"label": "Extra sign-ups a week", "value": f"{gain_vs_mix:+,.0f}", "accent": True, "note": "optimized mix vs today's mix at this budget"},
        {"label": "Lifetime value a year", "value": t.money(gain_vs_mix * 52 * E.CLV_NEW), "accent": True, "note": f"at {t.money(E.CLV_NEW)} per subscriber"},
        {"label": "Check against the truth", "value": f"{true_gain:+,.0f}", "note": "what the new mix really adds (known here)"},
    ])
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=E.CHANNELS, x=o["current"] * budget_pct / 100, name="Today's mix", orientation="h", marker=dict(color=t.BASE)))
        fig.add_trace(go.Bar(y=E.CHANNELS, x=o["optimal"], name="Optimized", orientation="h", marker=dict(color=[COL[c] for c in E.CHANNELS])))
        fig.update_layout(title="Weekly spend by channel ($K)", barmode="group", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.02),
                          margin=dict(l=8, t=100), bargap=0.25)
        t.chart(fig, height=380)
    with right:
        fr = E.frontier(True, guard)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fr["budget"] * 1000, y=fr["subs_current_mix"], name="Today's mix, scaled", line=dict(color=t.BASE, width=2, dash="dot")))
        fig2.add_trace(go.Scatter(x=fr["budget"] * 1000, y=fr["subs"], name="Optimized mix", line=dict(color=t.S1, width=2.5)))
        fig2.add_vline(x=budget * 1000, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Media-driven sign-ups a week by budget", xaxis=dict(title="Weekly budget", tickprefix="$", tickformat="~s"),
                           yaxis_title="New subscribers a week", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig2, height=380)
    plan = pd.DataFrame({"Channel": o["channel"], "Today": (o["current"] * 1000).map(t.money), "Plan": (o["optimal"] * 1000).map(t.money),
                         "Change": [f"{(b_ / a_ - 1):+.0%}" for a_, b_ in zip(o["current"], o["optimal"])],
                         "Sign-ups a week, today → plan": [f"{a_:,.0f} → {b_:,.0f}" for a_, b_ in zip(o["subs_current"], o["subs_optimal"])]})
    st.dataframe(plan, hide_index=True, width="stretch")
    need = fr.loc[fr["subs"] >= base_subs, "budget"].min() if (fr["subs"] >= base_subs).any() else None
    t.insight("The plan takes money out of the saturated channel and puts it where the curve is still steep, with no new budget."
              + (f" The frontier shows the other lever: the optimized mix matches today's results at about {t.money(need * 1000)} a week, "
                 f"{1 - need / cur_total:.0%} less than today." if need is not None and need < cur_total * 0.99 else ""))
    cui.call("Move in steps, inside guardrails, and measure each step: shift the budget, hold a geo out, read the result, refit. An "
             "optimizer that recommends a 60% swing in one quarter is extrapolating beyond anything the data has seen.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.mm_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Corvane Connect is fictional. Spend, sign-ups and the geo experiment are synthetic; true channel effects are planted so the model can be scored.")
