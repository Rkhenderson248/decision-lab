"""Causal impact lab: did the program work when nobody could run a test?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG

from lab import bench_ui as BU
from lab import theme as t
from lab import widgets as w
from lab.causal import engine as E
from lab.cu import ui as cui

cui.css()
STAGES = ["Frame", "Diff-in-diff", "Synthetic control", "Matching", "Decide"]
KEYS = {"frame": "Frame", "did": "Diff-in-diff", "synthetic": "Synthetic control", "matching": "Matching", "decide": "Decide"}

t.header(
    "Project · causal inference · Corvane Connect (fictional)",
    "Did it work, when nobody could run a test?",
    "A loyalty price lock went live in eight of forty markets, chosen by regional managers, not at random. An autopay "
    "discount was offered to everyone and joined by whoever wanted it. Naive comparisons give the wrong answer for both. "
    "Three causal methods, each scored against the true effect, which only synthetic data can reveal.",
)

q = str(st.query_params.get("stage", "")).lower()
if "ci_stage" not in st.session_state:
    st.session_state.ci_stage = KEYS.get(q, STAGES[0])
    st.session_state.ci_stage__last = st.session_state.ci_stage

scen_label = st.radio("How did managers choose the eight markets?", ["They picked high-churn markets", "They picked markets already getting worse"],
                      horizontal=True, key="ci_scen",
                      help="The second choice breaks the assumption difference-in-differences relies on: that treated and untreated markets were moving in parallel.")
scen = "level" if scen_label.startswith("They picked high") else "trend"
P = E.panel(scen)
stage = w.segmented("Stage", STAGES, key="ci_stage", label_visibility="collapsed")
st.query_params["stage"] = {v: k for k, v in KEYS.items()}[stage]
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")
TRUTH = E.TRUE_EFFECT
fmt = lambda v: f"{v:+.2f}".replace("-", "−")


def weekly_lines(df):
    g = df.groupby(["week", "treated"])["rate"].mean().unstack()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g[False], name="32 untreated markets", line=dict(color=t.BASE, width=2)))
    fig.add_trace(go.Scatter(x=g.index, y=g[True], name="8 price-lock markets", line=dict(color=t.S1, width=2.5)))
    fig.add_vline(x=E.LAUNCH, line=dict(color=t.INK, width=1, dash="dot"))
    fig.add_annotation(x=E.LAUNCH, y=1, yref="paper", text=" launch", showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color=t.MUTED))
    return fig


# ===========================================================================
if stage == "Frame":
    cui.stage_head(1, "Frame", "What does the dashboard say, and why is it wrong?",
                   "Weekly churn per 1,000 subscribers in each market, two years of it. The obvious comparisons are the ones "
                   "a dashboard makes: after versus before, and treated versus untreated.", "")
    nv = E.naive(P)
    d = E.did(P)
    t.tiles([
        {"label": "True effect (known here)", "value": fmt(TRUTH), "accent": True, "note": "churners per 1,000 a week, once ramped up"},
        {"label": "12 weeks after vs before", "value": fmt(nv["12 weeks after vs 12 weeks before"]), "note": "the dashboard view"},
        {"label": "Treated vs untreated", "value": fmt(nv["Treated vs untreated (after launch)"]), "note": "after launch"},
        {"label": "Difference-in-differences", "value": fmt(d["estimate"]), "accent": True, "note": f"95% interval {fmt(d['lo'])} to {fmt(d['hi'])}"},
    ])
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        fig = weekly_lines(P.df)
        fig.update_layout(title="Weekly churners per 1,000 subscribers", xaxis_title="Week", yaxis_title="Per 1,000", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=380)
    with right:
        est = pd.Series({"After vs before": nv["12 weeks after vs 12 weeks before"], "Treated vs untreated": nv["Treated vs untreated (after launch)"],
                         "Diff-in-diff": d["estimate"], "Truth": TRUTH})
        fig2 = go.Figure(go.Bar(x=est.values, y=est.index, orientation="h", marker=dict(color=[t.S2, t.S2, t.S1, t.INK], cornerradius=3),
                                text=[fmt(v) for v in est.values], textposition="outside", cliponaxis=False))
        fig2.add_vline(x=0, line=dict(color=t.GRID, width=1))
        fig2.update_layout(title="Estimated effect (negative = fewer leave)", xaxis=dict(range=[min(est.min(), -0.6) * 1.35, max(est.max(), 0.3) * 1.5]),
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=380)
    t.insight("Both naive views say the price lock made churn <b>worse</b>. After versus before is fooled by seasonality: launch came "
              "just as churn's seasonal climb began. Treated versus untreated is fooled by selection: managers chose markets that "
              "already had high churn. Each method that follows removes one of those problems." if scen == "level" else
              "Here managers chose markets whose churn was already rising. Every simple comparison, and even difference-in-differences, "
              "now credits the program with the deterioration that was coming anyway. The event study in the next stage shows the warning sign.")
    cui.call("Write the question as a counterfactual before touching data: what would churn in these eight markets have been without "
             "the price lock? Every method here is a different way of building that missing line.")

# ===========================================================================
elif stage == "Diff-in-diff":
    cui.stage_head(2, "Difference-in-differences", "Did the treated markets change more than the others did?",
                   "Two-way fixed effects remove each market's own level and each week's shared shocks, so only the change "
                   "specific to treated markets after launch remains. The event study checks the key assumption: before launch, "
                   "the two groups should move in parallel.", "")
    d = E.did(P)
    es = E.event_study(scen)
    pre = es[es["bin"] < -1]
    t.tiles([
        {"label": "Estimated effect", "value": fmt(d["estimate"]), "accent": True, "note": f"95% interval {fmt(d['lo'])} to {fmt(d['hi'])}"},
        {"label": "True effect", "value": fmt(TRUTH), "note": "known here"},
        {"label": "Error", "value": fmt(d["estimate"] - TRUTH), "note": "estimate minus truth"},
        {"label": "Pre-launch drift", "value": fmt(pre["estimate"].iloc[-1] - pre["estimate"].iloc[0]), "accent": abs(pre["estimate"].iloc[-1] - pre["estimate"].iloc[0]) < 0.15,
         "note": "first to last pre-launch bin; near 0 = parallel"},
    ])
    fig = go.Figure()
    fig.add_vrect(x0=-60, x1=0, fillcolor="rgba(225,228,222,.35)", line_width=0)
    fig.add_trace(go.Scatter(x=es["week"], y=es["estimate"], mode="markers+lines", line=dict(color=t.S1, width=2), marker=dict(size=8),
                             error_y=dict(type="data", array=1.96 * es["se"], color=t.S1, thickness=1.2, width=0), name="Effect vs week −4 to −1"))
    fig.add_hline(y=0, line=dict(color=t.INK, width=1))
    fig.add_hline(y=TRUTH, line=dict(color=t.S2, width=1.2, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=TRUTH, text="true effect ", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=t.S2))
    fig.add_annotation(x=-26, y=1, yref="paper", text="before launch: should hover at zero", showarrow=False, yanchor="top", font=dict(size=11, color=t.MUTED))
    fig.update_layout(title="Event study: treated minus untreated, by weeks from launch (4-week bins)", xaxis_title="Weeks from launch",
                      yaxis_title="Churners per 1,000 a week", showlegend=False)
    t.chart(fig, height=400)
    t.insight("Before launch the estimates sit near zero, so the parallel-trends assumption is believable, and after launch they "
              f"settle near the true effect. Difference-in-differences recovers {fmt(d['estimate'])} against a true {fmt(TRUTH)}."
              if scen == "level" else
              "The pre-launch estimates climb steadily toward zero: the treated markets were already deteriorating relative to the rest. "
              f"Parallel trends fail, and difference-in-differences reports {fmt(d['estimate'])}, the wrong sign. The chart is the warning; "
              "the number alone would have been believed.")
    cui.call("Never report a difference-in-differences number without its event study. The pre-launch half of the chart is the only "
             "evidence that the comparison group is a fair stand-in, and it takes one glance to read.")
    with st.expander("Method notes"):
        st.markdown("""
- **Model:** weekly rate on market and week fixed effects plus a treated-after-launch indicator; the six-week ramp after launch is excluded so the estimate is the fully ramped effect.
- **Standard errors:** clustered by market, because weeks within a market are correlated.
- **Event study:** the same model with one indicator per four-week bin relative to launch, the bin just before launch as reference.
- **When it breaks:** when treated units were already on a different trajectory, or when the program changes the comparison group too.
""")

# ===========================================================================
elif stage == "Synthetic control":
    cui.stage_head(3, "Synthetic control", "What would the treated markets have done without the program?",
                   "Build the missing line directly: a weighted blend of untreated markets that tracks the treated markets "
                   "before launch. After launch, the gap between the two is the effect. Placebo runs on every untreated market "
                   "say whether a gap that size could appear by chance.", "")
    pre_weeks = st.slider("Weeks of pre-launch history used to fit the weights", 16, 52, 52, 4, key="ci_pre")
    sc = E.synthetic_control(scen, pre_weeks)
    t.tiles([
        {"label": "Estimated effect", "value": fmt(sc["effect"]), "accent": True, "note": "average gap after the ramp"},
        {"label": "True effect", "value": fmt(TRUTH), "note": "known here"},
        {"label": "Placebo p-value", "value": f"{sc['p_value']:.2f}", "accent": sc["p_value"] <= 0.1, "note": "share of placebo markets with a gap this unusual"},
        {"label": "Pre-launch fit error", "value": f"{sc['pre_rmspe']:.2f}", "note": "root mean squared gap before launch"},
    ])
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        weeks = np.arange(E.WEEKS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=sc["synthetic"], name="Synthetic control", line=dict(color=t.S3, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=weeks, y=sc["treated"], name="Price-lock markets (actual)", line=dict(color=t.S1, width=2.5)))
        fig.add_vline(x=E.LAUNCH, line=dict(color=t.INK, width=1, dash="dot"))
        fig.update_layout(title="Actual vs synthetic: weekly churners per 1,000", xaxis_title="Week", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=330)
        fig3 = go.Figure()
        for pl in sc["placebo"]:
            fig3.add_trace(go.Scatter(x=weeks, y=pl["gap"], mode="lines", line=dict(color="rgba(163,171,165,.35)", width=1), hoverinfo="skip", showlegend=False))
        fig3.add_trace(go.Scatter(x=weeks, y=sc["gap"], mode="lines", line=dict(color=t.S1, width=2.8), name="Price-lock markets"))
        fig3.add_vline(x=E.LAUNCH, line=dict(color=t.INK, width=1, dash="dot"))
        fig3.add_hline(y=0, line=dict(color=t.INK, width=1))
        fig3.update_layout(title="Gap vs synthetic, against 32 placebo markets (grey)", xaxis_title="Week", yaxis_title="Gap per 1,000",
                           legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig3, height=300)
    with right:
        wts = sc["weights"][sc["weights"] > 0.01].head(10).iloc[::-1]
        fig2 = go.Figure(go.Bar(y=wts.index, x=wts.values * 100, orientation="h", marker=dict(color=t.S3, cornerradius=3),
                                hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        fig2.update_layout(title="Who makes up the synthetic control", xaxis_title="Weight (%)", margin=dict(l=8, t=70))
        t.chart(fig2, height=420)
    t.insight(f"The blend of untreated markets tracks the treated ones closely before launch and separates after it. Synthetic "
              f"control estimates {fmt(sc['effect'])} against a true {fmt(TRUTH)}, and the treated gap is more extreme than almost every "
              "placebo." if scen == "level" else
              f"Because the blend is fitted to the pre-launch path, it picks untreated markets that were also deteriorating. Synthetic "
              f"control estimates {fmt(sc['effect'])}: the right sign and much closer to {fmt(TRUTH)} than difference-in-differences.")
    cui.call("Synthetic control earns its keep when there are few treated units and a long history, which describes most regional "
             "pilots. It is also the easiest causal estimate to explain to an executive: here is the market, here is its twin.")

# ===========================================================================
elif stage == "Matching":
    cui.stage_head(4, "Propensity matching", "Did the autopay discount reduce churn, or did loyal customers join it?",
                   "Customers chose to join, so joiners were already different: longer tenure, more digital, fewer care calls. "
                   "Matching pairs each joiner with a non-joiner who had the same chance of joining, then compares churn.", "")
    with st.container(border=True):
        a, b = st.columns(2)
        hidden = a.toggle("Add a hidden driver", value=False, key="ci_hidden", help="Engagement, never recorded, makes customers both more likely to join and less likely to leave.")
        caliper = b.slider("Caliper (maximum distance for a match, in standard deviations)", 0.02, 0.5, 0.2, 0.02, key="ci_cal")
    mt = E.matching(hidden, caliper)
    t.tiles([
        {"label": "Naive difference", "value": f"{mt['naive'] * 100:+.1f} pts".replace("-", "−"), "note": "joiners vs everyone else"},
        {"label": "Matched estimate", "value": f"{mt['matched'] * 100:+.1f} pts".replace("-", "−"), "accent": True,
         "note": f"95% interval {mt['lo'] * 100:+.1f} to {mt['hi'] * 100:+.1f}".replace("-", "−")},
        {"label": "Weighted estimate (IPW)", "value": f"{mt['ipw'] * 100:+.1f} pts".replace("-", "−"), "note": "a second method, same assumptions"},
        {"label": "True effect", "value": f"{mt['truth'] * 100:+.1f} pts".replace("-", "−"), "accent": True, "note": f"{mt['matched_share']:.0%} of joiners matched"},
    ])
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        sb, sa = mt["smd_before"], mt["smd_after"]
        labels = [E.COV_LABELS[c] for c in sb.index]
        fig = go.Figure()
        fig.add_vrect(x0=-0.1, x1=0.1, fillcolor="rgba(127,208,190,.18)", line_width=0)
        fig.add_trace(go.Scatter(x=sb.values, y=labels, mode="markers", name="Before matching", marker=dict(size=11, color=t.S2)))
        fig.add_trace(go.Scatter(x=sa.values, y=labels, mode="markers", name="After matching", marker=dict(size=11, color=t.S1, symbol="diamond")))
        fig.add_vline(x=0, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Covariate balance (shaded band = balanced)", xaxis_title="Joiners minus non-joiners (SD)",
                          legend=dict(orientation="h", y=1.02), margin=dict(t=100, l=8))
        t.chart(fig, height=340)
    with right:
        ps = mt["ps"]
        fig2 = go.Figure()
        for flag, name, color in ((True, "Joined", t.S1), (False, "Did not join", t.S2)):
            fig2.add_trace(go.Histogram(x=ps.loc[ps["joined"] == flag, "ps"], name=name, nbinsx=40, opacity=0.6, marker=dict(color=color),
                                        histnorm="probability density"))
        fig2.update_layout(title="Overlap: chance of joining, by who joined", barmode="overlay", xaxis_title="Estimated probability of joining",
                           yaxis_title="Density", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig2, height=340)
    t.insight("Joiners churned far less, but most of that gap is who they were, not what the discount did. After matching, the "
              "covariates balance and the estimate lands near the truth." if not hidden else
              "With an unmeasured driver, the covariates still balance perfectly and the estimate is still wrong. Matching only removes "
              "differences you measured. Balance tables cannot reveal what is missing; that takes a sensitivity analysis or an experiment.")
    cui.call("Matching is the right tool when a program is opt-in and the reasons for opting in are recorded. When they are not, the "
             "honest answer is a range, and a small randomized holdout next time.")

# ===========================================================================
elif stage == "Decide":
    cui.stage_head(5, "Decide", "Which number goes to the business, and with what caveat?",
                   "Every estimate for the price lock side by side with the truth, the rule for choosing among them, and how "
                   "strong a hidden confounder would need to be to overturn the matched result.", "")
    nv = E.naive(P)
    d = E.did(P)
    sc = E.synthetic_control(scen)
    est = pd.DataFrame([
        ("After vs before", nv["12 weeks after vs 12 weeks before"], None, None),
        ("Treated vs untreated", nv["Treated vs untreated (after launch)"], None, None),
        ("Difference-in-differences", d["estimate"], d["lo"], d["hi"]),
        ("Synthetic control", sc["effect"], None, None),
    ], columns=["method", "estimate", "lo", "hi"])
    fig = go.Figure()
    for r in est.itertuples():
        color = t.S2 if r.method in ("After vs before", "Treated vs untreated") else t.S1
        fig.add_trace(go.Scatter(x=[r.estimate], y=[r.method], mode="markers", marker=dict(size=13, color=color), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[(r.hi - r.estimate) if r.hi else 0],
                                              arrayminus=[(r.estimate - r.lo) if r.lo else 0], color=color, thickness=1.5, width=0)))
    fig.add_vline(x=TRUTH, line=dict(color=t.INK, width=1.5, dash="dot"))
    fig.add_annotation(x=TRUTH, y=1, yref="paper", text=" truth", showarrow=False, xanchor="left", yanchor="top", font=dict(size=11))
    fig.add_vline(x=0, line=dict(color=t.GRID, width=1))
    fig.update_layout(title="Price lock: every estimate against the truth", xaxis_title="Churners per 1,000 a week", yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        t.chart(fig, height=330)
    with right:
        mt = E.matching(False)
        base = E.customers(False)
        p_ctrl = base.loc[~base["joined"], "churned"].mean()
        rr = (p_ctrl + mt["matched"]) / p_ctrl
        ev = E.e_value(rr)
        t.tiles([
            {"label": "Matched risk ratio", "value": f"{rr:.2f}", "note": "autopay discount, churn joiners vs matched"},
            {"label": "E-value", "value": f"{ev:.2f}", "accent": True, "note": "how strong a hidden confounder must be to erase it"},
        ])
        st.markdown(f"An unmeasured factor would need to be associated with both joining and churning by a risk ratio of **{ev:.1f}** each "
                    "to fully explain away the matched effect. That is the bar any alternative explanation has to clear, and the reason a small randomized holdout is still worth running.")
    rules = pd.DataFrame([
        ("Randomized test", "Whenever the decision can wait for one", "Nothing to assume"),
        ("Difference-in-differences", "Many treated and untreated units, clean launch date", "Parallel trends; check the event study"),
        ("Synthetic control", "Few treated units, long pre-period", "A blend of controls can track the treated before launch"),
        ("Matching / weighting", "Opt-in programs with recorded reasons", "Every driver of joining is measured"),
    ], columns=["Method", "Use when", "Assumes"])
    st.dataframe(rules, hide_index=True, width="stretch")
    t.insight("The naive numbers point the wrong way in both scenarios. When trends are parallel, difference-in-differences and "
              "synthetic control agree, which is the strongest evidence a non-experimental study can offer. When they disagree, "
              "trust the method whose assumption you can see holding, and say so in the readout." if scen == "level" else
              "When the methods disagree, the event study explains why: difference-in-differences is invalid here, synthetic control "
              "is not. The readout reports synthetic control, shows the failed pre-trend, and recommends a randomized rollout for the next wave.")
    cui.call("Report a causal number with three things beside it: the method, the assumption it rests on, and the evidence that "
             "assumption holds. Agreement between two methods with different assumptions is worth more than a tighter interval from one.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.ci_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Corvane Connect is fictional. Markets, customers and effects are synthetic; the true effects are planted so every method can be scored.")
