"""Next-best-action: a fictional outdoor retailer learns from last month's log which offer to send each customer,
within stock and contact limits, and proves the new policy is better before anyone receives it."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.cu import ui
from lab.nba import data as D
from lab.nba import models as M

ui.css()
STAGES = ["The log", "Response model", "Choose within limits", "Judge before launch", "Guardrails", "Roll out"]

t.header(
    "Product · Fernwood Outfitters (fictional)",
    "Next best action: the right offer for each customer, proven before it ships",
    "Sixty thousand customers, six things the company can do this month (nothing, a coupon, free shipping, a "
    "new-arrivals email, triple loyalty points, a gear bundle) and one log of what was sent last month and what "
    "happened. The job: learn from that log, pick one action per customer within stock and contact limits, and "
    "estimate the profit of the new policy without sending a single offer.",
)


@st.cache_resource(show_spinner="Simulating the customers, training the response model, evaluating policies…")
def study() -> M.Study:
    return M.run()


S = study()
W, log = S.world, S.world.log
N = len(W.customers)
OPE = S.ope
truth = OPE.groupby("policy").truth.first()
dr = OPE[OPE.estimator == "Doubly robust"].set_index("policy")
RULES, BEST, SHIP = "Current rules", "Model, within limits", "Shipped (with guardrails)"
PCOL = {RULES: t.S2, "One offer for all": t.BASE, "Model, no limits": t.S3, BEST: t.S1, SHIP: t.S1}

q = str(st.query_params.get("stage", "")).lower().replace("-", "").replace(" ", "")
lookup = {s.lower().replace(" ", "").replace("-", ""): s for s in STAGES}
lookup.update({"log": "The log", "model": "Response model", "policy": "Choose within limits", "ope": "Judge before launch",
               "offpolicy": "Judge before launch", "rollout": "Roll out", "launch": "Roll out"})
if "nba_stage" not in st.session_state:
    st.session_state.nba_stage = lookup.get(q, STAGES[0])
    st.session_state.nba_stage__last = st.session_state.nba_stage
stage = w.segmented("Stage", STAGES, key="nba_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace(" ", "-")
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")


def per_month(v: float) -> str:
    return t.money(v * N)


# ===========================================================================
if stage == "The log":
    ui.stage_head(1, "Read the log honestly", "What did last month's offers actually do?",
                  "The log records who got what and who bought. It was produced by the old rules, which sent coupons "
                  "to lapsing customers and points to loyalty members, so the groups that received each offer were "
                  "different to begin with. Comparing their purchase rates mixes up the offer with the audience.",
                  "diagnostic")
    t.tiles([
        {"label": "Customers", "value": f"{N:,}", "note": "one decision each this month"},
        {"label": "Logged profit", "value": t.money(log.profit.mean(), 2), "note": "per customer, last month"},
        {"label": "Random share", "value": f"{D.EPS:.0%}", "accent": True, "note": "chosen at random, so every offer was tried on everyone"},
        {"label": "Propensity logged", "value": "Yes", "note": "the chance each sent offer had of being chosen"},
    ])
    naive = log.groupby("action_id").purchased.mean().reindex(range(M.A))
    true_lift = W.p_true.mean(0) - W.p_true[:, 0].mean()
    naive_lift = naive.values - naive.values[0]
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        fig = go.Figure()
        fig.add_bar(y=D.ACTIONS[1:], x=naive_lift[1:] * 100, orientation="h", name="Naive: buy rate vs no-contact group", marker_color=t.S2)
        fig.add_bar(y=D.ACTIONS[1:], x=true_lift[1:] * 100, orientation="h", name="True effect on the average customer", marker_color=t.S1)
        fig.update_layout(barmode="group", xaxis_title="Change in purchase rate (points)", legend=dict(orientation="h", y=1.15),
                          yaxis=dict(autorange="reversed"))
        t.chart(fig, 340)
    with b:
        mix = log.groupby(["action", "rule_action"]).size().unstack(fill_value=0)
        sent = log.action.value_counts().reindex(D.ACTIONS).fillna(0)
        fig = go.Figure(go.Bar(x=sent.values, y=D.ACTIONS, orientation="h", marker_color=[t.BASE] + [t.S3] * 5,
                               text=[f"{v:,.0f}" for v in sent.values], textposition="outside", cliponaxis=False))
        fig.update_layout(xaxis_title="Customers who received it", xaxis_range=[0, sent.max() * 1.22], yaxis=dict(autorange="reversed"),
                          title=dict(text="What the old rules sent", font=dict(size=13)))
        t.chart(fig, 340)
    coupon = D.ACTIONS.index("10% coupon")
    ui.call(f"Read naively, the coupon {'lowers' if naive_lift[coupon] < 0 else 'raises'} purchases by "
            f"{abs(naive_lift[coupon]) * 100:.1f} points. On the same customers it actually raises them by "
            f"{true_lift[coupon] * 100:.1f}: it looked bad because it went to customers who had stopped buying. The random 30% and the "
            "recorded propensities are what make an honest answer possible: they let every later estimate reweight "
            "the log as if offers had been assigned fairly.")

# ===========================================================================
elif stage == "Response model":
    ui.stage_head(2, "Model the response", "For each customer, what happens under each of the six actions?",
                  "One gradient-boosted model learns purchase from customer features plus the action taken, trained on "
                  "half the log. Asking it the same customer six times, once per action, gives six purchase chances; "
                  "times basket, margin and cost, six expected profits.", "build")
    tr = W.value_true
    t.tiles([
        {"label": "AUC, held-out half", "value": f"{S.auc:.3f}", "note": "ranking buyers above non-buyers"},
        {"label": "Calibration gap", "value": f"{(S.calib.pred - S.calib.true).abs().mean() * 100:.1f} pts", "accent": True, "note": "mean |predicted − true| by decile"},
        {"label": "Best action matches truth", "value": f"{(S.v_hat.argmax(1) == tr.argmax(1)).mean():.0%}", "note": "share of customers"},
        {"label": "Training rows", "value": f"{len(S.train):,}", "note": "the other half is kept for judging"},
    ])
    a, b = st.columns([1, 1.1], gap="large")
    with a:
        c = S.calib
        fig = go.Figure()
        fig.add_scatter(x=[0, .4], y=[0, .4], mode="lines", line=dict(color=t.BASE, dash="dot"), showlegend=False)
        fig.add_scatter(x=c.pred, y=c.obs, mode="markers", name="Observed buy rate", marker=dict(color=t.S2, size=9))
        fig.add_scatter(x=c.pred, y=c.true, mode="lines+markers", name="True probability", line=dict(color=t.S1, width=2))
        fig.update_layout(xaxis_title="Predicted chance of purchase", yaxis_title="Actual", legend=dict(orientation="h", y=1.15),
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        t.chart(fig, 340)
    with b:
        seg = M.segment_of(W.customers)
        aff = W.customers.affinity.values
        rows = [f"{s}" for s in M.SEGMENTS] + [f"{a} fans" for a in D.AFFINITY]
        masks = [seg == s for s in M.SEGMENTS] + [aff == a for a in D.AFFINITY]
        z = np.array([(S.v_hat[m][:, 1:] - S.v_hat[m][:, [0]]).mean(0) for m in masks])
        fig = go.Figure(go.Heatmap(z=z, x=D.ACTIONS[1:], y=rows, colorscale=[[0, t.S2], [.5, "#1B2A26"], [1, t.S1]], zmid=0,
                                   text=[[f"{v:+.1f}" for v in r] for r in z], texttemplate="%{text}", colorbar=dict(title="$")))
        fig.update_layout(title=dict(text="Predicted profit vs no contact, $ per customer", font=dict(size=13)), yaxis=dict(autorange="reversed"))
        t.chart(fig, 360)
    ui.call("The model is not judged on accuracy alone. What matters is the difference between actions for the same "
            "customer, and that is where most models are weakest, because each customer only ever got one action. "
            "The heatmap is the claim; the next stages test it.")

# ===========================================================================
elif stage == "Choose within limits":
    ui.stage_head(3, "Choose within limits", "Who gets which offer, when bundles, coupons and attention are scarce?",
                  f"Five thousand bundles are in stock, the coupon budget covers {M.CAPACITY['10% coupon']:,}, and no more "
                  f"than {M.CONTACT_CAP:.0%} of customers should be contacted in a month. Each limit gets a shadow price: "
                  "the profit a slot must beat to be used. A customer gets the bundle only if it beats their next-best "
                  "option by more than the bundle's price.", "build")
    sh = S.shadow
    t.tiles([
        {"label": "Bundle shadow price", "value": t.money(sh["Gear bundle"], 2), "accent": True, "note": f"{M.CAPACITY['Gear bundle']:,} in stock; {int((S.policies['Model, no limits'].action == 5).sum()):,} wanted one"},
        {"label": "Coupon budget", "value": "Not binding" if sh["10% coupon"] < .01 else t.money(sh["10% coupon"], 2), "note": "the model sends fewer coupons than allowed"},
        {"label": "Contact slot price", "value": t.money(sh["Contact slot"], 2), "note": f"keeps contact at {M.CONTACT_CAP:.0%}"},
        {"label": "Cost of the limits", "value": t.money(dr.loc["Model, no limits", "est"] - dr.loc[BEST, "est"], 2), "note": "per customer, doubly robust estimate"},
    ])
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        mix = S.mix[[RULES, "Model, no limits", BEST]]
        fig = go.Figure()
        for k, col in zip(mix.columns, [t.S2, t.S3, t.S1]):
            fig.add_bar(y=D.ACTIONS, x=mix[k], orientation="h", name=k, marker_color=col)
        fig.add_scatter(x=[M.CAPACITY["Gear bundle"]], y=["Gear bundle"], mode="markers", marker=dict(symbol="line-ns", size=26, color=t.INK, line=dict(width=2, color=t.INK)), name="Stock")
        fig.update_layout(barmode="group", xaxis_title="Customers", legend=dict(orientation="h", y=1.15), yaxis=dict(autorange="reversed"))
        t.chart(fig, 380)
    with b:
        u = S.uplift.iloc[::60]
        fig = go.Figure()
        fig.add_scatter(x=u.share, y=u.random, name="Random order", line=dict(color=t.BASE, dash="dot"))
        fig.add_scatter(x=u.share, y=u.model, name="Model's order", line=dict(color=t.S1, width=2.5), fill="tonexty", fillcolor="rgba(203,250,124,.08)")
        fig.update_layout(xaxis_title="Share of customers contacted, best first", yaxis_title="True profit gain vs no contact, $ a month",
                          xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.15))
        t.chart(fig, 380)
    u = S.uplift
    top = u.iloc[int(len(u) * .3)]
    ui.call(f"Contacting the model's top 30% captures {top.model / u.model.iloc[-1]:.0%} of the total gain. That curve "
            "is what makes a contact cap cheap: the last customers contacted add little, so trimming them costs "
            f"about {t.money(sh['Contact slot'], 2)} each and protects inbox attention for next month.")

# ===========================================================================
elif stage == "Judge before launch":
    ui.stage_head(4, "Judge before launch", "How much will the new policy earn, before anyone receives it?",
                  "Off-policy evaluation scores a policy on the log of a different one. On the held-out half, keep the "
                  "customers where the old log happened to match the new choice and weight each by one over its "
                  "propensity (IPS). Self-normalised IPS steadies the weights; doubly robust adds the response model's "
                  "prediction and corrects it with the weighted error, which keeps it honest if either part is off.",
                  "advisory")
    gain, glo, ghi = S.gain
    t.tiles([
        {"label": "Current rules (DR)", "value": t.money(dr.loc[RULES, "est"], 2), "note": "profit per customer a month"},
        {"label": "New policy (DR)", "value": t.money(dr.loc[SHIP, "est"], 2), "accent": True, "note": f"95% interval {t.money(dr.loc[SHIP, 'lo'], 2)} – {t.money(dr.loc[SHIP, 'hi'], 2)}"},
        {"label": "Estimated gain", "value": f"+{per_month(gain)}", "note": f"a month; interval {per_month(glo)} – {per_month(ghi)}"},
        {"label": "Truth (simulation)", "value": f"+{per_month(truth[SHIP] - truth[RULES])}", "note": "known only because the world is simulated"},
    ])
    order = [RULES, "One offer for all", "Model, no limits", BEST, SHIP]
    ests = ["Direct method", "IPS", "SNIPS", "Doubly robust"]
    ecol = {"Direct method": t.BASE, "IPS": t.S2, "SNIPS": t.S3, "Doubly robust": t.S1}
    fig = go.Figure()
    for j, e in enumerate(ests):
        d = OPE[OPE.estimator == e].set_index("policy").reindex(order)
        fig.add_scatter(x=d.est, y=[f"{p}" for p in order], mode="markers", name=e, marker=dict(color=ecol[e], size=10),
                        error_x=dict(type="data", symmetric=False, array=d.hi - d.est, arrayminus=d.est - d.lo, color=ecol[e], thickness=1.5),
                        offsetgroup=str(j))
    fig.add_scatter(x=truth.reindex(order), y=order, mode="markers", name="Truth", marker=dict(symbol="line-ns", size=24, color=t.INK, line=dict(width=3, color=t.INK)))
    fig.update_layout(scattermode="group", xaxis_title="Profit per customer a month, $", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.1))
    t.chart(fig, 420)
    tab = OPE[OPE.policy.isin(order)].copy()
    tab["error"] = tab.est - tab.truth
    err = tab.groupby("estimator").error.apply(lambda e: e.abs().mean()).reindex(ests)
    cov = tab.assign(c=(tab.lo <= tab.truth) & (tab.truth <= tab.hi)).groupby("estimator").c.mean().reindex(ests)
    st.dataframe(pd.DataFrame({"Mean |error| vs truth, $": err.round(2), "Intervals containing truth": [f"{v:.0%}" for v in cov],
                               "Uses propensities": ["No", "Yes", "Yes", "Yes"], "Uses the model": ["Yes", "No", "No", "Yes"]}),
                 width="stretch")
    ui.call(f"The direct method alone is confident and wrong: it inherits the model's blind spots and its intervals "
            f"miss the truth. Doubly robust, compared customer by customer against the current rules, estimates a gain "
            f"of {per_month(gain)} a month with an interval that {'excludes' if glo > 0 else 'includes'} zero, and that is the number the decision rests on. The truth line is a luxury of simulation; in "
            "production the A/B test in the last stage plays that role.")

# ===========================================================================
elif stage == "Guardrails":
    ui.stage_head(5, "Guardrails", "Is any group of customers worse off under the new policy?",
                  f"An average gain can hide a segment that loses. For each segment the doubly robust gain over the "
                  f"current rules is estimated on held-out data. A segment changes policy only if the estimate is at "
                  f"least {t.money(M.MIN_GAIN, 2)} per customer; otherwise it stays on the rules it has today, and "
                  "the limits are re-solved for everyone else.", "advisory")
    sg = S.segments
    fb = sg[sg.fallback]
    t.tiles([
        {"label": "Segments checked", "value": f"{len(sg)}", "note": "lapsing, loyalty, price-sensitive, core"},
        {"label": "Kept on current rules", "value": ", ".join(fb.segment) if len(fb) else "None", "accent": True, "note": f"{int(fb.customers.sum()):,} customers" if len(fb) else ""},
        {"label": "Contact rate", "value": f"{(S.policies[SHIP].action > 0).mean():.0%}", "note": "was 100% under the rules"},
        {"label": "Discount share", "value": f"{(S.policies[SHIP].action == 1).mean():.0%}", "note": f"rules: {(S.policies[RULES].action == 1).mean():.0%}"},
    ])
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        fig = go.Figure()
        cols = [t.S2 if f else t.S1 for f in sg.fallback]
        fig.add_scatter(x=sg.dr_diff, y=sg.segment, mode="markers", marker=dict(color=cols, size=12), name="DR gain",
                        error_x=dict(type="data", symmetric=False, array=sg.dr_hi - sg.dr_diff, arrayminus=sg.dr_diff - sg.dr_lo, color=t.MUTED))
        fig.add_scatter(x=sg.true_diff, y=sg.segment, mode="markers", name="Truth", marker=dict(symbol="line-ns", size=20, color=t.INK, line=dict(width=3, color=t.INK)))
        fig.add_vline(x=0, line=dict(color=t.BASE, dash="dot"))
        fig.add_vline(x=M.MIN_GAIN, line=dict(color=t.S2, dash="dash"), annotation_text="bar to change", annotation_font_color=t.S2)
        fig.update_layout(xaxis_title="Gain over current rules, $ per customer a month", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.15))
        t.chart(fig, 330)
    with b:
        st.dataframe(pd.DataFrame({
            "Segment": sg.segment,
            "DR gain": sg.dr_diff.map(lambda v: f"{v:+.2f}"), "Interval": [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(sg.dr_lo, sg.dr_hi)],
            "Decision": ["Keep rules" if f else "Change" for f in sg.fallback],
        }), hide_index=True, width="stretch")
    if len(fb):
        s0 = fb.iloc[0]
        ui.call(f"{s0.segment} customers show no evidence of gain ({s0.dr_diff:+.2f}, interval {s0.dr_lo:+.2f} to "
                f"{s0.dr_hi:+.2f}), so they keep today's treatment. In the simulation the model would have cost them "
                f"{t.money(abs(s0.true_diff), 2)} each: the rule caught a real, small harm. The price is "
                f"{t.money(truth[BEST] - truth[SHIP], 2)} per customer of true gain given up elsewhere, paid for a "
                "change that can be defended segment by segment.")

# ===========================================================================
elif stage == "Roll out":
    ui.stage_head(6, "Roll out", "How does the new policy go live without betting the month on an estimate?",
                  "Off-policy evaluation decides whether a test is worth running; a randomised test decides whether "
                  "to ship. The new policy runs on a share of customers against the current rules, with the same "
                  "exploration kept on so next month's log can train the next model.", "build")
    gain = S.gain[0]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        mde = st.slider("Smallest gain worth detecting, $ per customer", .5, 4.0, float(round(max(.5, min(4.0, gain / 2)) * 2) / 2), .25, key="nba_mde")
    with c2:
        share = st.slider("Share of customers in the test arm", .05, .5, .2, .05, key="nba_share")
    n_arm = M.sample_size(S.profit_sd, mde)
    months = n_arm / (N * share)
    t.tiles([
        {"label": "Customers per arm", "value": f"{n_arm:,}", "note": f"5% test, 80% power, sd {t.money(S.profit_sd, 0)}"},
        {"label": "Test arm size", "value": f"{int(N * share):,}", "accent": True, "note": f"{share:.0%} of the base each month"},
        {"label": "Time to read", "value": f"{max(1, int(np.ceil(months)))} month{'s' if np.ceil(months) > 1 else ''}", "note": "at one decision per customer a month"},
        {"label": "Expected gain at stake", "value": per_month(gain), "note": "a month, doubly robust"},
    ])
    ui.cards([
        ("Week 0 · shadow mode", "The policy scores every customer and logs its choice, but the rules still decide. Checks: no action outside stock, contact rate at cap, no segment over its discount share."),
        (f"Month 1 · {share:.0%} test", "Random assignment by customer id, stratified by segment. Primary metric: profit per customer. Guardrails: unsubscribe rate, discount cost, returns."),
        ("Month 2 · read and widen", "Ship if the test confirms a gain above the bar; widen to 50%, then 100%. If the test disagrees with the estimate, the model is retrained before anything widens."),
        ("Always · keep exploring", f"{D.EPS:.0%} of decisions stay random with propensities logged. Without that, next month's log could only evaluate the policy that wrote it."),
    ])
    ui.call("Two numbers can disagree here: the off-policy estimate and the test. The test wins. What the estimate "
            "buys is the right to run the test at all, sized correctly, with a clear bar for shipping.")

st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.nba_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Fernwood Outfitters is fictional. Customers, offers, costs and responses are synthetic; the true effect "
           "of every action is known only because the world is simulated, and is used to grade the estimates.")
