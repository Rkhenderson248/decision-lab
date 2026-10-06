"""Subscriber value lab: one fictional subscription business, from subscriber trends to an executive brief."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG

from lab import bench_ui as BU
from lab import theme as t
from lab import widgets as w
from lab.cu import ui as cui
from lab.sub import data as D
from lab.sub import models as M
from lab.sub import ui

cui.css()
STAGES = ["Trends", "Value", "Segment", "Churn", "Treat", "Price", "Test", "Care", "Govern"]

t.header(
    f"Featured project · {D.NAME} (fictional)",
    "Customer value, retention and pricing for a subscription business",
    "Sixty thousand synthetic subscribers to a home internet and mobile provider, consumer and small business, followed "
    "through nine connected decisions: what the trends say, what each customer is worth, who will leave, who an offer "
    "actually saves, how far a price can rise, how to test it, what care calls reveal and how it all reaches an executive.",
)

company = D.company()
subs = company.subs
base = M.base_today()

q = str(st.query_params.get("stage", "")).lower()
lookup = {s.lower(): s for s in STAGES}
if "sv_stage" not in st.session_state:
    st.session_state.sv_stage = lookup.get(q, STAGES[0])
    st.session_state.sv_stage__last = st.session_state.sv_stage


@st.cache_data(show_spinner=False)
def featured() -> dict:
    b = base
    picks = {}

    def pick(mask, label, by=None, asc=False):
        c = b[mask]
        if len(c):
            r = c.sort_values(by, ascending=asc).iloc[min(3, len(c) - 1)] if by else c.iloc[len(c) // 2]
            picks[f"{r.sub_id} · {label}"] = r.sub_id

    pick((b.segment == "Promo switchers") & (b.uplift > 0.05), "promo customer worth saving", "uplift")
    pick((b.segment == "Streamers & gamers") & (b.tier == "300 Mbps") & (b.usage_gb > 1200), "heavy user on the entry plan", "usage_gb")
    pick((b.segment == "Small business") & (b.outage_hrs_90d > 8), "business with outages", "clv")
    pick((b.segment == "Light users") & (b.uplift < -0.02), "light user best left alone", "uplift", True)
    pick((b.segment == "Bundled households"), "high-value bundle", "clv")
    pick((b.segment == "Movers & renters") & (b.churn_6m > 0.2), "renter likely to move", "churn_6m")
    return picks


picks = featured()
top_l, top_r = st.columns([1.4, 2.6], gap="large")
with top_l:
    label = st.selectbox("Follow a subscriber through every stage", list(picks), key="sv_sub_label")
sub_id = picks[label]
me = base.set_index("sub_id").loc[sub_id]
with top_r:
    st.markdown(
        f'<div style="padding-top:1.9rem">{ui.chip(me.segment)} &nbsp;·&nbsp; {t.esc(me.tier)} · ${me.fee:,.0f}/month · '
        f'{int(me.mobile_lines)} mobile lines · {int(me.tenure_today)} months · {"business" if me.b2b else "consumer"} · {t.esc(me.region)}</div>',
        unsafe_allow_html=True)

stage = w.segmented("Stage", STAGES, key="sv_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower()
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")
ORDER = M.SEGMENT_ORDER
COL = M.SEG_COLORS


def seg_bar(series: pd.Series, title: str, xfmt: dict, height: int = 300, hover: str = "%{y}: %{x}"):
    s_ = series.reindex([o for o in ORDER if o in series.index])
    fig = go.Figure(go.Bar(y=s_.index, x=s_.values, orientation="h", marker=dict(color=[COL[i] for i in s_.index], cornerradius=3),
                           hovertemplate=hover + "<extra></extra>"))
    fig.update_layout(title=title, xaxis=xfmt, yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
    t.chart(fig, height=height)


# ===========================================================================
if stage == "Trends":
    cui.stage_head(1, "Subscriber trends", "What is happening to the base?",
                   "Net adds, cohort retention and churn by month of tenure. Before any model, the shape of churn says "
                   "where to look: it is a calendar with spikes, not a constant drip.", "")
    mo = company.monthly
    last3 = mo["churn_rate"].tail(3).mean()
    net12 = int((mo["starts"] - mo["churners"]).tail(12).sum())
    t.tiles([
        {"label": "Active subscribers", "value": f"{len(base):,}", "note": f"{base['b2b'].mean():.0%} small business"},
        {"label": "Monthly churn", "value": t.pct(last3, 2), "accent": True, "note": "average, last three months"},
        {"label": "Net adds, last 12 months", "value": f"{net12:+,}", "note": f"{int(mo['starts'].tail(12).sum()):,} joined · {int(mo['churners'].tail(12).sum()):,} left"},
        {"label": "Monthly recurring revenue", "value": t.money(mo["revenue"].iloc[-1]), "accent": True, "note": f"average ${base['fee'].mean():,.0f} per subscriber"},
    ])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mo["month"], y=mo["starts"], name="Joined", marker=dict(color=t.S1)))
        fig.add_trace(go.Bar(x=mo["month"], y=-mo["churners"], name="Left", marker=dict(color=t.S2), customdata=mo["churners"],
                             hovertemplate="Month %{x}: %{customdata:,} left<extra></extra>"))
        fig.add_trace(go.Scatter(x=mo["month"], y=mo["starts"] - mo["churners"], name="Net adds", mode="lines", line=dict(color=t.INK, width=2)))
        for m_, txt in ((D.PRICE_TEST_MONTH, "price test"), (D.CAMPAIGN_MONTH, "retention campaign")):
            fig.add_vline(x=m_, line=dict(color=t.GRID, width=1, dash="dot"))
            fig.add_annotation(x=m_, y=1, yref="paper", text=" " + txt, showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color=t.MUTED))
        fig.update_layout(title="Joined, left and net adds by month", barmode="relative", xaxis_title="Month", yaxis_title="Subscribers",
                          legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
    with right:
        cc = M.cohort_curves()
        fig2 = go.Figure()
        names_ = {"Q1": "Joined months 0–2", "Q3": "Joined months 6–8", "Q5": "Joined months 12–14", "Q7": "Joined months 18–20"}
        for c_, color in zip(names_, ["#0F4640", "#2E8F80", "#6FBBAA", "#A9D9CD"]):
            g = cc[cc["cohort"] == c_]
            fig2.add_trace(go.Scatter(x=g["month"], y=g["retained"] * 100, name=names_[c_],
                                      line=dict(color=color, width=2.2), hovertemplate="Month %{x}: %{y:.1f}% retained<extra></extra>"))
        fig2.update_layout(title="Cohort retention: share still subscribed", xaxis_title="Months since joining", yaxis_title="Retained (%)",
                           legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=11)), margin=dict(t=110))
        t.chart(fig2, height=360)
    hz = M.hazard_by_tenure()
    fig3 = go.Figure()
    for lab_, color in (("No contract", t.S3), ("12-month contract", t.S1), ("24-month contract", t.S2)):
        g = hz[hz["contract"] == lab_]
        fig3.add_trace(go.Scatter(x=g["tenure"], y=g["hazard"] * 100, name=lab_, mode="lines+markers", marker=dict(size=4),
                                  line=dict(color=color, width=2), hovertemplate="Month %{x}: %{y:.2f}% left<extra>" + lab_ + "</extra>"))
    g12 = hz[hz["contract"] == "12-month contract"]
    pk = g12.loc[g12["hazard"].idxmax()]
    fig3.add_annotation(x=pk["tenure"], y=pk["hazard"] * 100, text=f"Contract end: {pk['hazard']:.0%}",
                        showarrow=True, ax=90, ay=-6, xanchor="left", font=dict(size=12, color="#0E1311"), bgcolor="rgba(255,255,255,.9)", bordercolor="#E1E4DE", borderwidth=1, borderpad=4, arrowcolor="#4A534D", arrowwidth=1, arrowhead=0)
    fig3.update_layout(title="Monthly churn rate by month of tenure: the contract-end spikes", xaxis_title="Month of tenure",
                       yaxis_title="Left that month (%)", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
    t.chart(fig3, height=330)
    ch = (subs.assign(segment=subs["sub_id"].map(M.segments().labels).to_numpy())
          .pivot_table(index="segment", columns="channel", values="churned", aggfunc="mean").reindex(ORDER)[D.CHANNELS])
    with st.expander("Churn by segment and acquisition channel (share of everyone ever acquired who has left)"):
        st.dataframe(ch.style.format("{:.0%}"), width="stretch")
    ui.note(f"{t.esc(sub_id)} joined {int(me.tenure_today)} months ago on " +
            (f"a {int(me.contract)}-month contract" if me.contract else "no contract") +
            (" with a promotional price" if me.promo else "") + ". " +
            (f"Their contract ends in {int(me.contract - me.tenure_today)} months, right where the churn spike sits."
             if me.contract and 0 <= me.contract - me.tenure_today <= 6 else "Every later stage reads this history."))
    t.insight("Churn is concentrated in two places: the first months of tenure and the month a contract or promotion ends, when "
              "the bill changes and customers shop. A flat churn rate hides both. Retention spend and price changes should be "
              "timed against this calendar.")
    cui.call("The weekly report shows churn by tenure month and contract end, not one blended rate. A blended rate can improve "
             "simply because the base got older, while the thing that matters, the contract-end spike, got worse.")

# ===========================================================================
elif stage == "Value":
    cui.stage_head(2, "Customer lifetime value", "What is each customer worth?",
                   "A discrete-time survival model estimates every subscriber's monthly chance of leaving from tenure, "
                   "contract position, plan, usage, care and payment history. Lifetime value is the expected margin over "
                   "the next five years, discounted, given that chance.", "")
    total = base["clv"].sum()
    srt = np.sort(base["clv"].to_numpy())[::-1]
    top20 = srt[: int(len(srt) * 0.2)].sum() / total
    acq = subs.assign(segment=subs["sub_id"].map(M.segments().labels).to_numpy())
    # CLV at acquisition by channel: lifetime value for this year's joiners, against what they cost to acquire.
    recent = base[base["start_month"] >= 24]
    cac = recent.groupby("channel").agg(clv=("clv", "mean"), cac=("cac", "mean"), n=("sub_id", "size")).reindex(D.CHANNELS)
    cac["ratio"] = cac["clv"] / cac["cac"]
    t.tiles([
        {"label": "Lifetime value of the base", "value": t.money(total), "accent": True, "note": "five-year discounted margin"},
        {"label": "Average per subscriber", "value": t.money(base["clv"].mean()), "note": f"{base['exp_months'].mean():.0f} expected months of the next 60"},
        {"label": "Value held by the top 20%", "value": t.pct(top20), "accent": True, "note": "of all lifetime value"},
        {"label": "Best value-to-cost channel", "value": cac["ratio"].idxmax(), "note": f"{cac['ratio'].max():.1f}× lifetime value per $ of acquisition"},
    ])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        fig = go.Figure()
        for s_ in ORDER:
            d_ = base[base["segment"] == s_]
            fig.add_trace(go.Box(x=d_["clv"], name=s_, marker_color=COL[s_], boxpoints=False, line=dict(width=1.5)))
        fig.update_layout(title="Lifetime value by segment", xaxis=dict(tickprefix="$", tickformat="~s", title="Five-year discounted margin"),
                          showlegend=False, yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig, height=380)
    with right:
        fig2 = go.Figure(go.Bar(x=cac.index, y=cac["ratio"], marker=dict(color=t.S1, cornerradius=3), text=[f"{v:.1f}×" for v in cac["ratio"]],
                                textposition="outside", customdata=cac[["clv", "cac"]],
                                hovertemplate="%{x}: lifetime value $%{customdata[0]:,.0f} vs cost $%{customdata[1]:,.0f}<extra></extra>"))
        fig2.add_hline(y=3, line=dict(color=t.S2, width=1, dash="dot"))
        fig2.add_annotation(x=0, xref="paper", y=3, text=" 3× rule of thumb", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=t.S2))
        fig2.update_layout(title="Lifetime value per $1 of acquisition cost, this year's joiners", yaxis_title="×", yaxis=dict(range=[0, cac["ratio"].max() * 1.25]))
        t.chart(fig2, height=380)
    BU.subhead("Upgrades", "Heavy users on the entry plan are the clearest upgrade opportunity. Their value with and without a move "
               "to 1 Gig: fifteen dollars more a month and a lower churn hazard, the effect measured in a past upgrade pilot.")
    upv = M.upgrade_value()
    a, b_ = st.columns([1, 1.3], gap="large")
    with a:
        take = st.slider("Share of candidates who accept an upgrade offer", 5, 60, 20, 5, format="%d%%", key="sv_up_take") / 100
        t.tiles([
            {"label": "Upgrade candidates", "value": f"{len(upv):,}", "note": "over 900 GB a month on 300 Mbps"},
            {"label": "Value gained per upgrade", "value": t.money(upv["gain"].mean()), "accent": True, "note": "lifetime value, upgraded minus not"},
            {"label": "Value at your take rate", "value": t.money(upv["gain"].sum() * take), "accent": True, "note": f"{take:.0%} of candidates"},
        ])
    with b_:
        g = upv.groupby("segment")["gain"].agg(["sum", "size"]).reindex(ORDER).dropna()
        seg_bar(g["sum"] * take, "Upgrade value by segment, at your take rate", dict(tickprefix="$", tickformat="~s"), height=330,
                hover="%{y}: %{x:$,.0f}")
    ui.note(f"{t.esc(sub_id)} is worth <b>{t.money(me.clv)}</b> over five years: ${me.margin:,.0f} margin a month for an expected "
            f"{me.exp_months:.0f} more months. " + ("They are an upgrade candidate." if sub_id in set(upv["sub_id"]) else ""))
    t.insight(f"Value is concentrated: the top fifth of subscribers carry <b>{top20:.0%}</b> of it, mostly bundled households and small "
              "businesses. Light users are profitable but small. Partner channels bring customers in at the highest cost, which "
              "only pays off where those customers stay.")
    cui.call("Lifetime value is the currency every later stage spends: who gets a retention offer, who is protected from a price "
             "increase, which channel gets acquisition budget. One definition, owned by Finance and Analytics together, beats "
             "five versions in five decks.")
    with st.expander("Method notes"):
        st.markdown(f"""
- **Survival model:** logistic regression on subscriber-months 0–29 (every churn month kept, quiet months sampled at 8% and re-weighted). Inputs: tenure, months to contract end, promotion end, plan, fee, lines, usage, autopay, care contacts, outages, late payments, last call reason.
- **Lifetime value:** for each active subscriber, monthly survival is projected 60 months ahead from today's tenure; value is survival × (fee × {D.MARGIN:.0%} margin − care cost), discounted at 10% a year.
- **Acquisition cost:** channel cost including install, 40% higher for business accounts.
- **In production:** margins by product from Finance, survival refit monthly, value bands published to CRM so every team uses the same number.
""")

# ===========================================================================
elif stage == "Segment":
    cui.stage_head(3, "Segment", "Who are our customers, and what does each group need?",
                   "K-means on plan, usage, life stage, contract, care and payment behavior, named in business language. Then "
                   "the matrix that turns segments into action: lifetime value against churn risk.", "")
    S = M.segments()
    prof = S.profile.reindex(ORDER)
    t.tiles([
        {"label": "Segments", "value": "6", "note": f"on {len(M.SEG_FEATURES)} observable features"},
        {"label": "Recovered the true structure", "value": f"{S.ari:.2f}", "accent": True, "note": "agreement with the hidden personas (1 = perfect)"},
        {"label": "Largest segment", "value": base["segment"].value_counts().idxmax(), "note": f"{base['segment'].value_counts().max() / len(base):.0%} of active subscribers"},
        {"label": "Highest value", "value": base.groupby("segment")["clv"].sum().idxmax(), "accent": True, "note": "share of total lifetime value"},
    ])
    cards = []
    val = base.groupby("segment").agg(clv=("clv", "mean"), risk=("churn_6m", "mean"), n=("sub_id", "size"))
    for s_ in ORDER:
        r = prof.loc[s_]
        cls = "lab-card is-member" if s_ == me.segment else "lab-card"
        cards.append(f'<div class="{cls}"><h4><i style="background:{COL[s_]}"></i>{t.esc(s_)}</h4><dl>'
                     f'<dt>Active subscribers</dt><dd>{int(val.loc[s_, "n"]):,}</dd>'
                     f'<dt>Monthly fee · lines</dt><dd>${r.fee:,.0f} · {r.lines:.1f}</dd>'
                     f'<dt>Median usage · age</dt><dd>{r.usage:,.0f} GB · {r.age:.0f}</dd>'
                     f'<dt>No contract</dt><dd>{r.no_contract:.0%}</dd>'
                     f'<dt>Lifetime value</dt><dd>{t.money(val.loc[s_, "clv"])}</dd>'
                     f'<dt>6-month churn risk</dt><dd>{val.loc[s_, "risk"]:.1%}</dd></dl></div>')
    st.markdown(f'<div class="cu-seg">{"".join(cards)}</div>', unsafe_allow_html=True)
    hi_v = base["clv"] >= base["clv"].median()
    hi_r = base["churn_6m"] >= base["churn_6m"].quantile(0.75)
    quad = np.select([hi_v & hi_r, hi_v & ~hi_r, ~hi_v & hi_r], ["Protect", "Grow", "Serve efficiently"], "Maintain")
    b2 = base.assign(quadrant=quad)
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        qd = b2.groupby("quadrant").agg(n=("sub_id", "size"), clv=("clv", "sum"), risk=("churn_6m", "mean"))
        boxes = {"Protect": ("High value · high risk", "Proactive outreach and retention offers where uplift is positive."),
                 "Grow": ("High value · low risk", "Upgrades, added lines, loyalty recognition. No discounts."),
                 "Serve efficiently": ("Lower value · high risk", "Digital self-service and fixes; offers only where they pay back."),
                 "Maintain": ("Lower value · low risk", "Standard journeys. Leave well alone.")}
        html = ""
        for k_ in ["Protect", "Grow", "Serve efficiently", "Maintain"]:
            html += (f'<div class="lab-card"><h4>{k_}</h4><p style="margin:.1rem 0 .4rem;color:#6B756E;font-size:.85rem">{boxes[k_][0]}</p>'
                     f'<dl><dt>Subscribers</dt><dd>{int(qd.loc[k_, "n"]):,}</dd><dt>Lifetime value</dt><dd>{t.money(qd.loc[k_, "clv"])}</dd>'
                     f'<dt>Average risk</dt><dd>{qd.loc[k_, "risk"]:.1%}</dd></dl><p style="font-size:.86rem;margin:.5rem 0 0">{boxes[k_][1]}</p></div>')
        st.markdown(f'<div class="cu-seg" style="grid-template-columns:repeat(2,minmax(0,1fr))">{html}</div>', unsafe_allow_html=True)
    with right:
        mix = pd.crosstab(b2["segment"], b2["quadrant"], normalize="index").reindex(ORDER)[["Protect", "Grow", "Serve efficiently", "Maintain"]]
        fig = go.Figure(go.Heatmap(z=mix.to_numpy() * 100, x=mix.columns, y=mix.index, colorscale=[[0, "#F1F2EE"], [1, "#0F4640"]],
                                   text=[[f"{v * 100:.0f}%" for v in r] for r in mix.to_numpy()], texttemplate="%{text}", showscale=False,
                                   hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>", xgap=2, ygap=2))
        fig.update_layout(title="Where each segment sits in the value–risk matrix", yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig, height=380)
    me_q = b2.set_index("sub_id").loc[sub_id, "quadrant"]
    ui.note(f"{t.esc(sub_id)} is in {ui.chip(me.segment)} and the <b>{me_q}</b> quadrant: {boxes[me_q][1][0].lower() + boxes[me_q][1][1:]}")
    t.insight("Segments say who customers are; the value–risk matrix says what to do this quarter. The same segment splits across "
              "quadrants, which is why personalization runs on both: segment for the message, quadrant for the investment.")
    cui.call("Six segments, agreed with Marketing and Care, and frozen for a year. Re-clustering every month would make every trend "
             "chart a moving target. The matrix, by contrast, refreshes weekly, because value and risk change faster than identity.")

# ===========================================================================
elif stage == "Churn":
    ch = M.churn()
    cui.stage_head(4, "Churn propensity", "Who is about to leave?",
                   "A churn model scores every subscriber's chance of leaving in the next six months. It is trained on "
                   "month 24 and tested on month 30, so the test is a genuinely later period, and the champion is chosen on a "
                   "bench of five algorithms.", "")
    t.tiles([
        {"label": "Model accuracy (AUC)", "value": f"{ch.auc:.3f}", "accent": True, "note": "next period, held out"},
        {"label": "Leavers in the riskiest 10%", "value": t.pct(ch.deciles["capture"].iloc[0]), "accent": True, "note": "of everyone who left"},
        {"label": "Average 6-month risk", "value": t.pct(base["churn_6m"].mean(), 1), "note": "active subscribers today"},
        {"label": "Above 20% risk", "value": f"{(base['churn_6m'] > 0.2).sum():,}", "note": "subscribers today"},
    ])
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        dc = ch.deciles
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dc["decile"], y=dc["actual"] * 100, name="Actual", marker=dict(color=t.S1, cornerradius=3)))
        fig.add_trace(go.Scatter(x=dc["decile"], y=dc["predicted"] * 100, name="Predicted", mode="lines+markers", line=dict(color=t.INK, width=2)))
        fig.update_layout(title="Six-month churn by risk decile (1 = riskiest)", xaxis=dict(title="Risk decile", dtick=1), yaxis_title="Churned (%)",
                          legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
    with right:
        dr = ch.drivers.head(10).iloc[::-1]
        fig2 = go.Figure(go.Bar(y=dr["label"], x=dr["importance"], orientation="h", marker=dict(color=t.S3, cornerradius=3),
                                hovertemplate="%{y}: %{x:.3f} AUC lost when shuffled<extra></extra>"))
        fig2.update_layout(title="What drives the score (permutation importance)", xaxis_title="AUC lost when the feature is shuffled", margin=dict(l=8, t=70))
        t.chart(fig2, height=360)
    BU.subhead("Model bench · five algorithms, one decision",
               "The same subscribers and the same later test period for linear regression, logistic regression, random forest, "
               "XGBoost and a support vector machine. Value assumes the riskiest 20% are contacted, an offer saves 30% of those who "
               "would have left, each save is worth $1,200 and each contact costs $25.")
    BU.render(M.churn_bench(), key="sv_bench", regulated_default=False, value_label="Value from contacting the top 20%",
              decision="leavers", regulated_help="Retention targeting is not a regulated decision; switch on to see what changes if it were.")
    ui.note(f"{t.esc(sub_id)} has a <b>{me.churn_6m:.0%}</b> chance of leaving in the next six months, "
            f"{'well above' if me.churn_6m > 2 * base['churn_6m'].mean() else 'around' if me.churn_6m > 0.5 * base['churn_6m'].mean() else 'well below'} "
            f"the average of {base['churn_6m'].mean():.1%}.")
    t.insight("Risk is concentrated enough to act on: the riskiest tenth of subscribers holds a large share of next period's leavers. "
              "Tenure and contract timing lead, then care contacts and outages, which are things the company controls. But a risk "
              "score says who will leave, not who an offer would keep. That is the next stage.")
    cui.call("XGBoost is the champion here and logistic regression would be in lending. Retention targeting is not a regulated "
             "decision and a wrong call costs a discount, not a customer's access to credit. The algorithm follows the decision.")

# ===========================================================================
elif stage == "Treat":
    up = M.uplift()
    cp = up.test
    cui.stage_head(5, "Retention treatment", "Who does an offer actually save?",
                   "Last quarter's retention campaign randomly offered half its target list $15 a month off for six months. An "
                   "uplift model learns from that split who the offer changes: some customers stay anyway, some leave anyway, "
                   "and for some the contact itself prompts them to shop.", "")
    with st.container(border=True):
        a, b_ = st.columns(2)
        budget = a.slider("Offers you can make this quarter", 1000, 15000, 5000, 500, key="sv_budget")
        rank_by = b_.radio("Rank customers by", ["Value saved (uplift × lifetime value)", "Uplift (churners saved)", "Churn risk"],
                           key="sv_rank", horizontal=False)
    pool = base.copy()
    pool["value_uplift"] = pool["uplift"] * pool["clv"] - D.OFFER_COST
    col_ = {"Value saved (uplift × lifetime value)": "value_uplift", "Uplift (churners saved)": "uplift", "Churn risk": "churn_6m"}[rank_by]
    chosen = pool.nlargest(budget, col_)

    def outcome(df):
        return df["uplift"].sum(), (df["uplift"] * df["clv"]).sum() - D.OFFER_COST * len(df)

    saves, net = outcome(chosen)
    r_saves, r_net = outcome(pool.nlargest(budget, "churn_6m"))
    dogs = int((pool["uplift"] < -0.01).sum())
    t.tiles([
        {"label": "Churners saved", "value": f"{saves:,.0f}", "accent": True, "note": f"expected, from {budget:,} offers"},
        {"label": "Net value", "value": t.money(net), "accent": True, "note": f"after ${D.OFFER_COST:.0f} per offer"},
        {"label": "Same budget on the riskiest", "value": t.money(r_net), "note": f"{r_saves:,.0f} saved; the usual approach"},
        {"label": "Better left alone", "value": f"{dogs:,}", "note": "an offer makes them more likely to leave"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure()
        n_c = len(cp)
        for col, name, color in (("uplift", "Uplift model", t.S1), ("risk", "Churn risk", t.S2), ("uplift_boosted", "Uplift, boosted (lost)", t.S3)):
            qn = M.qini(cp, col)
            fig.add_trace(go.Scatter(x=qn["share"] * n_c, y=qn["saved"], name=name, line=dict(color=color, width=2.5 if col == "uplift" else 1.8)))
        qr = M.qini(cp, "uplift")
        fig.add_trace(go.Scatter(x=[0, n_c], y=[0, qr["saved"].iloc[-1]], name="Random", line=dict(color=t.BASE, width=1.5, dash="dot")))
        fig.update_layout(title="Churners saved by number contacted, last quarter's campaign (Qini)", xaxis_title="Customers offered",
                          yaxis_title="Churners saved vs no offer", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=380)
    with right:
        g = pool.groupby("segment")["uplift"].mean() * 100
        g = g.reindex(ORDER)
        fig2 = go.Figure(go.Bar(y=g.index, x=g.values, orientation="h", marker=dict(color=[t.S1 if v > 0 else t.S2 for v in g.values], cornerradius=3),
                                hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
        fig2.add_vline(x=0, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Offer effect on 6-month churn", xaxis_title="Churn avoided (pts)",
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=380)
    st.markdown("**This quarter's offer list, top 12**")
    h12 = chosen.head(12)
    show = pd.DataFrame({"Subscriber": h12["sub_id"], "Segment": h12["segment"], "Fee": h12["fee"].map(lambda v: f"${v:,.0f}"),
                         "Lifetime value": h12["clv"].map(lambda v: f"${v:,.0f}"), "Risk": h12["churn_6m"].map(lambda v: f"{v:.0%}"),
                         "Churn avoided": h12["uplift"].map(lambda v: f"{v * 100:+.1f} pts"),
                         "Expected value of offer": h12["value_uplift"].map(lambda v: f"${v:,.0f}")})
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("In production this list flows to the CRM (for example Salesforce Marketing Cloud) as a daily audience, with a 10% random hold-out kept out of it.")
    ui.note(f"An offer changes {t.esc(sub_id)}'s six-month churn by <b>{-me.uplift * 100:+.1f} pts</b>"
            + (" in the right direction: they are worth contacting." if me.uplift > 0.01 else
               ": the contact would do more harm than good, so they stay off the list." if me.uplift < -0.01 else
               ": almost nothing. They would make the same choice either way."))
    t.insight("Targeting by risk sends offers to renters who are moving anyway and to light users for whom the call prompts a plan "
              "review. Targeting by uplift sends them to promotion customers at the end of their deal, where the offer changes the "
              "outcome. Weighting by lifetime value then spends the budget where each save is worth most.")
    cui.call(f"The simpler uplift model won: logistic regression with treatment interactions beat a boosted two-model learner on "
             f"held-out campaign rows (Qini area {up.qini_area['Uplift (logistic, interactions)']:.0f} vs "
             f"{up.qini_area['Uplift (boosted, two models)']:.0f}). With a 6% outcome rate, stability beats flexibility, and every campaign "
             "keeps a random hold-out so the model can be retrained.")

# ===========================================================================
elif stage == "Price":
    cui.stage_head(6, "Price changes", "How far can prices rise, and for whom?",
                   "Eighteen months ago half the base received a randomized increase of $0, $3, $5 or $8. A response model "
                   "learned from that test says how each segment reacts. A price increase earns new revenue and loses some "
                   "customers' lifetime value; commercial guardrails decide who is exempt.", "")
    grid = M.best_increase_by_segment()
    best = grid.loc[grid.groupby("segment")["net"].idxmax()].set_index("segment")["increase"]
    with st.container(border=True):
        a, b_, c_ = st.columns([1.1, 1, 1])
        mode = a.radio("Increase", ["Same for everyone", "Best by segment"], key="sv_pmode", horizontal=True)
        flat = a.slider("Monthly increase", 0, 10, 5, 1, format="$%d", key="sv_inc", disabled=mode != "Same for everyone")
        cap = b_.slider("Cap on any increase", 0, 10, 8, 1, format="$%d", key="sv_cap")
        ex_risk = b_.toggle("Exempt the riskiest 10%", value=True, key="sv_ex_risk")
        ex_win = c_.toggle("Exempt near a contract or promotion end", value=True, key="sv_ex_win")
        prot = c_.toggle("Protect high-value, at-risk customers", value=True, key="sv_prot")
    inc_by = {s_: (flat if mode == "Same for everyone" else int(best.get(s_, 0))) for s_ in ORDER}
    plan = M.plan_value(inc_by, cap, 0.10 if ex_risk else None, ex_win, prot)
    noguard = M.plan_value(inc_by, 10, None, False, False)
    t.tiles([
        {"label": "Net value, 12 months", "value": t.money(plan["net"].sum()), "accent": True, "note": "new revenue minus lifetime value lost"},
        {"label": "New revenue", "value": t.money(plan["revenue_12m"].sum()), "note": f"{(plan['increase'] > 0).mean():.0%} of subscribers see an increase"},
        {"label": "Extra customers lost", "value": f"{plan['extra_churn'].sum():,.0f}", "note": "within six months of the change"},
        {"label": "Guardrails changed net value by", "value": t.money(plan["net"].sum() - noguard["net"].sum()), "accent": True,
         "note": f"{plan['exempt'].mean():.0%} of subscribers exempt"},
    ])
    fig = go.Figure()
    for s_ in ORDER:
        g = grid[grid["segment"] == s_]
        fig.add_trace(go.Scatter(x=g["increase"], y=g["net"], name=s_, line=dict(color=COL[s_], width=2),
                                 hovertemplate="$%{x}: %{y:$,.0f}<extra>" + s_ + "</extra>"))
    fig.add_hline(y=0, line=dict(color=t.GRID, width=1))
    fig.update_layout(title="Net 12-month value of an increase, by segment (no guardrails)", xaxis=dict(title="Monthly increase ($)", dtick=1),
                      yaxis=dict(tickprefix="$", tickformat="~s"), legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=11)), margin=dict(t=120))
    t.chart(fig, height=420)
    tb = plan.groupby("segment").agg(increase=("increase", "max"), share=("increase", lambda v: (v > 0).mean()),
                                     extra=("extra_churn", "sum"), revenue=("revenue_12m", "sum"), lost=("value_lost", "sum"), net=("net", "sum")).reindex(ORDER)
    st.markdown("**The plan, by segment**")
    tshow = pd.DataFrame({"Segment": tb.index, "Increase": tb["increase"].map(lambda v: f"${v:.0f}"), "Raised": tb["share"].map(lambda v: f"{v:.0%}"),
                          "Extra lost": tb["extra"].map(lambda v: f"{v:,.0f}"), "Revenue": tb["revenue"].map(t.money),
                          "Value lost": tb["lost"].map(t.money), "Net": tb["net"].map(t.money)})
    st.dataframe(tshow, hide_index=True, width="stretch")
    pt = company.price_test
    st.caption(f"Learned from {len(pt):,} subscribers in the randomized test: six-month churn was "
               + ", ".join(f"{v:.1%} at \\${int(k)}" for k, v in pt.groupby('increase')['churned_6m'].mean().items()) + ".")
    me_inc = int(plan["increase"].iloc[int(np.flatnonzero(base["sub_id"].to_numpy() == sub_id)[0])])
    ui.note(f"Under this plan {t.esc(sub_id)} " + (f"sees a <b>${me_inc}</b> monthly increase." if me_inc else "is <b>exempt</b> or not raised.")
            + f" Their lifetime value is {t.money(me.clv)} and their six-month risk {me.churn_6m:.0%}.")
    t.insight("Bundled households and light users absorb increases; promotion customers and renters do not, and raising them costs "
              "more lifetime value than it earns. Guardrails trade a little revenue for a lot of protection: exempting customers "
              "near a contract end avoids handing them a reason to shop at the exact moment they are free to leave.")
    cui.call("A price increase is a retention decision as much as a revenue one. Finance sees the revenue line; the lifetime value "
             "lost shows up quarters later in churn. Putting both on one chart, with guardrails agreed in advance, is what lets "
             "Pricing, Finance and Care sign the same plan.")

# ===========================================================================
elif stage == "Test":
    cui.stage_head(7, "Test and learn", "How do we know it worked?",
                   "Design the next experiment before it runs: how many customers, how long, and how a pre-period covariate "
                   "shortens it. Then read a finished test properly, funnel and all, and see what checking results every week "
                   "does to false wins.", "")
    rho = M.cuped_rho()
    with st.container(border=True):
        a, b_, c_, d_ = st.columns(4)
        p0 = a.slider("Baseline 6-month churn", 0.02, 0.20, 0.06, 0.005, format="%.3f", key="sv_p0")
        mde = b_.slider("Smallest effect worth finding", 0.05, 0.40, 0.15, 0.01, format="%.2f", key="sv_mde",
                        help="Relative reduction in churn. 0.15 = churn falls from 6% to 5.1%.")
        weekly = c_.slider("Eligible customers a week", 500, 10000, 3000, 250, key="sv_weekly")
        cuped = d_.toggle("CUPED adjustment", value=True, key="sv_cuped", help="Adjust for each customer's pre-period churn risk.")
    n_plain = M.sample_size(p0, mde)
    n_adj = M.sample_size(p0, mde, variance_cut=rho ** 2)
    n = n_adj if cuped else n_plain
    weeks = 2 * n / weekly
    t.tiles([
        {"label": "Customers per arm", "value": f"{n:,}", "accent": True, "note": f"{n_plain:,} without adjustment"},
        {"label": "Weeks to enroll", "value": f"{weeks:.1f}", "accent": True, "note": f"at {weekly:,} eligible a week, two arms"},
        {"label": "Variance removed", "value": t.pct(rho ** 2 if cuped else 0), "note": f"pre-period risk correlates {rho:.2f} with the outcome"},
        {"label": "Detectable change", "value": f"{p0:.1%} → {p0 * (1 - mde):.1%}", "note": "95% confidence, 80% power"},
    ])
    left, right = st.columns([1, 1], gap="large")
    with left:
        mdes = np.linspace(0.05, 0.4, 36)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mdes * 100, y=[M.sample_size(p0, m_) for m_ in mdes], name="Plain comparison", line=dict(color=t.S2, width=2)))
        fig.add_trace(go.Scatter(x=mdes * 100, y=[M.sample_size(p0, m_, variance_cut=rho ** 2) for m_ in mdes], name="With CUPED", line=dict(color=t.S1, width=2.5)))
        fig.add_vline(x=mde * 100, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Customers per arm by the effect you need to detect", xaxis_title="Relative reduction in churn (%)",
                          yaxis=dict(type="log", title="Customers per arm (log scale)"), legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
    with right:
        pk = M.peeking()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(1, len(pk["by_look"]) + 1)), y=np.array(pk["by_look"]) * 100, mode="lines+markers",
                                  line=dict(color=t.S2, width=2.5), name="Check weekly, stop at first win"))
        fig2.add_hline(y=5, line=dict(color=t.S1, width=1.5, dash="dot"))
        fig2.add_annotation(x=1, xref="paper", y=5, text="the 5% you signed up for ", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=t.S1))
        fig2.update_layout(title="False wins in tests with no real effect", xaxis=dict(title="Weekly looks so far", dtick=1),
                           yaxis=dict(title="Tests declared a winner (%)", range=[0, max(pk["by_look"]) * 130]), showlegend=False)
        t.chart(fig2, height=360)
    BU.subhead("A finished test: the new save flow", "Cancel calls were randomized between the current save script and a new flow "
               "that starts with a plan review. The decision metric was agreed before launch: still active at 90 days.")
    sf = M.save_flow_test()
    piv = sf.pivot(index="stage", columns="arm", values="count").reindex(sf["stage"].unique())
    start = piv.iloc[0]
    rate = piv / start
    c_r, t_r = rate["Current script"].iloc[-1], rate["New plan-review flow"].iloc[-1]
    se = np.sqrt(c_r * (1 - c_r) / start["Current script"] + t_r * (1 - t_r) / start["New plan-review flow"])
    diff = t_r - c_r
    fl, fr = st.columns([1.3, 1], gap="large")
    with fl:
        fig3 = go.Figure()
        for arm, color in (("Current script", t.S2), ("New plan-review flow", t.S1)):
            fig3.add_trace(go.Bar(y=rate.index, x=rate[arm] * 100, name=arm, orientation="h", marker=dict(color=color),
                                  customdata=piv[arm], hovertemplate="%{y}: %{x:.1f}% (%{customdata:,})<extra>" + arm + "</extra>"))
        fig3.update_layout(title="Save funnel by arm (% of cancel calls)", barmode="group", yaxis=dict(autorange="reversed"),
                           xaxis_title="% of customers who called to cancel", legend=dict(orientation="h", y=1.02), margin=dict(l=8, t=100), bargap=0.25)
        t.chart(fig3, height=360)
    with fr:
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        ok = lo > 0
        st.markdown(
            f'<div class="lab-reco" style="background:{"#0F4640" if ok else "#4A534D"}"><div class="e">Readout · retained at 90 days</div>'
            f'<div class="t">{"Ship the new flow" if ok else "Keep testing"}</div>'
            f'<div class="d">{t_r:.1%} vs {c_r:.1%}: <b>{diff * 100:+.1f} pts</b> (95% interval {lo * 100:+.1f} to {hi * 100:+.1f}). '
            f'Every 10,000 cancel calls, that is about <b>{diff * 10000:,.0f}</b> more customers kept, '
            f'worth roughly {t.money(diff * 10000 * base["clv"].median())} in lifetime value.</div></div>', unsafe_allow_html=True)
        st.caption("The gain comes from more callers hearing an offer and more accepting it; reach was unchanged. The funnel says where "
                   "it worked, which is what the next test builds on.")
    ui.note(f"If {t.esc(sub_id)} calls to cancel, they go through whichever flow their random assignment says. Their pre-period risk "
            f"({me.churn_6m:.0%}) is the covariate that makes the test {t.pct(rho ** 2)} cheaper.")
    t.insight(f"Checking a test every week and stopping at the first significant result declares a winner in <b>{pk['any_look']:.0%}</b> of "
              f"tests where nothing changed, against the 5% everyone thinks they are getting. A sequential boundary brings it back to "
              f"{pk['corrected']:.0%}. Adjusting for pre-period risk removes {rho ** 2:.0%} of the noise, which shortens every test.")
    cui.call("Every test has a one-page plan before launch: the decision it informs, the metric, the sample size, the stopping rule "
             "and who signs off. That page is what turns a test-and-learn agenda from a stream of dashboards into decisions.")

# ===========================================================================
elif stage == "Care":
    cui.stage_head(8, "Care and collections", "What are customers telling us, and who needs help paying?",
                   "A text classifier reads care-contact notes and tags the reason, so every contact feeds the churn model and "
                   "the weekly report. A randomized payment-reminder test shows where a reminder prevents a disconnect and "
                   "where it changes nothing.", "")
    vec, clf, acc, f1, conf = M.care_classifier()
    callers = subs[subs["last_driver"] != "No contact"]
    drv = callers.groupby("last_driver").agg(n=("sub_id", "size"), churn=("churned", "mean")).reindex(D.DRIVERS)
    rem = M.reminder_effects()
    r_all = company.reminder
    t.tiles([
        {"label": "Reason tagged correctly", "value": t.pct(acc), "accent": True, "note": "held-out care notes, seven reasons"},
        {"label": "Contacts tagged a month", "value": f"{int(callers['care_calls_90d'].sum() / 3):,}", "note": "would otherwise be read by hand or not at all"},
        {"label": "Churn after a cancel request", "value": t.pct(drv.loc["Cancel request", "churn"]), "note": f"vs {t.pct(drv['churn'].mean())} across all contacts"},
        {"label": "Disconnects a reminder prevents", "value": f"{(rem['effect'] * rem['n'] / 2).clip(lower=0).sum():,.0f}", "accent": True,
         "note": f"of {len(r_all):,} past-due accounts this month"},
    ])
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown("**Try the classifier**")
        txt = st.text_input("A care note", "internet keeps dropping at night and now my bill went up", key="sv_note")
        if txt.strip():
            pr = clf.predict_proba(vec.transform([txt]))[0]
            o = np.argsort(-pr)[:3]
            st.markdown("".join(f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid #EDEFEA;padding:6px 0">'
                                f'<span>{t.esc(clf.classes_[i])}</span><b>{pr[i]:.0%}</b></div>' for i in o), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(y=drv.index, x=drv["churn"] * 100, orientation="h", marker=dict(color=t.S2, cornerradius=3),
                               customdata=drv["n"], hovertemplate="%{y}: %{x:.0f}% left (%{customdata:,} customers)<extra></extra>"))
        fig.update_layout(title="Later left, by last contact reason", xaxis_title="Left (%)", yaxis=dict(autorange="reversed"),
                          margin=dict(l=8, t=70))
        t.chart(fig, height=320)
    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rem["effect"] * 100, y=rem["band"], mode="markers", marker=dict(size=12, color=t.S1),
                                  error_x=dict(type="data", symmetric=False, array=(rem["hi"] - rem["effect"]) * 100,
                                               arrayminus=(rem["effect"] - rem["lo"]) * 100, color=t.S1, thickness=1.5, width=0),
                                  customdata=rem["n"], hovertemplate="%{y}: %{x:+.1f} pts (n=%{customdata})<extra></extra>"))
        fig2.add_vline(x=0, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Disconnects prevented by a payment reminder, with 95% intervals", xaxis_title="Disconnect rate, control minus reminded (pts)",
                           yaxis=dict(title="Late payments in the last year", autorange="reversed"), margin=dict(t=70))
        t.chart(fig2, height=320)
        fig3 = go.Figure(go.Heatmap(z=(conf.to_numpy() / conf.to_numpy().sum(1, keepdims=True)) * 100, x=[c.split(" ")[0] for c in conf.columns],
                                    y=conf.index, colorscale=[[0, "#F1F2EE"], [1, "#0F4640"]], showscale=False, xgap=1, ygap=1,
                                    hovertemplate="Actual %{y} · predicted %{x}: %{z:.0f}%<extra></extra>"))
        fig3.update_layout(title="Classifier confusion (row %)", yaxis=dict(autorange="reversed"), margin=dict(l=8, t=60))
        t.chart(fig3, height=300)
    ui.note(f"{t.esc(sub_id)}'s last contact was <b>{t.esc(me.last_driver.lower())}</b>, with {int(me.care_calls_90d)} care calls in 90 days "
            f"and {int(me.late_payments_12m)} late payments in the past year.")
    t.insight("Cancel requests and price-increase calls are the strongest warning signs, and both are now tagged automatically and fed "
              "back into the churn score within a day. Reminders work for customers who are occasionally late and do nothing for "
              "those who are repeatedly behind; those accounts need a payment plan, not a text.")
    cui.call("Automating the tag is the easy part. The value comes from routing: a cancel request flags the account for the save "
             "team the same day, and repeat outage calls trigger a proactive credit before the customer asks.")

# ===========================================================================
elif stage == "Govern":
    cui.stage_head(9, "Govern and brief", "Can we trust it, and what should leadership decide?",
                   "Every model in one inventory with its validation and drift, then the one-page brief that turns nine stages "
                   "of analysis into three decisions.", "")
    ch = M.churn()
    up = M.uplift()
    S = M.segments()
    vec, clf, acc, f1, conf = M.care_classifier()
    drift = M.churn_psi()
    bench = M.churn_bench().table.set_index("algorithm")
    inv = pd.DataFrame([
        ("Customer segments", "Segment", "Agreement with true structure", f"{S.ari:.2f}", "Healthy", "K-means, six segments, frozen annually"),
        ("Lifetime value (survival)", "Value", "Monthly hazard, logistic", "Calibrated", "Healthy", "Refit monthly; margins from Finance"),
        ("Churn propensity", "Churn", "AUC, later period", f"{ch.auc:.3f}", "Champion", f"XGBoost; score PSI {drift:.2f}"),
        ("Churn challenger", "Churn", "AUC, later period", f"{bench.loc['Logistic regression', 'auc']:.3f}", "Challenger", "Logistic regression, shadow scoring"),
        ("Retention uplift", "Treat", "Qini area vs risk targeting", f"{up.qini_area['Uplift (logistic, interactions)']:.0f} vs {up.qini_area['Churn risk']:.0f}", "Healthy", "10% random hold-out every campaign"),
        ("Price response", "Price", "Learned from randomized test", f"{len(company.price_test):,} customers", "Monitor", "Re-test before any increase above $8"),
        ("Care reason classifier", "Care", "Accuracy, held-out notes", f"{acc:.0%}", "Healthy", "Monthly sample reviewed by Care QA"),
        ("Payment reminders", "Care", "Randomized test", "Positive for 0–2 late", "Healthy", "Policy: remind 0–2 late, plan for 3+"),
    ], columns=["Model", "Stage", "Measure", "Value", "Status", "Notes"])
    t.tiles([
        {"label": "Models in production", "value": f"{len(inv)}", "note": "each with an owner and a review date"},
        {"label": "Churn score drift (PSI)", "value": f"{drift:.2f}", "accent": drift < 0.1, "note": "training vs today; 0.10 watch, 0.25 act"},
        {"label": "Learned from experiments", "value": "3", "accent": True, "note": "price test, retention campaign, reminders"},
        {"label": "Needing attention", "value": f"{(inv['Status'] == 'Monitor').sum()}", "note": "status Monitor"},
    ])
    st.dataframe(inv, hide_index=True, width="stretch")
    # Executive brief, built from the same numbers as the stages above (default settings).
    pool = base.copy()
    pool["value_uplift"] = pool["uplift"] * pool["clv"] - D.OFFER_COST
    k = 5000
    by_val = pool.nlargest(k, "value_uplift")
    by_risk = pool.nlargest(k, "churn_6m")
    v_val = (by_val["uplift"] * by_val["clv"]).sum() - D.OFFER_COST * k
    v_risk = (by_risk["uplift"] * by_risk["clv"]).sum() - D.OFFER_COST * k
    grid = M.best_increase_by_segment()
    best = grid.loc[grid.groupby("segment")["net"].idxmax()].set_index("segment")["increase"]
    plan = M.plan_value({s_: int(best.get(s_, 0)) for s_ in ORDER}, 8, 0.10, True, True)
    raised = plan.groupby("segment")["increase"].max()
    raised_txt = ", ".join(f"${int(v)} for {s_.lower()}" for s_, v in raised.items() if v > 0)
    sf = M.save_flow_test().pivot(index="stage", columns="arm", values="count")
    lift = (sf.loc["Still active at 90 days"] / sf.loc["Cancel intent"]).diff().iloc[-1]
    upv = M.upgrade_value()
    items = {
        "eyebrow": f"{D.NAME} · Customer value and retention · Quarterly brief",
        "title": "Spend retention money where it changes the outcome",
        "lede": (f"The base is {len(base):,} subscribers worth {t.money(base['clv'].sum())} in five-year lifetime value. Churn runs at "
                 f"{company.monthly['churn_rate'].tail(3).mean():.2%} a month and spikes when contracts and promotions end. Three decisions this quarter "
                 f"are worth about <b>{t.money(v_val - v_risk + plan['net'].sum() + upv['gain'].sum() * 0.2)}</b>."),
        "recommendations": [
            ("Retarget retention offers by uplift and value, not risk.", f"Same {k:,} offers: {t.money(v_val)} net value instead of {t.money(v_risk)}. "
             "Stop offering to customers the contact pushes out."),
            ("Raise prices selectively, with guardrails.", f"{raised_txt or 'No segment'}; exempt the riskiest 10%, anyone near a contract or promotion end and "
             f"high-value customers at risk. Net {t.money(plan['net'].sum())} over 12 months."),
            ("Ship the plan-review save flow and launch an upgrade pilot.", f"The randomized test kept {lift * 100:+.1f} pts more cancel callers at 90 days. "
             f"{len(upv):,} heavy users on the entry plan are worth {t.money(upv['gain'].mean())} more each if upgraded."),
        ],
        "numbers": [
            ("Active subscribers", f"{len(base):,}"), ("Lifetime value of the base", t.money(base["clv"].sum())),
            ("Churn model accuracy (AUC, later period)", f"{ch.auc:.3f}"), ("Retention: value vs risk targeting, same budget", f"{t.money(v_val)} vs {t.money(v_risk)}"),
            ("Price plan: net 12-month value", t.money(plan["net"].sum())), ("Price plan: extra customers lost", f"{plan['extra_churn'].sum():,.0f}"),
            ("Save flow: retained at 90 days", f"{lift * 100:+.1f} pts"), ("Upgrade pilot at 20% take rate", t.money(upv["gain"].sum() * 0.2)),
        ],
        "risks": ["Price response was measured at $0–8; anything higher needs a new test before rollout.",
                  "Uplift estimates are noisy at segment level; a 10% random hold-out stays in every campaign.",
                  f"Churn score drift is {drift:.2f} (PSI); refit if it passes 0.10."],
        "asks": ["Pricing and Finance: approve the guardrails and the segment increases.", "Care: staff the save flow rollout and the outage credit trigger.",
                 "Marketing: move the retention audience to the uplift list from next cycle."],
        "footer": f"{D.NAME} is fictional and every number is synthetic, from the subscriber lab at richardhenderson.io.",
    }
    BU.subhead("Executive brief", "One page, three decisions, the numbers behind them, the risks and who needs to act. Built from the "
               "same models as the stages above at their default settings, and downloadable.")
    recs = "".join(f"<li style='margin:.45rem 0'><b>{t.esc(h)}</b> {b}</li>" for h, b in items["recommendations"])
    st.markdown(f'<div class="lab-card" style="padding:22px 26px"><div style="font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#6B756E">{t.esc(items["eyebrow"])}</div>'
                f'<h3 style="font-family:Bodoni Moda,Georgia,serif;font-weight:400;color:#0F4640;font-size:1.7rem;margin:.3rem 0 .4rem">{t.esc(items["title"])}</h3>'
                f'<p style="font-size:1.02rem">{items["lede"]}</p><ol style="padding-left:1.2rem">{recs}</ol></div>', unsafe_allow_html=True)
    bl, br = st.columns([1.2, 1], gap="large")
    with bl:
        st.dataframe(pd.DataFrame(items["numbers"], columns=["Measure", "Value"]), hide_index=True, width="stretch")
    with br:
        st.markdown("**Risks and guardrails**\n" + "\n".join(f"- {r}" for r in items["risks"]))
        st.markdown("**Decisions needed**\n" + "\n".join(f"- {a_}" for a_ in items["asks"]))
        st.download_button("Download the brief (HTML)", ui.brief_html(items), file_name="corvane-quarterly-brief.html", mime="text/html",
                           key="sv_dl", type="primary")
    ui.note(f"Eight models touch {t.esc(sub_id)}: their segment, lifetime value, churn score, offer decision, price decision, upgrade "
            "eligibility, care tags and, if they fall behind, a reminder. Each decision is logged with the model version behind it.")
    cui.call("The brief leads with decisions and dollars, not models. The models are in the appendix and in this inventory, where "
             "Analytics owns them. Executives get three choices, what each is worth and what could go wrong.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.sv_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote(f"{D.NAME} is fictional. All subscribers, experiments and outcomes are synthetic and illustrative.")
