"""Lending decision lab: one fictional credit union, one member base, the whole lifecycle."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import theme as t
from lab import widgets as w
from lab.cu import data as d
from lab.cu import models as M
from lab.cu import ui

ui.css()
STAGES = ["Know", "Acquire", "Underwrite", "Fraud", "Price", "Cross-sell", "Retain", "Collect", "Govern"]

t.header(
    f"Flagship product · {d.NAME} (fictional)",
    "One lender, one member base, the whole lending lifecycle",
    "Fifty thousand synthetic members, five loan products and nine connected decisions, from finding the right "
    "prospects to collecting a late payment and proving it is all fair. Every stage reuses the same members and "
    "the same segments, so you can follow one person through all of it.",
)

bank = d.bank()
members = bank.members
seg = M.segments()
members = members.assign(segment=members["member_id"].map(seg).to_numpy())


# ---------------------------------------------------------------------------
# Stage state (linkable with ?stage=) and the member being followed.
# ---------------------------------------------------------------------------
q = str(st.query_params.get("stage", "")).lower()
lookup = {s.lower().replace("-", ""): s for s in STAGES}
if "cu_stage" not in st.session_state:
    st.session_state.cu_stage = lookup.get(q.replace("-", ""), STAGES[0])
    st.session_state.cu_stage__last = st.session_state.cu_stage


@st.cache_data(show_spinner=False)
def featured() -> dict:
    """A few members worth following, one or two per segment, chosen for an interesting story."""
    m = members
    picks = {}
    rng = np.random.default_rng(3)

    def pick(mask, label):
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx):
            r = m.iloc[idx[rng.integers(len(idx))]]
            picks[f"{r.member_id} · {label}"] = r.member_id

    pick((m.segment == "Rate-sensitive refinancers") & (m.mortgage_rate >= 7.2) & m.has_auto, "refinancer on a 7%+ mortgage")
    dq = bank.delinquency.set_index("member_id")["state"]
    pick((m.segment == "Credit builders") & m.member_id.map(dq).eq(1), "credit builder, 30 days late")
    pick((m.segment == "Digital starters") & ~m.has_card & (m.credit_score > 680), "digital starter without a card")
    pick((m.segment == "Growing families") & m.has_mortgage & m.has_auto, "family with a mortgage and auto loan")
    pick((m.segment == "Affluent savers") & (m.deposit_balance > 150_000), "affluent saver, few loans")
    pick((m.segment == "Retired loyalists") & (m.tenure_years > 25), "retired, 25+ year member")
    return picks


picks = featured()
top_l, top_r = st.columns([1.4, 2.6], gap="large")
with top_l:
    label = st.selectbox("Follow a member through every stage", list(picks), key="cu_member_label")
member_id = picks[label]
mrow = members.set_index("member_id").loc[member_id]

with top_r:
    st.markdown(
        f'<div style="padding-top:1.9rem">{ui.chip(mrow.segment)} &nbsp;·&nbsp; age {mrow.age} · {t.esc(mrow.metro)} · '
        f'member {mrow.tenure_years:.0f} years · {int(mrow.products_held)} products</div>',
        unsafe_allow_html=True,
    )

stage = w.segmented("Stage", STAGES, key="cu_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace("-", "")
idx = STAGES.index(stage)
st.write("")


def fmt_pct(x, dp=0):
    return f"{x * 100:.{dp}f}%"


# ===========================================================================
if stage == "Know":
    ui.stage_head(1, "Know your members", "Who are our members, really?",
                  "Segmentation is the spine of the lab. Members are clustered once on behavior, credit, products and "
                  "life stage, the clusters are named in business language, and every later stage reports by them.",
                  "diagnostic")
    with st.container(border=True):
        a, b = st.columns([3, 1])
        groups = a.multiselect("Cluster on", list(M.FEATURE_GROUPS), default=list(M.FEATURE_GROUPS), key="cu_groups")
        k = b.slider("Segments", 3, 9, 6, key="cu_k")
    groups = tuple(g for g in M.FEATURE_GROUPS if g in groups) or tuple(M.FEATURE_GROUPS)
    S = M.segment(groups, k)
    prof = S.profiles
    t.tiles([
        {"label": "Members", "value": f"{len(members):,}", "note": "five loan products, twelve metros"},
        {"label": "Segments found", "value": f"{len(prof)}", "accent": True, "note": f"on {len(S.features)} features"},
        {"label": "Recovered the true structure", "value": f"{S.ari:.2f}", "accent": True,
         "note": "agreement with the hidden archetypes (1 = perfect)"},
        {"label": "Separation", "value": f"{S.silhouette:.2f}", "note": "silhouette; real customer data rarely beats 0.3"},
    ])
    mseg_this = S.names[S.labels[members.index[members.member_id == member_id][0]]]
    cards_html = []
    for r in prof.sort_values("members", ascending=False).itertuples():
        color = ui.SEG_COLORS.get(r.segment, t.BASE)
        cls = "lab-card is-member" if r.segment == mseg_this else "lab-card"
        cards_html.append(
            f'<div class="{cls}"><h4><i style="background:{color}"></i>{t.esc(r.segment)}</h4><dl>'
            f'<dt>Members</dt><dd>{r.members:,} ({r.members / len(members):.0%})</dd>'
            f'<dt>Median age · income</dt><dd>{r.age:.0f} · ${r.income / 1000:.0f}K</dd>'
            f'<dt>Median credit score</dt><dd>{r.score:.0f}</dd>'
            f'<dt>Products held</dt><dd>{r.products:.1f}</dd>'
            f'<dt>Digital share</dt><dd>{r.digital:.0%}</dd>'
            f'<dt>Default · churn (12m)</dt><dd>{r.default_rate:.1%} · {r.churn:.0%}</dd></dl></div>')
    st.markdown(f'<div class="cu-seg">{"".join(cards_html)}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        cz = S.centroids_z[["age", "log_income", "log_deposits", "credit_score", "utilization", "digital_share",
                            "products_held", "high_rate_mortgage", "tenure_years"]]
        labels_y = [S.names[i] for i in cz.index]
        fig = go.Figure(go.Heatmap(
            z=cz.to_numpy().clip(-2, 2), x=["Age", "Income", "Deposits", "Credit score", "Utilization", "Digital",
                                            "Products", "7%+ mortgage", "Tenure"],
            y=labels_y, colorscale=[[0, "#B4561B"], [0.5, "#F1F2EE"], [1, "#0F4640"]], zmid=0, zmin=-2, zmax=2,
            colorbar=dict(title="vs average", thickness=10, len=0.8),
            hovertemplate="%{y} · %{x}: %{z:.1f} sd<extra></extra>", xgap=2, ygap=2))
        fig.update_layout(title="What makes each segment different (standard deviations from average)", margin=dict(l=8, t=70))
        t.chart(fig, height=380)
    with right:
        rf = M.rfm(members).assign(segment=members["segment"].to_numpy())
        rf["score"] = rf["R"] + rf["F"] + rf["M"]
        g = rf.groupby("segment")[["R", "F", "M"]].mean().reindex(ui.SEGMENT_ORDER)
        fig2 = go.Figure()
        for col, name, color in (("R", "Recency", t.S1), ("F", "Frequency", t.S2), ("M", "Monetary", t.S3)):
            fig2.add_trace(go.Bar(y=g.index, x=g[col], name=name, orientation="h", marker=dict(color=color, cornerradius=3),
                                  hovertemplate="%{y}: %{x:.1f}<extra>" + name + "</extra>"))
        fig2.update_layout(title="RFM by segment (quintile 1–5)", barmode="group", xaxis=dict(range=[0, 5.2]),
                           yaxis=dict(autorange="reversed"), bargap=0.25, margin=dict(l=8, t=70))
        t.chart(fig2, height=380)
    ui.member_note(f"{t.esc(member_id)} sits in {ui.chip(mrow.segment)}. Every later stage reads this segment: it sets "
                   "their price sensitivity, their likely next product, their prepayment and churn risk, and the "
                   "collection approach that works for people like them.")
    t.insight(f"Without being told, clustering recovers the six behavioral groups behind the data (agreement "
              f"<b>{M.segment().ari:.2f}</b>). Drop a feature group or change the count and watch segments merge or "
              "split. That is why segment definitions are agreed with the business, not left to the algorithm.")
    ui.call("Six segments, not the statistically best number. A seventh cluster adds a little separation and a lot of "
            "confusion for the teams who have to act on it. A segment earns its place when a different action follows from it.")

# ===========================================================================
elif stage == "Acquire":
    ui.stage_head(2, "Acquire", "Which prospects, and which markets?",
                  "Look-alike modeling finds non-members who resemble a chosen segment, then sizes the opportunity by "
                  "metro, the same question the market-intelligence product answers with public data.", "build")
    with st.container(border=True):
        a, b = st.columns([1.4, 1])
        target = a.selectbox("Find prospects who look like", ui.SEGMENT_ORDER, index=ui.SEGMENT_ORDER.index(mrow.segment), key="cu_acq_seg")
        thresh = b.slider("Similarity needed", 0.5, 0.95, 0.75, 0.05, key="cu_acq_t")
    L = M.lookalike(target)
    hits = L[L["similarity"] >= thresh]
    by_metro = hits.groupby("metro").agg(prospects=("prospect_id", "size"), score=("credit_score", "median")).sort_values("prospects", ascending=False)
    pool = members.groupby("metro").size()
    by_metro["members_now"] = pool.reindex(by_metro.index).fillna(0).astype(int)
    seg_share = members[members.segment == target].groupby("metro").size() / pool
    by_metro["penetration"] = seg_share.reindex(by_metro.index).fillna(0)
    t.tiles([
        {"label": "Look-alike prospects", "value": f"{len(hits):,}", "accent": True, "note": f"of {len(L):,} scored"},
        {"label": "Top metro", "value": by_metro.index[0] if len(by_metro) else "–", "note": f"{int(by_metro.prospects.iloc[0]):,} prospects" if len(by_metro) else ""},
        {"label": "Median credit score", "value": f"{hits['credit_score'].median():.0f}" if len(hits) else "–", "note": "of the look-alikes"},
        {"label": "Expected new members", "value": f"{len(hits) * 0.035:,.0f}", "accent": True, "note": "at a 3.5% response to outreach"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        bm = by_metro.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(y=bm.index, x=bm["prospects"], orientation="h", marker=dict(color=ui.SEG_COLORS[target], cornerradius=3),
                               customdata=bm[["members_now", "penetration"]], hovertemplate="%{y}: %{x:,} look-alikes · %{customdata[0]:,} members today<extra></extra>"))
        fig.update_layout(title=f"Where the {target.lower()} look-alikes are", xaxis_title="Prospects above the similarity threshold")
        t.chart(fig, height=400)
    with right:
        fig2 = go.Figure(go.Histogram(x=L["similarity"], nbinsx=40, marker=dict(color=t.BASE)))
        fig2.add_vline(x=thresh, line=dict(color=t.S2, width=1.5))
        fig2.update_layout(title="Similarity of all prospects to the segment", xaxis_title="Similarity (0–1)", yaxis_title="Prospects")
        t.chart(fig2, height=400)
    ui.member_note(f"There are <b>{int((L['similarity'] >= thresh).sum() if target == mrow.segment else 0):,}</b> prospects who look like "
                   f"{t.esc(member_id)}'s segment" + (f", {int(hits['metro'].eq(mrow.metro).sum()):,} of them in {t.esc(mrow.metro)}." if target == mrow.segment else ". Switch the target back to their segment to see how many live near them."))
    ui.call("Look-alikes find people who resemble today's members, including today's mistakes. Score them for risk at "
            "Underwrite before spending on them, and check that a look-alike list does not quietly exclude neighborhoods "
            "(Govern runs that test).")
    st.markdown(f"Public-data view of the same question: the **Market intelligence** product scores every U.S. metro.")

# ===========================================================================
elif stage == "Underwrite":
    ui.stage_head(3, "Underwrite", "Should we approve this application?",
                  "A points-based scorecard built on weight of evidence, compared with gradient boosting, with reject "
                  "inference to correct for only seeing outcomes on loans that were approved. Reason codes come straight "
                  "from the points a declined applicant missed.", "build")
    U = M.underwriting()
    ct = M.cutoff_table(U.test)
    with st.container(border=True):
        a, b, c = st.columns(3)
        cut = a.slider("Approval cut-off (score)", 480, 680, 560, 5, key="cu_cut")
        nim = b.slider("Net margin on a good loan", 0.02, 0.08, 0.045, 0.005, format="%.3f", key="cu_nim")
        lgd = c.slider("Loss given default", 0.3, 0.9, 0.6, 0.05, key="cu_lgd")
    ct = M.cutoff_table(U.test, nim=nim, lgd=lgd)
    row = ct.iloc[(ct["cutoff"] - cut).abs().argmin()]
    best = ct.loc[ct["profit"].idxmax()]
    st.session_state["cu_cut_value"] = cut
    t.tiles([
        {"label": "Approval rate", "value": fmt_pct(row.approval_rate), "accent": True, "note": f"{int(row.approved):,} of {len(U.test):,} test applications"},
        {"label": "Bad rate among approved", "value": fmt_pct(row.bad_rate, 1), "note": "defaults within 12 months"},
        {"label": "Profit, test window", "value": t.money(row.profit), "accent": True, "note": f"best cut-off {int(best.cutoff)} → {t.money(best.profit)}"},
        {"label": "Adverse impact ratio", "value": f"{row.air:.2f}", "note": "group B ÷ group A approval; below 0.80 needs review"},
    ])
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ct["cutoff"], y=ct["approval_rate"] * 100, name="Approval rate (%)", line=dict(color=t.S1, width=2.5),
                                 hovertemplate="Cut-off %{x}: %{y:.0f}% approved<extra></extra>"))
        fig.add_trace(go.Scatter(x=ct["cutoff"], y=ct["bad_rate"] * 100 * 5, name="Bad rate × 5 (%)", line=dict(color=t.S2, width=2),
                                 customdata=ct["bad_rate"] * 100, hovertemplate="Cut-off %{x}: %{customdata:.1f}% bad<extra></extra>"))
        fig.add_vline(x=cut, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Approval and bad rate move together as the cut-off rises", xaxis_title="Score cut-off",
                          yaxis_title="%", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
        fig_p = go.Figure(go.Scatter(x=ct["cutoff"], y=ct["profit"], line=dict(color=t.PETROL, width=2.5), fill="tozeroy",
                                     fillcolor="rgba(15,70,64,.08)", hovertemplate="Cut-off %{x}: %{y:$,.0f}<extra></extra>"))
        fig_p.add_vline(x=cut, line=dict(color=t.INK, width=1))
        fig_p.add_hline(y=0, line=dict(color=t.GRID, width=1))
        fig_p.update_layout(title="Profit by cut-off", xaxis_title="Score cut-off", yaxis=dict(tickprefix="$", tickformat="~s"))
        t.chart(fig_p, height=280)
    with right:
        auc = pd.Series(U.auc).sort_values()
        fig2 = go.Figure(go.Bar(y=auc.index, x=auc.values, orientation="h", text=[f"{v:.3f}" for v in auc.values], textposition="outside",
                                marker=dict(color=[t.BASE if "Legacy" in i else t.S1 if "reject" in i else t.S3 for i in auc.index], cornerradius=3),
                                hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
        fig2.update_layout(title="Ranking power (AUC), next year's applicants", xaxis=dict(range=[0.75, 0.86]), margin=dict(l=8, t=70))
        t.chart(fig2, height=260)
        cal = U.calib
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cal["score"], y=cal["actual"] * 100, name="Actual", mode="lines+markers", line=dict(color=t.INK, width=2)))
        fig3.add_trace(go.Scatter(x=cal["score"], y=cal["kgb"] * 100, name="Approved-only model", mode="lines+markers", line=dict(color=t.S2, width=2)))
        fig3.add_trace(go.Scatter(x=cal["score"], y=cal["ri"] * 100, name="With reject inference", mode="lines+markers", line=dict(color=t.S1, width=2)))
        fig3.update_layout(title="Predicted vs actual bad rate by score", xaxis_title="Score", yaxis_title="Bad rate (%)",
                           legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig3, height=380)
    # This member as an applicant for a $15,000 auto loan.
    app = pd.DataFrame([{"credit_score": mrow.credit_score, "dti": mrow.dti, "utilization": mrow.utilization, "income": mrow.income,
                         "inquiries_6m": mrow.inquiries_6m, "tenure_years": mrow.tenure_years, "member_id": member_id}])
    xa = M.uw_frame(app)
    score = float(U.ri.score(xa)[0])
    decision = "Approve" if score >= cut else "Decline"
    reasons = U.ri.reasons(xa)
    rs = "".join(f"<li><span>{t.esc(r)}</span><span>−{p:.0f} pts</span></li>" for r, p in reasons) if decision == "Decline" else ""
    ui.member_note(f"As an applicant for a $15,000 auto loan, {t.esc(member_id)} scores <b>{score:.0f}</b> against a cut-off of {cut}: "
                   f"<b>{decision.lower()}</b>, predicted default {U.ri.pd(xa)[0]:.1%}." +
                   (f" The adverse-action notice would cite:<ul class='cu-reasons'>{rs}</ul>" if rs else ""))
    t.insight("The approved-only model <b>understates risk at the low end</b>, because it never saw the outcomes of the "
              "applicants the old policy turned down. Reject inference closes that gap, erring slightly cautious at the "
              "very bottom, which is the safe direction, and it is the version that should set the cut-off. Gradient boosting ranks a little better but cannot produce points-based reasons on its own.")
    ui.call("The scorecard stays the champion and gradient boosting runs as challenger. The extra ranking power is real "
            "but small, and a scorecard's reasons are exact and auditable, which matters more for adverse-action notices "
            "and examiners than a fraction of a point of AUC.")
    with st.expander("Method notes"):
        st.markdown("""
- **Data:** consumer-loan applications (card, auto, personal). Trained on months 0–17, tested on months 18–23 (out of time) across all applicants, including those the old policy declined, whose outcomes are known only because the data is synthetic.
- **Scorecard:** six quantile bins per characteristic, weight of evidence, logistic regression, then scaled so 600 points means 50:1 odds and every 20 points doubles them.
- **Reject inference:** fuzzy augmentation. Each rejected applicant enters twice, as good and as bad, weighted by an inflated approved-only risk estimate.
- **Reason codes:** the characteristics where the applicant lost the most points against the maximum available. In production these map to standard adverse-action reason statements.
- **In production:** monotonic constraints on the boosted model, SHAP-based reasons for the challenger, and a policy layer for hard rules (bankruptcy, fraud flags) above the score.
""")

# ===========================================================================
elif stage == "Fraud":
    ui.stage_head(4, "Detect fraud", "Is this application real?",
                  "An isolation forest scores how unusual each application is, simple rules catch known patterns, and a "
                  "link graph joins applications that share a phone or an address. Synthetic-identity rings show up as "
                  "clusters no single application would reveal.", "build")
    F = M.fraud()
    with st.container(border=True):
        a, b = st.columns(2)
        queue = a.slider("Investigator queue (applications per review cycle)", 50, 800, 250, 25, key="cu_fq")
        weight = b.slider("Weight on the anomaly model vs rules", 0.0, 1.0, 0.55, 0.05, key="cu_fw")
    F = F.assign(score=weight * F["anomaly_pct"] + (1 - weight) * (F["rules_hit"] / 4).clip(0, 1))
    top = F.nlargest(queue, "score")
    caught = int(top["is_fraud"].sum())
    total = int(F["is_fraud"].sum())
    rings = F.loc[F["fraud_ring"] >= 0, "fraud_ring"].nunique()
    rings_hit = top.loc[top["fraud_ring"] >= 0, "fraud_ring"].nunique()
    t.tiles([
        {"label": "Precision of the queue", "value": fmt_pct(caught / queue), "accent": True, "note": f"{caught} of {queue} reviewed are fraud"},
        {"label": "Fraud caught", "value": fmt_pct(caught / total), "note": f"{caught} of {total} fraudulent applications"},
        {"label": "Rings exposed", "value": f"{rings_hit} of {rings}", "accent": True, "note": "at least one member of the ring in the queue"},
        {"label": "Loss avoided", "value": t.money(top.loc[top.is_fraud, "amount"].sum() * 0.85), "note": "if caught before funding"},
    ])
    left, right = st.columns([1, 1], gap="large")
    with left:
        ks = np.arange(25, 1001, 25)
        srt = F.sort_values("score", ascending=False)["is_fraud"].to_numpy()
        prec = [srt[:k].mean() for k in ks]
        rec = [srt[:k].sum() / total for k in ks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=np.array(prec) * 100, name="Precision", line=dict(color=t.S1, width=2.5)))
        fig.add_trace(go.Scatter(x=ks, y=np.array(rec) * 100, name="Fraud caught", line=dict(color=t.S3, width=2.5)))
        fig.add_vline(x=queue, line=dict(color=t.INK, width=1))
        fig.update_layout(title="The queue-size trade-off", xaxis_title="Applications reviewed", yaxis_title="%",
                          legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
    with right:
        bylink = F.assign(linked=pd.cut(F["link_size"], [0, 1, 2, 4, 100], labels=["Alone", "2 linked", "3–4 linked", "5+ linked"]))
        g = bylink.groupby("linked", observed=True)["is_fraud"].mean() * 100
        fig_l = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker=dict(color=t.S2, cornerradius=3),
                                 hovertemplate="%{x}: %{y:.1f}% fraud<extra></extra>"))
        fig_l.update_layout(title="Fraud rate by size of the linked cluster", xaxis_title="Applications sharing a phone or address", yaxis_title="Fraud (%)")
        t.chart(fig_l, height=360)
    if True:
        show = top.head(12)[["app_id", "product", "amount", "link_size", "score", "reasons", "is_fraud"]].copy()
        show["reasons"] = show["reasons"].apply(lambda r: "; ".join(r) if r else "Unusual combination (anomaly model)")
        show["is_fraud"] = show["is_fraud"].map({True: "Fraud", False: "Legitimate"})
        st.markdown("**Top of the review queue**")
        st.dataframe(show.rename(columns={"app_id": "Application", "product": "Product", "amount": "Amount",
                                          "link_size": "Linked apps", "score": "Risk", "reasons": "Why it is here", "is_fraud": "Outcome (known here)"}),
                     hide_index=True, width="stretch", height=320,
                     column_config={"Amount": st.column_config.NumberColumn(format="$%d"), "Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    ui.member_note(f"Long-standing members almost never reach the queue: <b>{top['member_id'].isna().mean():.0%}</b> of flagged applications come from "
                   f"people with no existing relationship. {t.esc(member_id)}'s history is itself a strong signal of a genuine identity.")
    ui.call("Rings are reviewed as rings. When one application in a linked cluster is confirmed as fraud, every linked "
            "application is pulled, which is why the graph links matter more than any single score. Households that share "
            "an address are the main false alarm, so a shared address alone never declines an application.")

# ===========================================================================
elif stage == "Price":
    ui.stage_head(5, "Price", "What rate should we offer?",
                  "Risk-based pricing builds the rate from cost of funds, expected loss and operating cost; price "
                  "elasticity, estimated from randomised rate variation in past quotes, says how many members accept at "
                  "each rate. Profit is take-up times margin, and riskier borrowers are the least rate-sensitive.", "build")
    elas = M.segment_elasticities()
    _, _, quotes = M.pricing_model()
    tiers = {"A": "A · 740+", "B": "B · 680–739", "C": "C · 620–679", "D": "D · below 620"}
    with st.container(border=True):
        a, b, c, d_ = st.columns(4)
        segp = a.selectbox("Segment", ui.SEGMENT_ORDER, index=ui.SEGMENT_ORDER.index(mrow.segment), key="cu_p_seg")
        tier = b.selectbox("Risk tier", list(tiers), format_func=tiers.get, index=1, key="cu_p_tier")
        cof = c.slider("Cost of funds", 2.0, 6.0, 4.1, 0.1, format="%.1f%%", key="cu_cof")
        rate_pick = d_.slider("Offer rate", 3.0, 18.0, 7.5, 0.05, format="%.2f%%", key="cu_rate")
    qt = quotes[quotes["tier"] == tier]
    comp = float(qt["competitor_rate"].mean())
    pd_t = float(qt["pd"].mean())
    rates = np.linspace(max(2.5, comp - 2), comp + 3.5, 141)
    tu = M.take_up(rates, comp, segp, tier, pd_t)
    bal, life, opex, lgd = 25_000, 2.5, 0.9, 0.55
    # Adverse selection: the takers' risk rises as the rate climbs above the market.
    pd_takers = pd_t * (1 + 0.6 * np.clip(rates - comp, 0, None))
    margin = (rates - cof - opex) / 100 * bal * life - pd_takers * lgd * bal
    profit = tu * margin
    k = int(np.argmax(profit))
    j = int(np.argmin(np.abs(rates - rate_pick)))
    floor = cof + opex + pd_t * lgd / life * 100
    t.tiles([
        {"label": "Profit-maximizing rate", "value": f"{rates[k]:.2f}%", "accent": True, "note": f"market rate {comp:.2f}%"},
        {"label": "Take-up at your rate", "value": fmt_pct(tu[j]), "note": f"{fmt_pct(tu[k])} at the optimum"},
        {"label": "Profit per 100 offers", "value": t.money(profit[j] * 100), "accent": True, "note": f"{t.money(profit[k] * 100)} at the optimum"},
        {"label": "Break-even rate", "value": f"{floor:.2f}%", "note": "cost of funds + operating cost + expected loss"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rates, y=profit * 100, name="Profit per 100 offers", line=dict(color=t.S1, width=2.5),
                                 hovertemplate="%{x:.2f}% → %{y:$,.0f}<extra></extra>", showlegend=False))
        fig.add_trace(go.Scatter(x=[rates[k]], y=[profit[k] * 100], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=11, color=t.PETROL, line=dict(color="#fff", width=2))))
        fig.add_vline(x=rate_pick, line=dict(color=t.S2, width=1.2))
        fig.add_annotation(x=rate_pick, y=1, yref="paper", text=" your rate", showarrow=False, xanchor="left", yanchor="top", font=dict(size=12, color=t.S2))
        fig.add_hline(y=0, line=dict(color=t.GRID, width=1))
        fig.update_layout(title=f"{segp}, tier {tier}: profit per 100 offers by rate", xaxis_title="Offer rate (%)",
                          yaxis=dict(tickprefix="$", tickformat="~s", range=[min(0, float(profit.min() * 100)) * 0.3, float(profit.max() * 100) * 1.25]))
        t.chart(fig, height=320)
        fig_t = go.Figure(go.Scatter(x=rates, y=tu * 100, line=dict(color=t.S3, width=2.5), hovertemplate="%{x:.2f}%: %{y:.0f}% accept<extra></extra>"))
        fig_t.add_vline(x=comp, line=dict(color=t.GRID, width=1, dash="dot"))
        fig_t.add_annotation(x=comp, y=1, yref="paper", text=" market", showarrow=False, xanchor="left", yanchor="top", font=dict(size=12, color=t.MUTED))
        fig_t.add_vline(x=rate_pick, line=dict(color=t.S2, width=1.2))
        fig_t.update_layout(title="Take-up by rate", xaxis_title="Offer rate (%)", yaxis_title="Accept (%)", yaxis=dict(range=[0, 100]))
        t.chart(fig_t, height=240)
    with right:
        e = elas.reindex(ui.SEGMENT_ORDER)
        fig2 = go.Figure(go.Bar(y=e.index, x=-e.values, orientation="h", marker=dict(color=[ui.SEG_COLORS[s] for s in e.index], cornerradius=3),
                                hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig2.update_layout(title="Rate sensitivity by segment", xaxis_title="Drop in acceptance log-odds per +1 pt",
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=230)
        quotes["gap_b"] = pd.cut(quotes["rate"] - quotes["competitor_rate"], [-3, -0.5, 0, 0.5, 1, 4], labels=["≤ −0.5", "−0.5–0", "0–0.5", "0.5–1", "> 1"])
        sel = quotes[quotes["accepted"]].groupby("gap_b", observed=True)["pd"].mean() * 100
        fig3 = go.Figure(go.Bar(x=sel.index.astype(str), y=sel.values, marker=dict(color=t.S2, cornerradius=3),
                                hovertemplate="%{x} pts over market: %{y:.1f}% risk<extra></extra>"))
        fig3.update_layout(title="Adverse selection: who accepts when the rate is high", xaxis_title="Offer rate minus market (pts)",
                           yaxis_title="Average risk of acceptors (%)")
        t.chart(fig3, height=240)
    tier_m = "A" if mrow.credit_score >= 740 else "B" if mrow.credit_score >= 680 else "C" if mrow.credit_score >= 620 else "D"
    qm = quotes[quotes["tier"] == tier_m]
    comp_m = float(qm["competitor_rate"].mean())
    r_m = np.linspace(comp_m - 3, comp_m + 4, 141)
    tu_m = M.take_up(r_m, comp_m, mrow.segment, tier_m, float(mrow.pd_true))
    pr_m = tu_m * ((r_m - cof - opex) / 100 * bal * life - float(mrow.pd_true) * (1 + 0.6 * np.clip(r_m - comp_m, 0, None)) * lgd * bal)
    km = int(np.argmax(pr_m))
    ui.member_note(f"{t.esc(member_id)} is risk tier <b>{tier_m}</b> in {ui.chip(mrow.segment)}. For a $25,000 auto loan the "
                   f"copilot would quote <b>{r_m[km]:.2f}%</b> (market {comp_m:.2f}%), with a {tu_m[km]:.0%} chance they accept.")
    ui.call("Price to the segment, never to the protected class. Segment-level elasticity is a legitimate commercial input, "
            "but the resulting rate spread is tested by group in Govern before it ships, and a rate floor stops the optimizer "
            "from buying volume below break-even.")

# ===========================================================================
elif stage == "Cross-sell":
    ui.stage_head(6, "Cross-sell", "What should we offer next, and to whom?",
                  "Next-best-product propensity says what a member is likely to want; uplift modeling, trained on last "
                  "quarter's randomised card campaign, says whether contacting them changes anything. They are different "
                  "questions, and only the second one is worth paying for.", "advisory")
    C = M.uplift()
    with st.container(border=True):
        budget = st.slider("Members you can contact this quarter", 1000, 12000, 4000, 500, key="cu_xs_budget")
    q_up = M.qini(C, "uplift")
    q_pr = M.qini(C, "p1")
    q_rd = C[C["test"]].sample(frac=1, random_state=1)
    tot = q_up["incremental"].iloc[-1]
    share = budget / len(C)
    inc_up = float(np.interp(share, q_up["share"], q_up["incremental"])) / C["test"].mean()
    inc_pr = float(np.interp(share, q_pr["share"], q_pr["incremental"])) / C["test"].mean()
    inc_rd = tot * share / C["test"].mean()
    dnd = C[C["uplift"] < -0.005]
    t.tiles([
        {"label": "Extra cards · uplift targeting", "value": f"{inc_up:,.0f}", "accent": True, "note": f"contacting {budget:,} members"},
        {"label": "Extra cards · propensity targeting", "value": f"{inc_pr:,.0f}", "note": "contacting the likeliest buyers"},
        {"label": "Extra cards · random", "value": f"{inc_rd:,.0f}", "note": "the business-as-usual baseline"},
        {"label": "Members better left alone", "value": f"{len(dnd):,}", "accent": True, "note": "contact lowers their chance of buying"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure()
        scale = 1 / C["test"].mean()
        fig.add_trace(go.Scatter(x=q_up["share"] * len(C), y=q_up["incremental"] * scale, name="Uplift model", line=dict(color=t.S1, width=2.5)))
        fig.add_trace(go.Scatter(x=q_pr["share"] * len(C), y=q_pr["incremental"] * scale, name="Propensity model", line=dict(color=t.S2, width=2)))
        fig.add_trace(go.Scatter(x=[0, len(C)], y=[0, tot * scale], name="Random", line=dict(color=t.BASE, width=1.5, dash="dot")))
        fig.add_vline(x=budget, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Incremental cards by number of members contacted (Qini curve)", xaxis_title="Members contacted",
                          yaxis_title="Extra cards vs no contact", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=380)
    with right:
        g = C.groupby("segment").agg(uplift=("uplift", "mean"), base=("p0", "mean")).reindex(ui.SEGMENT_ORDER).dropna()
        fig2 = go.Figure(go.Bar(y=g.index, x=g["uplift"] * 100, orientation="h",
                                marker=dict(color=[t.S1 if v > 0 else t.S2 for v in g["uplift"]], cornerradius=3),
                                customdata=g["base"] * 100, hovertemplate="%{y}: %{x:+.1f} pts (buys anyway: %{customdata:.0f}%)<extra></extra>"))
        fig2.add_vline(x=0, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Effect of contact on card take-up, by segment", xaxis_title="Change in take-up (pts)",
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=380)
    _, probs = M.next_best_product()
    pm = probs.loc[member_id].dropna()
    if len(pm):
        val = pm * pd.Series(M.PRODUCT_VALUE).reindex(pm.index)
        best_p = val.idxmax()
        others = ", ".join(f"{M.PRODUCT_LABEL[p].lower()} ({pm[p]:.0%})" for p in val.sort_values(ascending=False).index[1:3])
        in_camp = C[C["member_id"] == member_id]
        upl = f" Card-campaign uplift for them: <b>{in_camp['uplift'].iloc[0] * 100:+.1f} pts</b>." if len(in_camp) else ""
        ui.member_note(f"Next best product for {t.esc(member_id)}: <b>{M.PRODUCT_LABEL[best_p]}</b>, a {pm[best_p]:.0%} likelihood "
                       f"worth about ${val[best_p]:,.0f} in expected annual margin. Runners-up: {others}.{upl}")
    t.insight("Propensity targeting spends most of the budget on members who would have bought anyway. Uplift targeting "
              "spends it where contact changes the outcome, and leaves <b>retired loyalists</b> alone: for them the "
              "campaign <b>lowered</b> take-up.")
    ui.call("The campaign measures incremental cards against a randomised hold-out, never total cards sold. That rule "
            "is what lets the uplift model exist next quarter, so a slice of every campaign stays random by design.")

# ===========================================================================
elif stage == "Retain":
    ui.stage_head(7, "Retain", "Who is about to leave?",
                  "Survival analysis on the mortgage book estimates monthly prepayment hazard from each loan's rate "
                  "incentive; a churn model scores every member's risk of leaving. Move the market rate and watch "
                  "which segments run off.", "build")
    with st.container(border=True):
        a, b = st.columns(2)
        mkt = a.slider("30-year market rate, next 12 months", 4.0, 8.0, 6.25, 0.05, format="%.2f%%", key="cu_mkt")
        recapture = b.slider("Prepaying borrowers we refinance in-house", 0, 60, 25, 5, format="%d%%", key="cu_recap",
                             help="A proactive refinance offer before they shop elsewhere.") / 100
    proj = M.project_prepay(mkt)
    cm, cauc = M.churn_model()
    runoff = (proj["p_12m"] * proj["mortgage_balance"]).sum()
    kept = runoff * recapture
    t.tiles([
        {"label": "Mortgage balance at risk, 12 months", "value": t.money(runoff), "accent": True, "note": f"{proj['p_12m'].mean():.0%} of {len(proj):,} loans expected to prepay"},
        {"label": "Kept by refinancing in-house", "value": t.money(kept), "accent": True, "note": f"{recapture:.0%} of prepaying balance recaptured"},
        {"label": "Members likely to leave", "value": f"{(cm['churn_score'] > 0.25).sum():,}", "note": "churn score above 25%"},
        {"label": "Churn model accuracy", "value": f"{cauc:.2f}", "note": "AUC on held-out members"},
    ])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        _, _, mort = M.prepay_model()
        inc0 = mort["mortgage_rate"] - 6.9
        bucket = pd.cut(inc0, [-9, 0, 0.75, 9], labels=["Below market", "0–0.75 pts above", "0.75+ pts above"])
        km = M.kaplan_meier(mort, bucket)
        fig = go.Figure()
        for g_, color in zip(["Below market", "0–0.75 pts above", "0.75+ pts above"], [t.S3, t.S1, t.S2]):
            label_ = {"Below market": "Note rate below market", "0–0.75 pts above": "0–0.75 pts above market", "0.75+ pts above": "0.75+ pts above market"}[g_]
            s_ = km[km["group"] == g_]
            fig.add_trace(go.Scatter(x=s_["month"], y=s_["survival"] * 100, name=label_, line=dict(color=color, width=2.5, shape="hv")))
        fig.update_layout(title="Share of mortgages still on the books (Kaplan–Meier)", xaxis_title="Months observed",
                          yaxis_title="Still on books (%)", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=380)
    with right:
        g = proj.groupby("segment").apply(lambda s: (s["p_12m"] * s["mortgage_balance"]).sum(), include_groups=False).reindex(ui.SEGMENT_ORDER).fillna(0)
        fig2 = go.Figure(go.Bar(y=g.index, x=g.values, orientation="h", marker=dict(color=[ui.SEG_COLORS[s] for s in g.index], cornerradius=3),
                                hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
        fig2.update_layout(title=f"Balance expected to prepay at {mkt:.2f}%", xaxis=dict(tickprefix="$", tickformat="~s"),
                           yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig2, height=380)
    churn_m = float(cm.set_index("member_id").loc[member_id, "churn_score"])
    pre = proj.set_index("member_id")["p_12m"].get(member_id)
    ui.member_note(f"{t.esc(member_id)} has a <b>{churn_m:.0%}</b> chance of leaving in the next year."
                   + (f" Their mortgage at {mrow.mortgage_rate:.2f}% has a <b>{pre:.0%}</b> chance of prepaying if market rates sit at {mkt:.2f}%." if pre is not None else "")
                   + (" A proactive refinance offer from the credit union keeps the relationship even if the old loan goes." if pre is not None and pre > 0.3 else ""))
    t.insight("Prepayment is concentrated: <b>rate-sensitive refinancers</b> carry most of the runoff even though they are "
              "about a seventh of members. Retention offers belong there, timed to the rate moves, not spread across the book.")
    ui.call("Some prepayment should be welcomed, not fought. When a borrower will refinance anyway, the cheapest outcome is "
            "that they refinance with us, so the retention play for refinancers is a proactive offer, not a rate cut on the old loan.")

# ===========================================================================
elif stage == "Collect":
    ui.stage_head(8, "Collect", "Who needs help paying, and how?",
                  "A roll-rate Markov chain projects how today's delinquent balances will move; treatment effects from "
                  "past randomised outreach say which action works for which segment; a capacity-aware allocation decides "
                  "who gets a call, a text or a hardship plan.", "build")
    with st.container(border=True):
        a, b = st.columns(2)
        collectors = a.slider("Collectors on the team", 1, 12, 4, key="cu_coll")
        segc = b.selectbox("Roll rates for", ["All members"] + ui.SEGMENT_ORDER, key="cu_roll_seg")
    plan, util = M.allocate(collectors)
    curve = [(c, M.allocate(c)[0]["expected_recovery"].sum()) for c in range(1, 13)]
    rec = plan["expected_recovery"].sum()
    marg = curve[min(collectors, 11)][1] - curve[collectors - 1][1] if collectors < 12 else 0
    P = M.roll_rates(None if segc == "All members" else segc)
    dq = bank.delinquency
    dist = np.bincount(dq["state"], minlength=5) / len(dq)
    proj_states = [dist]
    for _ in range(12):
        proj_states.append(proj_states[-1] @ P.to_numpy())
    t.tiles([
        {"label": "Accounts in collections", "value": f"{len(plan):,}", "note": f"{t.money(plan['balance'].sum())} past due"},
        {"label": "Expected recoveries", "value": t.money(rec), "accent": True, "note": f"collector time {util:.0%} used"},
        {"label": "Value of one more collector", "value": t.money(marg), "accent": True, "note": "a month, at the margin"},
        {"label": "Charged off in 12 months", "value": fmt_pct(proj_states[-1][4] - dist[4], 2), "note": "of accounts, projected by roll rates"},
    ])
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        fig = go.Figure(go.Heatmap(z=P.to_numpy() * 100, x=P.columns, y=P.index, colorscale=[[0, "#F1F2EE"], [1, "#0F4640"]],
                                   text=[[f"{v * 100:.0f}%" if v >= 0.005 else "" for v in r] for r in P.to_numpy()], texttemplate="%{text}",
                                   hovertemplate="%{y} → %{x}: %{z:.1f}%<extra></extra>", showscale=False, xgap=2, ygap=2))
        fig.update_layout(title="Monthly roll rates (from → to)", yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
        t.chart(fig, height=360)
    with right:
        fig2 = go.Figure(go.Scatter(x=[c for c, _ in curve], y=[v for _, v in curve], mode="lines+markers", line=dict(color=t.S1, width=2.5),
                                    hovertemplate="%{x} collectors: %{y:$,.0f}<extra></extra>"))
        fig2.add_vline(x=collectors, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Expected recoveries by team size", xaxis_title="Collectors", yaxis=dict(tickprefix="$", tickformat="~s"))
        t.chart(fig2, height=220)
        mix = plan.groupby(["segment", "treatment"]).size().unstack(fill_value=0).reindex(ui.SEGMENT_ORDER).dropna(how="all").fillna(0)
        fig3 = go.Figure()
        for tr_, color in (("Text", t.S3), ("Call", t.S1), ("Hardship plan", t.S2)):
            if tr_ in mix:
                fig3.add_trace(go.Bar(y=mix.index, x=mix[tr_], name=tr_, orientation="h", marker=dict(color=color)))
        fig3.update_layout(title="Treatment chosen, by segment", barmode="stack", yaxis=dict(autorange="reversed"),
                           legend=dict(orientation="h", y=1.02), margin=dict(l=8, t=90), bargap=0.3)
        t.chart(fig3, height=300)
    me = plan[plan["member_id"] == member_id]
    if len(me):
        ui.member_note(f"{t.esc(member_id)} is <b>{d.STATES[int(me['state'].iloc[0])].lower()}</b> past due on {t.money(me['balance'].iloc[0])}. "
                       f"With {collectors} collectors the plan assigns a <b>{me['treatment'].iloc[0].lower()}</b>, the action with the best expected "
                       "recovery for their segment and stage.")
    else:
        state_m = bank.delinquency.set_index("member_id")["state"].get(member_id)
        ui.member_note(f"{t.esc(member_id)} is " + ("current on every loan" if state_m in (0, None) else d.STATES[int(state_m)].lower()) +
                       ", so they are not in today's queue. Choose the credit builder under “Follow a member” to see an account through collections.")
    ui.call("A hardship plan is the most expensive action and often the best one: for credit builders it cures more accounts "
            "than any call, and it keeps a member instead of charging one off. Collector time goes where it changes the "
            "outcome, and texts cover everyone else.")

# ===========================================================================
elif stage == "Govern":
    ui.stage_head(9, "Govern", "Can we trust and defend all of this?",
                  "Every model above in one inventory: what it decides, how well it performs out of sample, whether its "
                  "inputs have drifted and whether its decisions are fair. Fair-lending testing uses a synthetic "
                  "protected-class proxy that no model ever sees.", "advisory")
    U = M.underwriting()
    cut = st.session_state.get("cu_cut_value", 560)
    ct = M.cutoff_table(U.test)
    air_now = float(ct.iloc[(ct["cutoff"] - cut).abs().argmin()]["air"])
    a_all = bank.applications
    early = a_all[(a_all["month"] < 12) & a_all["product"].isin(["card", "auto", "personal"])]
    late = a_all[(a_all["month"] >= 18) & a_all["product"].isin(["card", "auto", "personal"])]
    psi_score = M.psi(U.ri.score(M.uw_frame(early)), U.ri.score(M.uw_frame(late)))
    F = M.fraud()
    C = M.uplift()
    cm, cauc = M.churn_model()
    S = M.segment()
    from sklearn.metrics import roc_auc_score
    q_up = M.qini(C, "uplift")
    qini_gain = float(q_up["incremental"].max() / max(C[C["test"]]["adopted"].sum(), 1))
    inv = pd.DataFrame([
        ("Member segmentation", "Know", "Agreement with true structure", f"{S.ari:.2f}", "Healthy", "K-means, 6 segments, quarterly refresh"),
        ("Underwriting scorecard", "Underwrite", "AUC, next-year applicants", f"{U.auc['Scorecard + reject inference']:.3f}", "Champion", f"Score PSI {psi_score:.2f}; AIR {air_now:.2f} at cut-off {cut}"),
        ("Gradient boosting", "Underwrite", "AUC, next-year applicants", f"{U.auc['Gradient boosting (approved only)']:.3f}", "Challenger", "Shadow mode; needs reasons before promotion"),
        ("Fraud screening", "Fraud", "Precision, top 250", f"{F.nlargest(250, 'fraud_score')['is_fraud'].mean():.0%}", "Healthy", "Rings reviewed together"),
        ("Rate take-up", "Price", "AUC on quotes", f"{roc_auc_score(M.pricing_model()[2]['accepted'], M.pricing_model()[0].predict_proba(M._price_design(M.pricing_model()[2])[M.pricing_model()[1]])[:, 1]):.2f}", "Monitor", "Elasticities re-estimated monthly from rate tests"),
        ("Card uplift", "Cross-sell", "Share of adopters that are incremental", f"{qini_gain:.0%}", "Healthy", "10% random hold-out every campaign"),
        ("Prepayment hazard", "Retain", "Fit on 460K loan-months", "Calibrated", "Healthy", "Re-fit when rates move 50 bps"),
        ("Member churn", "Retain", "AUC, held-out members", f"{cauc:.2f}", "Monitor", "Below 0.75: add product-usage features"),
        ("Roll rates + treatment", "Collect", "Treatment effects from randomised outreach", "Shrunk", "Healthy", "Thin segment cells pulled to overall"),
    ], columns=["Model", "Stage", "Measure", "Value", "Status", "Notes"])
    status_counts = inv["Status"].value_counts()
    t.tiles([
        {"label": "Models in production", "value": f"{len(inv)}", "note": "one inventory, one owner each"},
        {"label": "Adverse impact ratio", "value": f"{air_now:.2f}", "accent": air_now >= 0.8, "note": f"at the cut-off set in Underwrite ({cut}); 0.80 is the line"},
        {"label": "Score drift (PSI)", "value": f"{psi_score:.2f}", "note": "early vs recent applicants; 0.10 watch, 0.25 act"},
        {"label": "Needing attention", "value": f"{status_counts.get('Monitor', 0)}", "accent": True, "note": "status Monitor"},
    ])
    st.dataframe(inv, hide_index=True, width="stretch",
                 column_config={"Status": st.column_config.TextColumn(help="Champion / Challenger / Healthy / Monitor")})
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure(go.Scatter(x=ct["cutoff"], y=ct["air"], line=dict(color=t.S1, width=2.5), hovertemplate="Cut-off %{x}: AIR %{y:.2f}<extra></extra>"))
        fig.add_hrect(y0=0, y1=0.8, fillcolor="rgba(217,119,43,.08)", line_width=0)
        fig.add_hline(y=0.8, line=dict(color=t.S2, width=1.2, dash="dot"))
        fig.add_vline(x=cut, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Adverse impact ratio across cut-offs", xaxis_title="Score cut-off", yaxis_title="Group B ÷ group A approval rate",
                          yaxis=dict(range=[0.5, 1.02]))
        t.chart(fig, height=340)
    with right:
        te = U.test
        tbl = pd.DataFrame({
            "Group A": [(te.loc[~te.group_b, "score"] >= cut).mean(), te.loc[~te.group_b & (te.score >= cut), "default_12m"].mean(), te.loc[~te.group_b, "score"].median()],
            "Group B": [(te.loc[te.group_b, "score"] >= cut).mean(), te.loc[te.group_b & (te.score >= cut), "default_12m"].mean(), te.loc[te.group_b, "score"].median()],
        }, index=["Approval rate", "Bad rate if approved", "Median score"])
        st.markdown("**Fair-lending read-out at the current cut-off**")
        st.dataframe(tbl.style.format({"Group A": lambda v: f"{v:.1%}" if v < 1 else f"{v:.0f}", "Group B": lambda v: f"{v:.1%}" if v < 1 else f"{v:.0f}"}), width="stretch")
        st.caption("Similar bad rates at the same score suggest the score predicts equally well for both groups; the approval "
                   "gap comes through income and credit history, which is where a less discriminatory alternative search starts.")
    ui.member_note(f"Seven models touch {t.esc(member_id)}: their segment, an underwriting score, fraud screening, a price, a next-best "
                   "offer, a churn score and, if they fall behind, a collection treatment. Each decision is logged with the model "
                   "version and its reasons, so any one of them can be explained on request.")
    ui.call("Fairness is tested on outcomes, not assumed from inputs. No model sees the protected-class proxy, yet the approval "
            "gap is real because it flows through correlated inputs. The cut-off is set with the adverse impact ratio on the "
            "same chart as profit, and any change below 0.80 goes to compliance before it ships.")
    with st.expander("Model card · underwriting scorecard"):
        st.markdown(f"""
- **Decision:** approve or decline consumer-loan applications; produce adverse-action reasons.
- **Data:** {len(U.test):,} out-of-time test applications (months 18–23); trained on months 0–17 with reject inference.
- **Performance:** AUC {U.auc['Scorecard + reject inference']:.3f}; calibrated by score band (see Underwrite).
- **Fairness:** adverse impact ratio {air_now:.2f} at cut-off {cut}; bad rates at equal scores compared by group.
- **Drift:** score PSI {psi_score:.2f}, early versus recent applicants.
- **Limits:** synthetic data; no bureau attributes beyond score, DTI, utilization and inquiries; illustrative only.
- **Owner and review:** credit risk; quarterly validation, annual independent review.
""")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.cu_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote(f"{d.NAME} is fictional. All members, applications and outcomes are synthetic; underwriting, pricing and "
           "fair-lending results are illustrative and are not credit policy or legal advice.")
