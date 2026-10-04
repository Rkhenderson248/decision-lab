import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lab import pipeline_model as pm
from lab import theme as t
from lab import widgets as w


def render() -> None:
    t.header(
        "Pipeline intelligence · Predictive prioritization",
        "Where should the team focus today?",
        "A large pipeline and a fixed number of conversations. This demo scores a synthetic lending "
        "pipeline, ranks it, and shows how much more a capacity-limited team captures when the order "
        "comes from a model instead of habit.",
    )

    data = pm.build()
    today = data.today

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1, 1])
        capacity = c1.slider(
            "Team capacity — conversations today",
            min_value=50,
            max_value=1500,
            value=1200,
            step=50,
            help="How many leads the team can realistically work in one day.",
        )
        with c2:
            lead_filter = w.segmented("Pipeline", ["All", "Refinance", "Purchase"], key="pipe_filter", default="All")
        with c3:
            compare = w.segmented("Compare against", ["Newest first", "Random"], key="pipe_compare", default="Newest first")

    pool = today if lead_filter == "All" else today[today["lead_type"] == lead_filter]
    capacity = min(capacity, len(pool))

    model_conv = pm.expected_conversions(pool, "Model score", capacity)
    base_conv = pm.expected_conversions(pool, compare, capacity)
    total_conv = float(pool["true_p"].sum())
    lift = model_conv / base_conv - 1 if base_conv else 0.0

    t.tiles(
        [
            {"label": "Leads in today's pipeline", "value": f"{len(pool):,}", "note": f"{capacity:,} can be worked ({capacity / len(pool):.0%})"},
            {"label": "Expected conversions · model order", "value": f"{model_conv:,.0f}", "accent": True, "note": f"{model_conv / total_conv:.0%} of all available"},
            {"label": f"Expected conversions · {compare.lower()}", "value": f"{base_conv:,.0f}", "note": f"{base_conv / total_conv:.0%} of all available"},
            {"label": "Lift from the same effort", "value": t.pct(lift, 0, signed=True), "accent": True, "note": f"{model_conv - base_conv:,.0f} more outcomes, no added headcount"},
        ]
    )

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        fig = go.Figure()
        share = capacity / len(pool)
        for name, color, width in (("Random", t.BASE, 1.5), ("Newest first", t.S2, 2), ("Model score", t.S1, 2.5)):
            x, y = pm.gains(pool, name)
            step = max(1, len(x) // 400)
            fig.add_trace(
                go.Scatter(
                    x=np.r_[0, x[::step]] * 100,
                    y=np.r_[0, y[::step]] * 100,
                    name=name,
                    mode="lines",
                    line=dict(color=color, width=width),
                    hovertemplate="%{x:.0f}% of pipeline worked → %{y:.0f}% of conversions<extra>" + name + "</extra>",
                )
            )
        fig.add_vline(x=share * 100, line=dict(color=t.PETROL, width=1))
        fig.add_annotation(
            x=share * 100, y=4, text=f" today's capacity ({share:.0%})", showarrow=False,
            xanchor="left", font=dict(size=12, color=t.PETROL),
        )
        fig.update_layout(
            title="Share of conversions captured as the pipeline is worked",
            xaxis_title="Share of pipeline worked (%)",
            yaxis_title="Share of expected conversions (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 101]),
            hovermode="x unified",
        )
        t.chart(fig, height=380)

    with right:
        by_tier = pool.assign(
            tier=np.select(
                [pool["score"] >= pool["score"].quantile(0.9), pool["score"] >= pool["score"].quantile(0.6)],
                ["High", "Medium"],
                default="Low",
            )
        )
        summary = (
            by_tier.groupby("tier")
            .agg(leads=("lead_id", "count"), rate=("true_p", "mean"))
            .reindex(["High", "Medium", "Low"])
        )
        fig2 = go.Figure(
            go.Bar(
                x=summary.index,
                y=summary["rate"] * 100,
                marker=dict(color=t.S1, cornerradius=4),
                text=[f"{v:.1%}" for v in summary["rate"]],
                textposition="outside",
                textfont=dict(color=t.GRAPHITE, size=12),
                customdata=summary["leads"],
                hovertemplate="%{x} priority<br>%{customdata:,} leads<br>%{y:.1f}% expected conversion<extra></extra>",
                width=0.55,
            )
        )
        fig2.update_layout(
            title="Expected conversion rate by priority tier",
            yaxis_title="Expected conversion (%)",
            yaxis=dict(range=[0, summary["rate"].max() * 125]),
            bargap=0.35,
        )
        t.chart(fig2, height=380)

    ratio = summary.loc["High", "rate"] / max(summary.loc["Low", "rate"], 1e-9)
    t.insight(
        f"A high-priority lead is <b>{ratio:.1f}× more likely</b> to convert than a low-priority one. "
        f"The model's ranking quality on this pipeline is <b>AUC {data.auc:.2f}</b>. Good, not magic. "
        "Most of the gain comes from getting the order right, not from finding certainties."
    )

    st.subheader("Today's worklist", anchor=False)
    st.caption("The top of the ranked queue, with the signals behind each score and a suggested next step.")

    worklist = pm.strategy_order(pool, "Model score").head(min(capacity, 200))
    show = worklist[["lead_id", "lead_type", "channel", "score", "action", "reasons", "days_since_inquiry", "credit_score"]].rename(
        columns={
            "lead_id": "Lead",
            "lead_type": "Type",
            "channel": "Source",
            "score": "Priority score",
            "action": "Recommended action",
            "reasons": "Why it ranks here",
            "days_since_inquiry": "Days since inquiry",
            "credit_score": "Credit",
        }
    )
    st.dataframe(
        show,
        hide_index=True,
        height=420,
        column_config={
            "Priority score": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=float(max(0.01, today["score"].max()))),
            "Days since inquiry": st.column_config.NumberColumn(format="%d"),
            "Credit": st.column_config.NumberColumn(format="%d"),
        },
    )

    with st.expander("How this works, and what is illustrative"):
        st.markdown(
            f"""
    - **Data.** {len(today):,} leads generated for today, plus a separate synthetic history used for training. Conversion depends on recency, rate gap (refinance), credit profile, engagement, prior attempts and source, with noise.
    - **Model.** Standardised logistic regression. Chosen for transparency: each score decomposes into per-signal contributions, and the two largest positive ones become the reason codes.
    - **Evaluation.** Because the data is simulated, every lead's true conversion probability is known. The tiles compare *expected* conversions under each ordering, not a single lucky draw.
    - **Actions.** Simple, readable rules layered on the score. In production these would be agreed with the people doing the work.
    - **What changes in a real deployment.** Weekly retraining, calibration checks, monitoring for drift, and a feedback loop from the team about which recommendations they ignored and why.
    """
        )

    t.footnote("Synthetic records only. No customer, lender or employer data is used.")

