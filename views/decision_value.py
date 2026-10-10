import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from lab import theme as t

t.header(
    "Decision economics · Model value",
    "What is a model actually worth?",
    "Accuracy is not value. Value is what changes when people act on the output, net of what acting "
    "costs, limited by how much the team can do and how often they trust the recommendation. Set the "
    "economics of your decision and see where the money is.",
)

with st.container(border=True):
    a, b, c, d = st.columns(4)
    population = a.number_input("Cases per month", 500, 2_000_000, 10_000, step=500, format="%d")
    base_rate = b.slider("Base rate of success (%)", 0.5, 40.0, 5.0, 0.5) / 100
    value = c.number_input("Value of one success ($)", 10, 1_000_000, 1_200, step=100, format="%d")
    cost = d.number_input("Cost of one action ($)", 0, 100_000, 45, step=5, format="%d")
    e, f, g = st.columns([1, 1, 1])
    auc = e.slider("Model ranking quality (AUC)", 0.51, 0.97, 0.78, 0.01, help="0.5 is a coin flip; 1.0 is perfect ordering.")
    capacity = f.slider("Capacity — actions per month", 100, int(population), min(3_000, int(population)), step=100)
    adoption = g.slider("Adoption — recommendations acted on (%)", 5, 100, 65, 5, help="Share of the model's recommendations people actually follow.") / 100


def curve(auc_value: float):
    """Binormal ROC: return share targeted, TPR, FPR along a threshold sweep."""
    sep = np.sqrt(2) * norm.ppf(auc_value)
    thresholds = np.linspace(-5, sep + 5, 1600)
    tpr = 1 - norm.cdf(thresholds - sep)
    fpr = 1 - norm.cdf(thresholds)
    share = base_rate * tpr + (1 - base_rate) * fpr
    return share[::-1], tpr[::-1], fpr[::-1]


def best_value(auc_value: float, adopt: float):
    share, tpr, fpr = curve(auc_value)
    net = adopt * population * (base_rate * tpr * value - share * cost)
    allowed = share <= capacity / population
    idx = int(np.argmax(np.where(allowed, net, -np.inf)))
    return net[idx], share[idx], tpr[idx], fpr[idx], share, net


net_star, q_star, tpr_star, fpr_star, share, net = best_value(auc, adoption)
cap_share = capacity / population
random_net = adoption * population * min(cap_share, 1.0) * (base_rate * value - cost)
random_best = max(0.0, random_net)
acted = population * q_star

t.tiles(
    [
        {"label": "Net value per month · model", "value": t.money(net_star), "accent": True, "note": f"acting on {q_star:.0%} of cases ({acted:,.0f})"},
        {"label": "Same capacity, no model", "value": t.money(random_best), "note": "random targeting" if random_net > 0 else "acting at random loses money; best is not to act"},
        {"label": "Value the model adds", "value": t.money(net_star - random_best), "accent": True, "note": f"{t.money((net_star - random_best) * 12)} a year"},
        {"label": "Successes captured", "value": f"{tpr_star:.0%}", "note": f"of {population * base_rate:,.0f} available, before adoption"},
    ]
)

left, right = st.columns([1.3, 1], gap="large")

with left:
    x = share * 100
    random_line = adoption * population * share * (base_rate * value - cost)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=random_line, name="No model (random)", line=dict(color=t.BASE, width=1.5),
                             hovertemplate="%{x:.0f}% targeted → %{y:$,.0f}<extra>No model</extra>"))
    fig.add_trace(go.Scatter(x=x, y=net, name="With model", line=dict(color=t.S1, width=2.5),
                             hovertemplate="%{x:.0f}% targeted → %{y:$,.0f}<extra>With model</extra>"))
    fig.add_vrect(x0=cap_share * 100, x1=100, fillcolor=t.MIST, opacity=0.8, line_width=0, layer="below")
    fig.add_annotation(x=min(99, cap_share * 100 + 1), y=1, yref="paper", text="beyond capacity", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=12, color=t.MUTED))
    fig.add_trace(go.Scatter(x=[q_star * 100], y=[net_star], mode="markers", name="Best operating point",
                             marker=dict(size=11, color=t.PETROL, line=dict(color="#101B1B", width=2)),
                             hovertemplate="Best: act on %{x:.1f}% → %{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t.GRID, width=1))
    fig.update_layout(
        title="Net monthly value by share of cases acted on",
        xaxis_title="Share of cases acted on, highest scores first (%)",
        yaxis_title="Net value per month",
        yaxis=dict(tickprefix="$", tickformat="~s"),
        xaxis=dict(range=[0, 100]),
        hovermode="x unified",
    )
    t.chart(fig, height=390)

with right:
    aucs = np.round(np.arange(0.60, 0.96, 0.05), 2)
    adopts = np.arange(0.3, 1.01, 0.1)
    grid = np.array([[best_value(au, ad)[0] for ad in adopts] for au in aucs])
    fig2 = go.Figure(
        go.Heatmap(
            z=grid,
            x=[f"{ad:.0%}" for ad in adopts],
            y=[f"{au:.2f}" for au in aucs],
            colorscale=[[0, "#16241F"], [0.35, "#3A693C"], [0.7, "#7FB554"], [1, "#CBFA7C"]],
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text="$/mo", font=dict(size=11, color=t.MUTED)), tickformat="~s", tickprefix="$",
                          thickness=10, outlinewidth=0, tickfont=dict(size=11, color=t.MUTED)),
            hovertemplate="AUC %{y} · adoption %{x}<br>%{z:$,.0f} per month<extra></extra>",
        )
    )
    fig2.add_trace(go.Scatter(
        x=[f"{min(adopts, key=lambda v: abs(v - adoption)):.0%}"],
        y=[f"{min(aucs, key=lambda v: abs(v - auc)):.2f}"],
        mode="markers", marker=dict(symbol="square-open", size=26, color=t.INK, line=dict(width=2)),
        hoverinfo="skip", showlegend=False,
    ))
    fig2.update_layout(
        title="Net value across model quality and adoption",
        xaxis_title="Adoption",
        yaxis_title="Model AUC",
        xaxis=dict(showline=False, ticks=""),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    t.chart(fig2, height=390)

better_model = best_value(min(auc + 0.05, 0.99), adoption)[0] - net_star
better_adopt = best_value(auc, min(adoption + 0.10, 1.0))[0] - net_star
if better_adopt >= better_model:
    verdict = (
        f"Raising adoption by ten points is worth <b>{t.money(better_adopt)}</b> a month. Improving the model by "
        f"0.05 AUC is worth <b>{t.money(better_model)}</b>. Here the next dollar is in trust and workflow, not in the algorithm."
    )
else:
    verdict = (
        f"Improving the model by 0.05 AUC is worth <b>{t.money(better_model)}</b> a month. Raising adoption by ten "
        f"points is worth <b>{t.money(better_adopt)}</b>. Here, ranking quality is the constraint worth investing in."
    )
t.insight(verdict)

st.subheader("At the best operating point", anchor=False)
tp = population * base_rate * tpr_star
fp = population * (1 - base_rate) * fpr_star
fn = population * base_rate - tp
tn = population * (1 - base_rate) - fp
m1, m2 = st.columns([1, 1.2], gap="large")
with m1:
    st.dataframe(
        [
            {"": "Acted on", "Would succeed": round(tp), "Would not": round(fp)},
            {"": "Not acted on", "Would succeed": round(fn), "Would not": round(tn)},
        ],
        hide_index=True,
        column_config={"Would succeed": st.column_config.NumberColumn(format="%d"), "Would not": st.column_config.NumberColumn(format="%d")},
    )
with m2:
    precision = tp / max(tp + fp, 1e-9)
    st.markdown(
        f"Of every 100 cases acted on, about **{precision * 100:.0f}** succeed, against **{base_rate * 100:.1f}** "
        f"without a model. Each action costs {t.money(cost).replace('$', chr(92) + '$')} and each success is worth {t.money(value).replace('$', chr(92) + '$')}, so acting "
        f"pays while the hit rate stays above **{cost / value * 100:.1f}%**. Only **{adoption:.0%}** of recommendations "
        "are acted on, which is why the value is lower than the model alone would suggest."
    )

with st.expander("The method"):
    st.markdown(
        """
- Model scores follow a **binormal** assumption: successes and non-successes are normally distributed with a separation set by the AUC. This is a standard, conservative way to translate a ranking metric into an operating curve.
- At every threshold the demo computes true and false positives, the share of cases acted on, and **net value = adoption × (successes × value − actions × cost)**.
- The best point is the highest net value that fits within capacity. The random baseline acts on the same number of cases without ranking.
- Real deployments add calibration, uncertainty in value per success, and the fact that adoption is rarely uniform. People override some recommendations more than others, and that pattern is itself worth measuring.
"""
    )

t.footnote("A planning model with illustrative defaults. Replace the inputs with your own economics.")
