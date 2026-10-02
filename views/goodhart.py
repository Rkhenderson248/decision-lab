import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lab import theme as t

t.header(
    "Measurement · Goodhart's law",
    "When a measure becomes a target",
    "Put pressure on a metric and two things happen. People work harder, and some find cheaper ways to "
    "move the number than doing the work. This simulation of a team of 300 shows the point where "
    "the reported metric keeps rising while the outcome it was meant to track turns down.",
)

with st.container(border=True):
    a, b, c, d = st.columns(4)
    pressure = a.slider("Pressure on the metric", 0.0, 1.0, 0.75, 0.05, help="How strongly pay, ranking and attention follow the reported number.")
    gaming = b.slider("How cheap gaming is", 0.5, 2.5, 1.4, 0.1, help="How much the metric moves per unit of effort spent gaming rather than working.")
    audit = c.slider("Audit and detection strength", 0.5, 4.0, 1.2, 0.1, help="Raises the personal cost of gaming.")
    motivation = d.slider("Effort response to incentives", 0.0, 0.8, 0.35, 0.05, help="How much genuine effort rises with pressure.")


@st.cache_data(show_spinner=False)
def population(n: int = 300, seed: int = 3):
    rng = np.random.default_rng(seed)
    ability = np.clip(rng.lognormal(mean=0.0, sigma=0.32, size=n) * 0.9, 0.35, 1.9)
    noise_true = rng.normal(0, 0.06, size=n)
    noise_proxy = rng.normal(0, 0.06, size=n)
    return ability, noise_true, noise_proxy


ability, eps_t, eps_p = population()


def simulate(w: float):
    effort = 1 + motivation * w
    g = np.clip((w * gaming - ability) / audit, 0, 1)
    real = ability * effort * (1 - g)
    true = real + eps_t
    proxy = real + gaming * g * effort + eps_p
    return true, proxy, g


def top_overlap(true, proxy, q=0.8):
    top_p = proxy >= np.quantile(proxy, q)
    top_t = true >= np.quantile(true, q)
    return (top_p & top_t).sum() / max(top_p.sum(), 1)


ws = np.linspace(0, 1, 41)
base_true, base_proxy, _ = simulate(0.0)
rows = []
for w in ws:
    tr, pr, g = simulate(w)
    rows.append((tr.mean() / base_true.mean() * 100, pr.mean() / base_proxy.mean() * 100, np.corrcoef(tr, pr)[0, 1], (g > 0.02).mean()))
idx_true, idx_proxy, validity, gamers = map(np.array, zip(*rows))

cur_true, cur_proxy, cur_g = simulate(pressure)
cur_idx_t = cur_true.mean() / base_true.mean() - 1
cur_idx_p = cur_proxy.mean() / base_proxy.mean() - 1
cur_r = np.corrcoef(cur_true, cur_proxy)[0, 1]
overlap = top_overlap(cur_true, cur_proxy)
peak_w = ws[int(np.argmax(idx_true))]

t.tiles(
    [
        {"label": "Reported metric vs no pressure", "value": t.pct(cur_idx_p, 0, signed=True), "note": "what the dashboard shows"},
        {"label": "True outcome vs no pressure", "value": t.pct(cur_idx_t, 0, signed=True), "accent": True, "note": "what the organisation gets"},
        {"label": "Metric–outcome correlation", "value": f"{cur_r:.2f}", "note": f"{(cur_g > 0.02).mean():.0%} of the team gaming at all"},
        {"label": "Top 20% by metric who are truly top 20%", "value": f"{overlap:.0%}", "note": "who gets promoted vs who should"},
    ]
)

left, right = st.columns([1.3, 1], gap="large")
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ws, y=idx_proxy, name="Reported metric", line=dict(color=t.S2, width=2),
                             hovertemplate="pressure %{x:.2f} → %{y:.0f}<extra>Reported metric</extra>"))
    fig.add_trace(go.Scatter(x=ws, y=idx_true, name="True outcome", line=dict(color=t.S1, width=2.5),
                             hovertemplate="pressure %{x:.2f} → %{y:.0f}<extra>True outcome</extra>"))
    fig.add_vline(x=pressure, line=dict(color=t.PETROL, width=1))
    fig.add_annotation(x=pressure, y=0, yref="paper", text=" you are here", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12, color=t.PETROL))
    fig.add_annotation(x=ws[-1], y=idx_proxy[-1], text="Reported", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=12, color=t.GRAPHITE))
    fig.add_annotation(x=ws[-1], y=idx_true[-1], text="True", showarrow=False, xanchor="right", yanchor="top",
                       font=dict(size=12, color=t.GRAPHITE))
    fig.add_hline(y=100, line=dict(color=t.GRID, width=1))
    fig.update_layout(
        title="Reported metric and true outcome as pressure rises (no pressure = 100)",
        xaxis_title="Pressure on the metric",
        yaxis_title="Index",
        hovermode="x unified",
    )
    t.chart(fig, height=390)

with right:
    gamer = cur_g > 0.02
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cur_proxy[~gamer], y=cur_true[~gamer], mode="markers", name="Doing the work",
                              marker=dict(size=8, color=t.S1, opacity=0.8, line=dict(color="#FFFFFF", width=1)),
                              hovertemplate="metric %{x:.2f} · true %{y:.2f}<extra>Doing the work</extra>"))
    fig2.add_trace(go.Scatter(x=cur_proxy[gamer], y=cur_true[gamer], mode="markers", name="Gaming the metric",
                              marker=dict(size=8, color=t.S2, opacity=0.85, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                              hovertemplate="metric %{x:.2f} · true %{y:.2f}<extra>Gaming</extra>"))
    fig2.update_layout(
        title="Each person at the current pressure",
        xaxis_title="Reported metric",
        yaxis_title="True contribution",
    )
    t.chart(fig2, height=390)

if peak_w < 0.999:
    t.insight(
        f"The true outcome peaks at a pressure of about <b>{peak_w:.2f}</b>. Past that, every extra unit of "
        "pressure buys a better-looking number and a worse result. Stronger audits move the peak right; "
        "cheaper gaming moves it left. The metric alone cannot tell you which side of the peak you are on."
    )
else:
    t.insight(
        "With these settings gaming never pays enough to outweigh the extra effort, so the metric stays honest "
        "across the range. Make gaming cheaper or audits weaker and watch the curves part."
    )

with st.expander("The model behind the curves"):
    st.markdown(
        """
- Each of 300 people has a fixed **ability**. Pressure on the metric raises genuine **effort** by a set response.
- Each person splits effort between real work and **gaming**. Gaming pays when pressure × how cheap gaming is exceeds their ability at the real work, discounted by audit strength: `gaming share = clip((pressure × cheapness − ability) / audit, 0, 1)`.
- **True outcome** comes from real work only. The **reported metric** counts real work plus whatever gaming adds. Both carry small measurement noise.
- The least able game first. That is what quietly breaks the metric's validity and promotes the wrong people.
- Theory: Goodhart (1975); Campbell's law; Holmström & Milgrom on multitask incentives. The simulation is a teaching device, not an estimate of any real organisation.
"""
    )

t.footnote("A stylised simulation for discussion. No real people or performance data are involved.")
