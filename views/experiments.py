"""Experimentation lab: design, peeking, variance reduction, analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from lab import theme as t

t.header(
    "Experimentation · A/B testing done properly",
    "Will this test actually tell you anything?",
    "Most experiments fail before they start: too small to detect the effect that matters, stopped "
    "the moment they look good, or read without accounting for noise. Design one, watch peeking "
    "manufacture false wins, see how pre-period data shortens a test, then analyse a result.",
)


def sample_size(p0: float, mde_rel: float, alpha: float, power: float, two_sided: bool = True) -> int:
    """Visitors per arm for a two-proportion z-test."""
    p1 = p0 * (1 + mde_rel)
    za = norm.ppf(1 - alpha / (2 if two_sided else 1))
    zb = norm.ppf(power)
    pbar = (p0 + p1) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / (p1 - p0) ** 2))


def power_at(n: int, p0: float, mde_rel: float, alpha: float) -> float:
    p1 = p0 * (1 + mde_rel)
    se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
    za = norm.ppf(1 - alpha / 2)
    return float(1 - norm.cdf(za - abs(p1 - p0) / se) + norm.cdf(-za - abs(p1 - p0) / se))


tabs = st.tabs(["1 · Design", "2 · The peeking problem", "3 · Variance reduction (CUPED)", "4 · Analyse a result"])

# ---------------------------------------------------------------------------
with tabs[0]:
    with st.container(border=True):
        a, b, c, d, e = st.columns(5)
        p0 = a.number_input("Baseline conversion (%)", 0.1, 90.0, 4.0, 0.1) / 100
        mde = b.number_input("Smallest lift worth detecting (%)", 1.0, 100.0, 8.0, 0.5,
                             help="Relative lift. 8% means 4.0% → 4.32%.") / 100
        traffic = c.number_input("Eligible visitors per day", 100, 5_000_000, 6_000, 100)
        alpha = d.select_slider("False-positive rate (α)", [0.01, 0.025, 0.05, 0.10], value=0.05)
        pw = e.select_slider("Power", [0.70, 0.80, 0.90, 0.95], value=0.80)
    n = sample_size(p0, mde, alpha, pw)
    days = int(np.ceil(2 * n / traffic))
    weeks = days / 7
    t.tiles([
        {"label": "Visitors needed per arm", "value": f"{n:,}", "accent": True},
        {"label": "Total visitors", "value": f"{2 * n:,}"},
        {"label": "Run time at this traffic", "value": f"{days} days", "note": f"≈ {weeks:.1f} weeks; round up to whole weeks"},
        {"label": "Detectable lift", "value": f"{p0 * 100:.2f}% → {p0 * (1 + mde) * 100:.2f}%"},
    ])
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ns = np.unique(np.geomspace(max(100, n / 20), n * 3, 120).astype(int))
        pows = [power_at(k, p0, mde, alpha) for k in ns]
        fig = go.Figure(go.Scatter(x=ns, y=np.array(pows) * 100, mode="lines", line=dict(color=t.S1, width=2.5),
                                   hovertemplate="%{x:,} per arm → %{y:.0f}% power<extra></extra>", showlegend=False))
        fig.add_hline(y=pw * 100, line=dict(color=t.GRID, width=1))
        fig.add_vline(x=n, line=dict(color=t.PETROL, width=1))
        fig.add_annotation(x=n, y=8, text=f" {n:,} per arm", showarrow=False, xanchor="left", font=dict(size=12, color=t.PETROL))
        fig.update_layout(title="Chance of detecting the lift, by sample size", xaxis_title="Visitors per arm",
                          yaxis_title="Power (%)", yaxis=dict(range=[0, 101]), xaxis=dict(type="log"))
        t.chart(fig, height=360)
    with right:
        lifts = np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
        req = [int(np.ceil(2 * sample_size(p0, l, alpha, pw) / traffic)) for l in lifts]
        fig2 = go.Figure(go.Bar(x=[f"{l:.0%}" for l in lifts], y=req, marker=dict(color=t.S1, cornerradius=4),
                                text=[f"{r:,}d" for r in req], textposition="outside", textfont=dict(size=11, color=t.GRAPHITE),
                                hovertemplate="Lift %{x}: %{y:,} days<extra></extra>", width=0.6))
        fig2.update_layout(title="Days needed by lift you want to detect", xaxis_title="Relative lift",
                           yaxis=dict(type="log", title="Days (log scale)"), bargap=0.3)
        t.chart(fig2, height=360)
    halve = int(np.ceil(2 * sample_size(p0, mde / 2, alpha, pw) / traffic))
    t.insight(f"Halving the lift you can detect does not double the test. It roughly <b>quadruples</b> it: "
              f"<b>{days}</b> days becomes <b>{halve}</b>. Decide the smallest effect worth acting on before you start.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Every simulated test below is an **A/A test**: both arms are identical, so any 'winner' is a false "
                "positive. Checking significance repeatedly and stopping at the first p < 0.05 inflates that rate far "
                "past the 5% everyone thinks they are running at.")
    with st.container(border=True):
        a, b, c = st.columns(3)
        looks = a.slider("How often results are checked", 1, 30, 14, help="Number of interim looks across the test.")
        sims = b.select_slider("Simulated tests", [500, 1000, 2000, 4000], value=2000)
        seq = c.toggle("Use a sequential correction (alpha spending)", value=False,
                       help="O'Brien-Fleming-style boundaries keep the overall false-positive rate near 5%.")

    @st.cache_data(show_spinner=False)
    def peeking(looks: int, sims: int, seq: bool, seed: int = 4):
        rng = np.random.default_rng(seed)
        n_per_look = 400
        p = 0.05
        a_conv = rng.binomial(n_per_look, p, size=(sims, looks)).cumsum(axis=1)
        b_conv = rng.binomial(n_per_look, p, size=(sims, looks)).cumsum(axis=1)
        n = n_per_look * np.arange(1, looks + 1)
        pa, pb = a_conv / n, b_conv / n
        pool = (a_conv + b_conv) / (2 * n)
        se = np.sqrt(pool * (1 - pool) * 2 / n)
        z = np.abs(pb - pa) / np.where(se > 0, se, np.nan)
        if seq:
            frac = np.arange(1, looks + 1) / looks
            bound = norm.ppf(1 - 0.05 / 2) / np.sqrt(frac)  # O'Brien-Fleming approximation
        else:
            bound = np.full(looks, norm.ppf(1 - 0.05 / 2))
        hit = z > bound
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        cum = np.array([(first >= 0) & (first <= k) for k in range(looks)]).mean(axis=1)
        return cum, z[:12], bound

    cum, paths, bound = peeking(looks, sims, seq)
    t.tiles([
        {"label": "False-positive rate you think you have", "value": "5%"},
        {"label": "False-positive rate you actually have", "value": f"{cum[-1] * 100:.0f}%", "accent": True,
         "note": f"with {looks} look{'s' if looks > 1 else ''}" + (", corrected" if seq else "")},
        {"label": "Tests that 'won' at some point", "value": f"{cum[-1] * sims:,.0f} of {sims:,}"},
    ])
    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure(go.Scatter(x=np.arange(1, looks + 1), y=cum * 100, mode="lines+markers",
                                   line=dict(color=t.S2, width=2.5), marker=dict(size=7),
                                   hovertemplate="After look %{x}: %{y:.1f}% false winners<extra></extra>", showlegend=False))
        fig.add_hline(y=5, line=dict(color=t.GRID, width=1))
        fig.add_annotation(x=1, y=5, text="nominal 5%", showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=12, color=t.MUTED))
        fig.update_layout(title="Share of identical tests declared a winner", xaxis_title="Look number",
                          yaxis_title="False positives (%)", yaxis=dict(rangemode="tozero"))
        t.chart(fig, height=340)
    with right:
        fig2 = go.Figure()
        for i, z in enumerate(paths):
            fig2.add_trace(go.Scatter(x=np.arange(1, looks + 1), y=z, mode="lines", line=dict(color=t.BASE, width=1),
                                      opacity=0.7, showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=np.arange(1, looks + 1), y=bound, mode="lines", name="Significance boundary",
                                  line=dict(color=t.PETROL, width=2)))
        fig2.update_layout(title="Twelve A/A tests wandering across the boundary", xaxis_title="Look number",
                           yaxis_title="|z| statistic")
        t.chart(fig2, height=340)

# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("**CUPED** subtracts the part of each user's outcome that was predictable from before the test. "
                "The stronger the pre-period predicts the in-test metric, the less noise remains and the smaller "
                "the test can be. Same answer, fewer visitors.")
    with st.container(border=True):
        a, b = st.columns(2)
        rho = a.slider("Correlation between pre-period and test-period metric", 0.0, 0.95, 0.6, 0.05)
        base_n = b.number_input("Visitors per arm without CUPED", 1_000, 10_000_000, int(min(max(n, 1_000), 10_000_000)), 1_000)
    reduction = rho ** 2
    new_n = int(np.ceil(base_n * (1 - reduction)))
    t.tiles([
        {"label": "Variance removed", "value": f"{reduction * 100:.0f}%", "accent": True, "note": "equals ρ²"},
        {"label": "Visitors per arm with CUPED", "value": f"{new_n:,}"},
        {"label": "Test shortened by", "value": f"{(1 - new_n / base_n) * 100:.0f}%",
         "note": f"{int(np.ceil(2 * base_n / traffic))} → {int(np.ceil(2 * new_n / traffic))} days at your traffic"},
    ])
    rng = np.random.default_rng(9)
    pre = rng.normal(100, 20, 1500)
    post = 100 + rho * (pre - 100) + rng.normal(0, 20 * np.sqrt(max(1 - rho ** 2, 1e-6)), 1500)
    theta = np.cov(pre, post)[0, 1] / np.var(pre)
    adj = post - theta * (pre - pre.mean())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=post, name="Raw outcome", marker=dict(color=t.BASE), opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=adj, name="CUPED-adjusted", marker=dict(color=t.S1), opacity=0.75, nbinsx=50))
    fig.update_layout(barmode="overlay", title="Same users, same mean, less spread after adjustment",
                      xaxis_title="Outcome per user", yaxis_title="Users")
    t.chart(fig, height=340)

# ---------------------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        na = a.number_input("Control visitors", 10, 100_000_000, 48_000, 100)
        ca = b.number_input("Control conversions", 0, 100_000_000, 1_920, 10)
        nb = c.number_input("Variant visitors", 10, 100_000_000, 48_200, 100)
        cb = d.number_input("Variant conversions", 0, 100_000_000, 2_085, 10)
    if ca > na or cb > nb:
        st.error("Conversions cannot exceed visitors.")
        st.stop()
    pa, pb = ca / na, cb / nb
    diff = pb - pa
    se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    z = diff / se if se > 0 else 0
    pval = 2 * (1 - norm.cdf(abs(z)))
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    rng = np.random.default_rng(1)
    da = beta_dist.rvs(1 + ca, 1 + na - ca, size=60_000, random_state=rng)
    db = beta_dist.rvs(1 + cb, 1 + nb - cb, size=60_000, random_state=rng)
    prob = float((db > da).mean())
    rel = (db / da - 1)
    srm = abs(na - nb) / ((na + nb) / 2)
    t.tiles([
        {"label": "Observed lift", "value": t.pct(diff / pa if pa else 0, 1, signed=True), "accent": True,
         "note": f"{pa * 100:.2f}% → {pb * 100:.2f}%"},
        {"label": "95% interval on the difference", "value": f"{lo * 100:+.2f} to {hi * 100:+.2f} pts"},
        {"label": "p-value", "value": f"{pval:.3f}", "note": "two-sided z-test"},
        {"label": "Chance variant is better", "value": f"{prob * 100:.1f}%", "note": "Bayesian, flat prior"},
    ])
    fig = go.Figure(go.Histogram(x=rel * 100, nbinsx=80, marker=dict(color=t.S1), opacity=0.85,
                                 hovertemplate="Lift %{x:.1f}%: %{y} draws<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=t.INK, width=1))
    fig.update_layout(title="Plausible relative lifts given the data", xaxis_title="Relative lift (%)",
                      yaxis_title="Posterior draws", showlegend=False)
    t.chart(fig, height=320)
    verdict = ("Ship it: the interval excludes zero and the effect is practically meaningful."
               if lo > 0 and diff / pa >= 0.02 else
               "Not proven: the interval includes zero. Keep running to the planned sample, or accept you cannot see an effect this small."
               if lo <= 0 <= hi else
               "The variant is worse. Stop and learn why.")
    extra = (" <b>Check traffic allocation:</b> arm sizes differ by more than 1%, a sample-ratio mismatch that often "
             "signals a bug.") if srm > 0.01 else ""
    t.insight(f"<b>{verdict}</b>{extra}")

t.footnote("Simulations and formulas are standard; inputs are yours. Two-proportion z-tests and Beta-Binomial posteriors.")
