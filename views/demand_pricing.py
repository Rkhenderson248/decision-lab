import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from lab import demand as dm
from lab import theme as t

t.header(
    "Revenue management · Forecasting and pricing",
    "How many rooms will sell, and at what price?",
    "A 150-room hotel, 30 days of future stay dates and the bookings already on the books. Forecast final "
    "demand from booking pace, test the method against what actually happened, then set a price that "
    "trades occupancy against rate with the capacity you have left.",
)

world = dm.build_world()
bt = dm.backtest()
bt["ape"] = bt["err"].abs() / bt["final"]
names = {"naive": "Last year (prior approach)", "pickup": "Booking pace only", "blend": "Pace + seasonality"}

fut = world.future.rename(columns={"on_books_now": "on_books"})
fc = dm.forecast(world, fut[["stay_date", "dow", "lead", "on_books"]], world.asof)
fc = fc.merge(fut[["stay_date", "true_final"]], on="stay_date")
sd_by_lead = bt[bt["method"].eq("blend")].groupby("lead")["err"].std()
fc["sd"] = np.interp(fc["lead"], sd_by_lead.index, sd_by_lead.values)
fc["lo"] = np.clip(fc["blend"] - 1.28 * fc["sd"], fc["on_books"], dm.CAPACITY)
fc["hi"] = np.clip(fc["blend"] + 1.28 * fc["sd"], fc["on_books"], dm.CAPACITY)

tab_f, tab_p = st.tabs(["Forecast", "Price"])

with tab_f:
    acc = bt.groupby("method")["ape"].mean()
    gain = 1 - acc["blend"] / acc["naive"]
    t.tiles([
        {"label": "Forecast error · pace + seasonality", "value": f"{acc['blend'] * 100:.1f}%", "accent": True,
         "note": "mean absolute % error, 120-day backtest"},
        {"label": "Forecast error · last year", "value": f"{acc['naive'] * 100:.1f}%", "note": "the prior approach"},
        {"label": "Improvement", "value": f"{gain * 100:.0f}%", "accent": True, "note": "relative reduction in error"},
        {"label": "Rooms on the books, next 30 nights", "value": f"{fc['on_books'].sum():,.0f}",
         "note": f"forecast to finish near {fc['blend'].sum():,.0f}"},
    ])
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=fc["stay_date"], y=fc["on_books"], name="On the books today",
                             marker=dict(color="#C3E2D9", cornerradius=3),
                             hovertemplate="%{x|%a %b %d}: %{y:.0f} booked<extra></extra>"))
        fig.add_trace(go.Scatter(x=list(fc["stay_date"]) + list(fc["stay_date"][::-1]),
                                 y=list(fc["hi"]) + list(fc["lo"][::-1]), fill="toself",
                                 fillcolor="rgba(0,138,115,.12)", line=dict(width=0), hoverinfo="skip",
                                 name="80% range"))
        fig.add_trace(go.Scatter(x=fc["stay_date"], y=fc["blend"], mode="lines+markers", name="Forecast",
                                 line=dict(color=t.S1, width=2.5), marker=dict(size=6),
                                 hovertemplate="%{x|%a %b %d}: forecast %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["stay_date"], y=fc["naive"], mode="lines", name="Last year",
                                 line=dict(color=t.S2, width=1.5),
                                 hovertemplate="%{x|%a %b %d}: last year %{y:.0f}<extra></extra>"))
        fig.add_hline(y=dm.CAPACITY, line=dict(color=t.INK, width=1))
        fig.add_annotation(x=fc["stay_date"].iloc[-1], y=dm.CAPACITY, text="capacity", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=12, color=t.MUTED))
        fig.update_layout(title=f"Next 30 nights as of {world.asof:%b %d, %Y}", yaxis_title="Rooms",
                          yaxis=dict(range=[0, dm.CAPACITY * 1.12]), bargap=0.25, hovermode="x unified")
        t.chart(fig, height=400)
    with right:
        g = bt.groupby(["lead", "method"])["ape"].mean().unstack()
        fig2 = go.Figure()
        for m, color, width in (("naive", t.S2, 2), ("pickup", t.S3, 2), ("blend", t.S1, 2.5)):
            fig2.add_trace(go.Scatter(x=g.index, y=g[m] * 100, mode="lines+markers", name=names[m],
                                      line=dict(color=color, width=width), marker=dict(size=7),
                                      hovertemplate="%{x} days out: %{y:.1f}%<extra>" + names[m] + "</extra>"))
        fig2.update_layout(title="Error by how far ahead you forecast", xaxis_title="Days before arrival",
                           yaxis_title="Mean absolute error (%)", yaxis=dict(rangemode="tozero"),
                           legend=dict(orientation="h", y=1.02))
        t.chart(fig2, height=400)
    t.insight("Booking pace is the best signal close to arrival; history is the best signal far out. Blending "
              "them by lead time beats either alone, and both beat last year's actuals, which carry last year's "
              "one-off events into this year's plan.")
    with st.expander("How the forecast works"):
        st.markdown("""
- **Pickup:** rooms on the books today plus the average rooms that arrived from this lead time to arrival for the same weekday, over the last ten weeks.
- **Seasonality:** the recent level for that weekday, moved by last year's seasonal change between today and the stay date.
- **Blend:** weighted by lead time: mostly history 40+ days out, mostly pace in the final week.
- **Backtest:** every method is re-run as of 3, 7, 14, 21, 30 and 45 days before each of the last 120 nights, seeing only what was known then.
- **Caveat:** sold rooms are censored at capacity, so true demand on sell-out nights is higher than recorded. A production system unconstrains it first.
""")

with tab_p:
    opts = {f"{r.stay_date:%a %b %d} · {int(r.on_books)} booked · {int(r.lead)} days out": i for i, r in fc.iterrows() if r.lead >= 3}
    with st.container(border=True):
        a, b, c = st.columns([1.6, 1, 1])
        pick = a.selectbox("Stay date", list(opts), index=min(8, len(opts) - 1))
        ref_price = b.number_input("Current rate ($)", 60, 2000, 189, 1)
        elastic = c.slider("Price sensitivity", 0.5, 4.0, 1.8, 0.1,
                           help="How sharply demand falls as price rises above the current rate. 1.8 means +10% price ≈ −16% demand.")
        d, e = st.columns(2)
        prem_share = d.slider("Share of remaining demand that books late at full rate", 0.0, 0.6, 0.25, 0.05)
        disc_rate = e.slider("Discount (advance) rate as % of full rate", 40, 95, 75, 5) / 100
    row = fc.loc[opts[pick]]
    left_rooms = dm.CAPACITY - row["on_books"]
    to_come = max(row["blend"] - row["on_books"], 0.5)
    sd = max(row["sd"], 1.0)

    rng = np.random.default_rng(3)
    draws = np.clip(rng.normal(to_come, sd, 4000), 0, None)
    prices = np.linspace(ref_price * 0.6, ref_price * 1.8, 121)
    exp_rev, exp_sold = [], []
    for p in prices:
        dem = draws * np.exp(-elastic * (p / ref_price - 1))
        sold = np.minimum(dem, left_rooms)
        exp_rev.append(float((p * sold).mean()))
        exp_sold.append(float(sold.mean()))
    exp_rev, exp_sold = np.array(exp_rev), np.array(exp_sold)
    k = int(np.argmax(exp_rev))
    base_k = int(np.argmin(abs(prices - ref_price)))

    # Two-class protection (EMSR-b style): rooms held back for late full-rate demand.
    mu_h, sd_h = prem_share * to_come, max(np.sqrt(prem_share) * sd, 0.5)
    protect = float(np.clip(mu_h + sd_h * norm.ppf(1 - disc_rate), 0, left_rooms))

    t.tiles([
        {"label": "Recommended rate", "value": f"${prices[k]:,.0f}", "accent": True,
         "note": f"{prices[k] / ref_price - 1:+.0%} vs current"},
        {"label": "Expected rooms sold (remaining)", "value": f"{exp_sold[k]:.0f} of {left_rooms:.0f}",
         "note": f"{(row['on_books'] + exp_sold[k]) / dm.CAPACITY:.0%} final occupancy"},
        {"label": "Expected extra revenue", "value": t.money(exp_rev[k] - exp_rev[base_k]), "accent": True,
         "note": f"vs holding ${ref_price:,} on this night"},
        {"label": "Rooms to protect for late full-rate demand", "value": f"{protect:.0f}",
         "note": "close the discount once this many remain"},
    ])
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=prices, y=exp_rev, mode="lines", line=dict(color=t.S1, width=2.5), showlegend=False,
                                  customdata=exp_sold, hovertemplate="$%{x:.0f} → $%{y:,.0f} expected, %{customdata:.0f} rooms<extra></extra>"))
        fig3.add_trace(go.Scatter(x=[prices[k]], y=[exp_rev[k]], mode="markers", showlegend=False,
                                  marker=dict(size=12, color=t.PETROL, line=dict(color="#fff", width=2)), hoverinfo="skip"))
        fig3.add_vline(x=ref_price, line=dict(color=t.S2, width=1))
        fig3.add_annotation(x=ref_price, y=1, yref="paper", text="current rate ", showarrow=False, xanchor="right",
                            yanchor="top", font=dict(size=12, color=t.S2))
        fig3.update_layout(title="Expected revenue from remaining rooms, by rate", xaxis_title="Rate ($)",
                           yaxis_title="Expected revenue ($)", xaxis=dict(tickprefix="$"), yaxis=dict(tickprefix="$", tickformat="~s"))
        t.chart(fig3, height=360)
    with right:
        fig4 = go.Figure(go.Histogram(x=np.minimum(draws, 400), nbinsx=40, marker=dict(color=t.S1), opacity=0.8,
                                      hovertemplate="%{x:.0f} rooms: %{y} draws<extra></extra>"))
        fig4.add_vline(x=left_rooms, line=dict(color=t.INK, width=1))
        fig4.add_annotation(x=left_rooms, y=1, yref="paper", text=" rooms left", showarrow=False, xanchor="left",
                            yanchor="top", font=dict(size=12, color=t.MUTED))
        fig4.update_layout(title="Forecast demand still to come at the current rate", xaxis_title="Rooms",
                           yaxis_title="Simulated outcomes", showlegend=False)
        t.chart(fig4, height=360)
    pressure = to_come / left_rooms if left_rooms else 9
    t.insight(f"Demand still to come is about <b>{to_come:.0f}</b> rooms for <b>{left_rooms:.0f}</b> left "
              f"({pressure:.0%} of what remains). " + (
                  "Demand exceeds supply, so the price should rise until the expected sell-out is just reached."
                  if pressure > 1.05 else "Supply exceeds demand, so the price trades a little rate for occupancy."
                  if pressure < 0.9 else "Supply and demand are close, so the current rate is near optimal."))

t.footnote("Synthetic hotel with realistic booking curves, seasonality and events. Methods mirror standard revenue-management practice.")
