"""Pricing & demand copilot: one product, framed, built, proven, run and valued, end to end."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import copilot as cp
from lab import demand as dm
from lab import theme as t
from lab import widgets as w

PILOT_SEED = 6  # one representative draw of the pilot; the estimator is unbiased across seeds
STAGES = ["Frame", "Data", "Model", "Decide", "Prove", "Run", "Value"]
PROVES = {
    "Frame": "Decision diagnostic",
    "Data": "Build · data foundations",
    "Model": "Build · applied modeling",
    "Decide": "Build · decision systems",
    "Prove": "Measurement & experimentation",
    "Run": "Build · MLOps & governance",
    "Value": "Advisory · adoption & ROI",
}

t.header(
    "Flagship product · Pricing & demand copilot",
    "From a pricing question to a product that pays for itself",
    "One decision followed the whole way: frame it, check the data, model demand and price response, recommend "
    "a rate inside guardrails, prove it in a live test, keep it healthy in production and count what it is worth. "
    "Seven stages, one synthetic 150-room hotel, every number computed live.",
)

# ---------------------------------------------------------------------------
# Stage state, kept in the URL so a stage can be linked to directly.
# ---------------------------------------------------------------------------
q = str(st.query_params.get("stage", "")).capitalize()
if "cp_stage" not in st.session_state:
    st.session_state.cp_stage = q if q in STAGES else STAGES[0]
    st.session_state.cp_stage__last = st.session_state.cp_stage


def _go(stage: str) -> None:
    st.session_state.cp_stage = stage


stage = w.segmented("Stage", STAGES, key="cp_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower()
idx = STAGES.index(stage)


def stage_head(title: str, why: str) -> None:
    st.markdown(
        f'<div class="cp-head"><h2>{t.esc(title)}</h2>'
        '</div>'
        f'<p class="cp-why">{t.esc(why)}</p>',
        unsafe_allow_html=True,
    )


def call(text: str) -> None:
    st.markdown(f'<div class="cp-call"><span class="e">The judgment call</span>{text}</div>', unsafe_allow_html=True)


def cards(items: list[tuple[str, str]]) -> None:
    body = "".join(f'<div class="lab-card"><h4>{t.esc(k)}</h4><p>{t.esc(v)}</p></div>' for k, v in items)
    st.markdown(f'<div class="cp-grid">{body}</div>', unsafe_allow_html=True)


# Shared, cached ingredients used by more than one stage.
world = dm.build_world()
hist = cp.rate_history()
el = cp.elasticity_table()
bt = dm.backtest()
bt["ape"] = bt["err"].abs() / bt["final"]
acc = bt.groupby("method")["ape"].mean()

fut = world.future.rename(columns={"on_books_now": "on_books"})
fc = dm.forecast(world, fut[["stay_date", "dow", "lead", "on_books"]], world.asof)
sd_by_lead = bt[bt["method"].eq("blend")].groupby("lead")["err"].std()
fc["sd"] = np.interp(fc["lead"], sd_by_lead.index, sd_by_lead.values)

iv = el[el["method"].eq("Instrumented by rate tests")].set_index("segment")["value"]
mix = {s: share for s, (share, _) in cp.SEGMENTS.items()}
blended_elasticity = float(sum(iv[s] * mix[s] for s in mix))

pre = cp.switchback(360, seed=99)
resid_sd = float((pre["revpar"] / pre["revpar"].rolling(7, min_periods=1).mean()).std())
rho_pre = float(np.corrcoef(pre["revpar"], pre["forecast"])[0, 1])

base_revpar = float(hist["revpar"].tail(90).mean())
base_occ = float(hist["sold"].tail(90).mean() / dm.CAPACITY)
base_adr = float((hist["rate"] * hist["sold"]).tail(90).sum() / hist["sold"].tail(90).sum())

# ===========================================================================
if stage == "Frame":
    stage_head(
        "Frame the decision before touching a model",
        "Most pricing projects fail at this stage, not at the modeling. The question is narrowed to one decision "
        "with an owner, a cadence, a measure of success, the guardrails it must respect and the baseline it has to beat.",
    )
    t.tiles([
        {"label": "RevPAR, last 90 nights", "value": t.money(base_revpar, 2), "accent": True,
         "note": "revenue per available room: the outcome measure"},
        {"label": "Occupancy", "value": t.pct(base_occ), "note": "rooms sold ÷ 150"},
        {"label": "Average daily rate", "value": t.money(base_adr, 2), "note": "set by hand from a weekly meeting"},
        {"label": "Nights sold out", "value": t.pct(hist["sold_out"].mean(), 1),
         "note": "demand on these nights is unknown"},
    ])
    cards([
        ("Decision", "The rate to publish for each of the next 60 nights, refreshed every morning."),
        ("Owner", "The revenue manager accepts, edits or rejects each recommendation and remains accountable."),
        ("Success measure", "RevPAR against a held-out control, not occupancy or rate alone, which can each be gamed."),
        ("Guardrails", "Rate floor and ceiling, a maximum daily move, and no change inside 48 hours of arrival."),
        ("Baseline to beat", "Last year's actuals and the manual rate. Both are measured, not assumed."),
        ("Out of scope", "Channel mix, group contracts and overbooking. Each one is a separate decision."),
    ])
    call("RevPAR is the target, not occupancy. A copilot measured on rooms sold will discount its way to a full "
         "hotel and a smaller margin. The success measure is written down here so that no stage can redefine it later.")
    with st.expander("Questions asked in the first week"):
        st.markdown("""
- Who changes rates today, how often, and what makes them override the system they already have?
- Which nights matter most: shoulder nights with spare rooms, or peak nights that sell out anyway?
- What would make a revenue manager stop trusting a recommendation? Those cases become the guardrails.
- Which rate changes in the last two years were deliberate tests? Without them, price response cannot be measured (see Data).
""")

# ===========================================================================
elif stage == "Data":
    stage_head(
        "Agree a data contract and test it on every run",
        "Four feeds: reservations, on-the-books snapshots by lead time, the rate history and an events calendar. "
        "Each run checks them first and blocks the pipeline when a check fails, so a bad night of data cannot set tomorrow's rates.",
    )
    qc = cp.quality_checks()
    passed = int(qc["passed"].sum())
    t.tiles([
        {"label": "Checks passing", "value": f"{passed} of {len(qc)}", "accent": True, "note": "run before every refresh"},
        {"label": "Nights of history", "value": f"{len(hist):,}", "note": f"to {world.asof:%b %d, %Y}"},
        {"label": "Booking snapshots", "value": f"{len(world.otb):,}", "note": "stay night × days before arrival"},
        {"label": "Nights with a rate test", "value": t.pct(hist["tested"].mean()), "accent": True,
         "note": "the random variation that identifies price response"},
    ])
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        rows = "".join(
            f'<li><span class="{"ok" if r.passed else "no"}">{"✓" if r.passed else "✕"}</span>'
            f'<span><b>{t.esc(r.check)}</b> · {t.esc(r.rule)}<br><small>{t.esc(r.result)} · on failure: {t.esc(r.on_fail)}</small></span></li>'
            for r in qc.itertuples()
        )
        st.markdown(f'<ul class="lab-checks">{rows}</ul>', unsafe_allow_html=True)
    with right:
        fig = go.Figure()
        for flag, name, color, size in ((False, "Rate set by hand", t.BASE, 6), (True, "Rate test night", t.S1, 7)):
            d = hist[hist["tested"].eq(flag) & ~hist["sold_out"]]
            fig.add_trace(go.Scatter(x=d["rate"], y=d["sold"], mode="markers", name=name,
                                     marker=dict(color=color, size=size, opacity=0.75, line=dict(width=0)),
                                     hovertemplate="$%{x:.0f}: %{y:.0f} rooms<extra>" + name + "</extra>"))
        fig.update_layout(title="Higher rates, more rooms sold? The raw data says so", xaxis_title="Rate charged ($)",
                          yaxis_title="Rooms sold", xaxis=dict(tickprefix="$"))
        t.chart(fig, height=380)
    t.insight("Rates went up on busy nights because the team saw demand coming, so price and volume rise together. "
              "Read naively, this data says guests <b>prefer higher prices</b>. The rate-test nights are what make "
              "the true response measurable.")
    call("Sold-out nights are excluded from estimation rather than treated as demand. A sold-out night shows "
         "only that demand reached 150 rooms, not how far above 150 it went. Treating it as demand would bias "
         "every estimate downward.")

# ===========================================================================
elif stage == "Model":
    stage_head(
        "Model two things: how many will book, and how they react to price",
        "Demand is forecast from booking pace blended with seasonality and backtested as of each past date. "
        "Price response is estimated separately for each segment, three ways, to show why the method matters.",
    )
    gain = 1 - acc["blend"] / acc["naive"]
    t.tiles([
        {"label": "Forecast error · copilot", "value": t.pct(acc["blend"], 1), "accent": True, "note": "mean absolute %, 120-night backtest"},
        {"label": "Forecast error · last year", "value": t.pct(acc["naive"], 1), "note": "the baseline from Frame"},
        {"label": "Improvement", "value": t.pct(gain), "accent": True, "note": "relative reduction in error"},
        {"label": "Blended price elasticity", "value": f"{blended_elasticity:.2f}".replace("-", "−"),
         "note": "+10% rate ≈ " + f"{(1.1 ** blended_elasticity - 1) * 100:.0f}% rooms".replace("-", "−")},
    ])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        fig = go.Figure()
        colors = {"Raw correlation": t.BASE, "With calendar controls": t.S2, "Instrumented by rate tests": t.S1}
        for method, color in colors.items():
            d = el[el["method"].eq(method)]
            fig.add_trace(go.Scatter(
                x=d["value"], y=d["segment"], mode="markers", name=method,
                marker=dict(color=color, size=12, line=dict(color="#fff", width=1.5)),
                error_x=dict(type="data", symmetric=False, array=d["hi"] - d["value"], arrayminus=d["value"] - d["lo"],
                             color=color, thickness=1.5, width=0),
                hovertemplate="%{y}: %{x:.2f}<extra>" + method + "</extra>"))
        truth = el.drop_duplicates("segment")
        fig.add_trace(go.Scatter(x=truth["truth"], y=truth["segment"], mode="markers", name="True value (known here)",
                                 marker=dict(symbol="line-ns-open", size=26, color=t.INK, line=dict(width=2)),
                                 hovertemplate="%{y}: true %{x:.2f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=t.GRID, width=1))
        fig.update_layout(title="Price elasticity by segment, with 95% intervals", xaxis_title="Elasticity (% change in rooms per 1% rate)",
                          yaxis=dict(categoryorder="array", categoryarray=["Business", "Leisure"]),
                          legend=dict(orientation="h", y=1.02), margin=dict(t=110))
        t.chart(fig, height=400)
    with right:
        g = bt.groupby(["lead", "method"])["ape"].mean().unstack()
        names = {"naive": "Last year", "pickup": "Booking pace only", "blend": "Copilot: pace + seasonality"}
        fig2 = go.Figure()
        for m, color, width in (("naive", t.S2, 2), ("pickup", t.S3, 2), ("blend", t.S1, 2.8)):
            fig2.add_trace(go.Scatter(x=g.index, y=g[m] * 100, mode="lines+markers", name=names[m],
                                      line=dict(color=color, width=width), marker=dict(size=7),
                                      hovertemplate="%{x} days out: %{y:.1f}%<extra>" + names[m] + "</extra>"))
        fig2.update_layout(title="Forecast error by days before arrival", xaxis_title="Days before arrival",
                           yaxis_title="Mean absolute error (%)", yaxis=dict(rangemode="tozero"),
                           legend=dict(orientation="h", y=1.02), margin=dict(t=110))
        t.chart(fig2, height=400)
    t.insight("The raw correlation puts business elasticity <b>above zero</b>, which would tell the copilot to raise "
              "rates without limit. Calendar controls remove part of the bias; only the random rate tests recover "
              "the true response. Leisure guests are close to three times as price-sensitive as business guests.")
    with st.expander("Inside the regression: the two linear models behind the leisure estimate", expanded=False):
        from lab import copilot_reg as CR
        R = CR.regression("Leisure")
        rl, rr = st.columns([1, 1.1], gap="large")
        with rl:
            figr = go.Figure()
            figr.add_trace(go.Scatter(x=R["rx"], y=R["ry"], mode="markers", marker=dict(size=5, color=t.S1, opacity=0.5),
                                      name="Nights", hovertemplate="rate %{x:+.3f} · rooms %{y:+.3f}<extra></extra>"))
            xs = np.linspace(R["rx"].min(), R["rx"].max(), 2)
            figr.add_trace(go.Scatter(x=xs, y=R["slope"] * xs, mode="lines", line=dict(color=t.INK, width=2), name=f"Slope {R['slope']:.2f}"))
            figr.update_layout(title="Demand vs the randomized part of price, calendar removed", xaxis_title="log rate driven by tests (residual)",
                               yaxis_title="log leisure rooms (residual)", legend=dict(orientation="h", y=1.02), margin=dict(t=100))
            t.chart(figr, height=340)
        with rr:
            st.dataframe(R["coefs"].rename(columns={"term": "Stage 2 term", "estimate": "Estimate", "se": "Std. error", "t": "t"}),
                         hide_index=True, width="stretch", height=340,
                         column_config={"Estimate": st.column_config.NumberColumn(format="%.3f"),
                                        "Std. error": st.column_config.NumberColumn(format="%.3f"), "t": st.column_config.NumberColumn(format="%.1f")})
        st.markdown(f"Two ordinary least-squares regressions on **{R['n']}** nights. **Stage 1** regresses log rate on the random test "
                    f"offset and the calendar: a 10% test offset moves the rate by about {R['first_stage'] * 10:.0f}% (F = {R['f_stat']:.0f}, far above "
                    f"the usual weak-instrument line of 10). **Stage 2** regresses log rooms on the stage-1 fitted rate and the same calendar terms. "
                    f"Its slope is the elasticity, **{R['slope']:.2f}** against a true {R['truth']:.1f}; R² {R['r2']:.2f}. "
                    "On a log–log scale a linear model is the right shape, every coefficient means something to a revenue manager, and "
                    "the guardrails in Decide are built on it. A tree model might forecast a little better and could not hand the pricing "
                    "team one number to argue with.")
    call("The instrumented estimate is used even though its interval is wider. A precise estimate of the wrong "
         "number is worse than a noisy estimate of the right one. The wide business interval goes into the "
         "decision as uncertainty, and it is the reason for more rate tests on weekday nights.")
    with st.expander("Method notes"):
        st.markdown("""
- **Forecast:** rooms on the books plus average remaining pickup by weekday and lead time, blended with a seasonal baseline. The blend weight moves from history (far out) to pace (close in).
- **Backtest:** each method re-run as of 3–45 days before each of the last 120 nights, using only the data known at that point.
- **Elasticity:** log rooms on log rate. The instrumented version runs two-stage least squares, with the random test offset as the instrument and weekday, seasonality and trend as controls. Sold-out nights are excluded.
- **In production:** demand on sell-out nights would be unconstrained, elasticities would be hierarchical across properties, and competitor rates would be added as a control.
""")

# ===========================================================================
elif stage == "Decide":
    stage_head(
        "Recommend a rate, inside guardrails, with its reasons",
        "For any future night the copilot combines demand still to come, rooms left, segment price response and "
        "forecast uncertainty into an expected-revenue curve. It then picks the best rate the guardrails allow and "
        "explains the choice in terms a revenue manager can check.",
    )
    opts = {f"{r.stay_date:%a %b %d} · {int(r.on_books)} booked · {int(r.lead)} days out": i
            for i, r in fc.iterrows() if r.lead >= 3}
    with st.container(border=True):
        a, b, c, d = st.columns([1.7, 1, 1, 1])
        pick = a.selectbox("Stay night", list(opts), index=min(8, len(opts) - 1))
        ref = b.number_input("Current rate ($)", 80, 1000, 189, 1)
        max_move = c.slider("Max move per day", 0.05, 0.40, 0.15, 0.05, format="%.2f",
                            help="Guardrail agreed in Frame: the largest single-day change, as a share of the current rate.")
        floor = d.number_input("Rate floor ($)", 50, 1000, 129, 1)
        e, f, g = st.columns(3)
        leisure_share = e.slider("Leisure share of remaining demand", 0.2, 0.9, 0.6, 0.05)
        late_share = f.slider("Share booking late at full rate", 0.0, 0.6, 0.25, 0.05)
        disc = g.slider("Advance rate as % of full", 40, 95, 75, 5) / 100
    row = fc.loc[opts[pick]]
    rooms_left = float(dm.CAPACITY - row["on_books"])
    to_come = max(float(row["blend"] - row["on_books"]), 0.5)
    eps = leisure_share * iv["Leisure"] + (1 - leisure_share) * iv["Business"]
    res = cp.optimise_rate(to_come, float(row["sd"]), rooms_left, ref, eps, floor, 999, max_move)
    k, kf, kr = res["k"], res["k_free"], res["k_ref"]
    protect = cp.protection_level(to_come, float(row["sd"]), late_share, disc, rooms_left)
    uplift = res["rev"][k] - res["rev"][kr]
    capped = abs(res["grid"][kf] - res["rate"]) > 1

    t.tiles([
        {"label": "Recommended rate", "value": f"${res['rate']:,.0f}", "accent": True,
         "note": f"{res['rate'] / ref - 1:+.0%} vs current".replace("-", "−")},
        {"label": "Expected rooms sold", "value": f"{res['sold'][k]:.0f} of {rooms_left:.0f} left",
         "note": f"{(row['on_books'] + res['sold'][k]) / dm.CAPACITY:.0%} final occupancy"},
        {"label": "Expected extra revenue", "value": t.money(uplift), "accent": True, "note": f"vs holding ${ref:,}"},
        {"label": "Protect for late demand", "value": f"{protect:.0f} rooms", "note": "close advance rate at this level"},
    ])
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_vrect(x0=res["lo"], x1=res["hi"], fillcolor="rgba(127,208,190,.16)", line_width=0)
        fig.add_annotation(x=(res["lo"] + res["hi"]) / 2, y=1, yref="paper", text="allowed by guardrails",
                           showarrow=False, yanchor="top", font=dict(size=12, color=t.PETROL))
        fig.add_trace(go.Scatter(x=res["grid"], y=res["rev"], mode="lines", line=dict(color=t.S1, width=2.5),
                                 customdata=res["sold"], showlegend=False,
                                 hovertemplate="$%{x:.0f} → $%{y:,.0f}, %{customdata:.0f} rooms<extra></extra>"))
        fig.add_trace(go.Scatter(x=[res["rate"]], y=[res["rev"][k]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=13, color=t.PETROL, line=dict(color="#fff", width=2))))
        if capped:
            fig.add_trace(go.Scatter(x=[res["grid"][kf]], y=[res["rev"][kf]], mode="markers", showlegend=False,
                                     hoverinfo="skip", marker=dict(size=11, color="#fff", line=dict(color=t.S1, width=2))))
        fig.add_vline(x=ref, line=dict(color=t.S2, width=1))
        fig.update_layout(title="Expected revenue from the rooms left, by rate", xaxis_title="Rate ($)",
                          yaxis_title="Expected revenue ($)", xaxis=dict(tickprefix="$"),
                          yaxis=dict(tickprefix="$", tickformat="~s"))
        t.chart(fig, height=380)
    with right:
        pressure = to_come / rooms_left if rooms_left else 9
        reasons = [
            f"Forecast demand still to come: <b>{to_come:.0f}</b> rooms for <b>{rooms_left:.0f}</b> left ({pressure:.0%}).",
            f"Remaining guests are {leisure_share:.0%} leisure, so blended elasticity is <b>{eps:.2f}</b>.".replace("-", "−"),
            f"Forecast uncertainty ±{row['sd']:.0f} rooms at {int(row['lead'])} days out, priced in through 4,000 simulated outcomes.",
        ]
        if capped:
            reasons.append(f"Unconstrained optimum is <b>${res['grid'][kf]:,.0f}</b>; the daily-move guardrail caps it at ${res['rate']:,.0f}.")
        if res["rate"] <= floor + 0.5:
            reasons.append("The rate floor is binding.")
        lis = "".join(f'<li><span class="ok">→</span><span>{r}</span></li>' for r in reasons)
        st.markdown(f'<div class="lab-reco"><div class="e">Recommendation · {row.stay_date:%a %b %d}</div>'
                    f'<div class="t">Publish ${res["rate"]:,.0f}</div>'
                    f'<div class="d">Hold {protect:.0f} rooms for late full-rate demand.</div></div>'
                    f'<ul class="lab-checks" style="margin-top:10px">{lis}</ul>', unsafe_allow_html=True)
    call("Guardrails are part of the product, not a restriction on it. A revenue manager who sees a 40% overnight "
         "jump will override every recommendation that follows. Capping the move gives up a little expected revenue "
         "today and keeps the adoption that every later dollar depends on.")

# ===========================================================================
elif stage == "Prove":
    stage_head(
        "Prove it with a test the business will believe",
        "Guest-level A/B testing does not work for hotel pricing: both arms draw on the same rooms, so the "
        "treatment changes what the control can sell. The copilot is tested with a switchback instead. Whole nights "
        "alternate, at random, between manual and copilot pricing.",
    )
    with st.container(border=True):
        a, b, c = st.columns(3)
        mde = a.slider("Smallest RevPAR lift worth detecting", 0.02, 0.08, 0.04, 0.005, format="%.3f")
        use_cuped = b.toggle("Adjust for the forecast (CUPED)", value=True,
                             help="Uses each night's pre-registered forecast as a covariate to remove predictable noise.")
        power = c.select_slider("Power", [0.7, 0.8, 0.9], value=0.8)
    n_raw = cp.nights_needed(resid_sd, 1.0, mde, 0.0, power=power)
    n_adj = cp.nights_needed(resid_sd, 1.0, mde, rho_pre, power=power)
    n_plan = n_adj if use_cuped else n_raw
    n_run = int(np.ceil(max(n_plan, 28) / 14) * 14)
    sim = cp.switchback(min(n_run, 500), seed=PILOT_SEED)
    r = cp.readout(sim)
    lift, se = (r["cuped"], r["cuped_se"]) if use_cuped else (r["raw"], r["raw_se"])
    lo, hi = lift - 1.96 * se, lift + 1.96 * se
    ship = lo > 0
    t.tiles([
        {"label": "Nights needed", "value": f"{n_plan:,}", "accent": True,
         "note": f"run as {n_run} nights ({n_run // 7} weeks), whole weeks"},
        {"label": "Without forecast adjustment", "value": f"{n_raw:,}", "note": "nights for the same power"},
        {"label": "Measured RevPAR lift", "value": t.pct(lift, 1, signed=True), "accent": True,
         "note": f"95% interval {t.pct(lo, 1, signed=True)} to {t.pct(hi, 1, signed=True)}"},
        {"label": "Decision", "value": "Ship" if ship else "Keep testing", "note": "pre-registered: ship if the interval excludes zero"},
    ])
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        fig = go.Figure()
        for arm, name, color in ((0, "Manual pricing", t.S2), (1, "Copilot pricing", t.S1)):
            d = sim[sim["arm"].eq(arm)]
            fig.add_trace(go.Bar(x=d["night"], y=d["revpar"], name=name, marker=dict(color=color),
                                 hovertemplate="Night %{x}: $%{y:.0f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=sim["night"], y=sim["forecast"], mode="lines", name="Forecast (covariate)",
                                 line=dict(color=t.INK, width=1, dash="dot"), hoverinfo="skip"))
        fig.update_layout(title="Pilot: each night's RevPAR by arm", xaxis_title="Night of pilot",
                          yaxis_title="RevPAR ($)", yaxis=dict(tickprefix="$"), bargap=0.15)
        t.chart(fig, height=360)
    with right:
        fig2 = go.Figure()
        for label, v, s_, color in (("Unadjusted", r["raw"], r["raw_se"], t.S2), ("Forecast-adjusted", r["cuped"], r["cuped_se"], t.S1)):
            fig2.add_trace(go.Scatter(x=[v * 100], y=[label], mode="markers", showlegend=False,
                                      marker=dict(size=13, color=color),
                                      error_x=dict(type="data", array=[1.96 * s_ * 100], color=color, thickness=2, width=0),
                                      hovertemplate=f"{label}: %{{x:.1f}}% ± {1.96 * s_ * 100:.1f}<extra></extra>"))
        fig2.add_vline(x=0, line=dict(color=t.INK, width=1))
        fig2.add_vline(x=mde * 100, line=dict(color=t.GRID, width=1, dash="dot"))
        fig2.update_layout(title="Read-out: RevPAR lift with 95% interval", xaxis_title="Lift (%)",
                           yaxis=dict(categoryorder="array", categoryarray=["Forecast-adjusted", "Unadjusted"]))
        t.chart(fig2, height=360)
    t.insight(f"Using the forecast as a covariate removes about <b>{rho_pre ** 2:.0%}</b> of nightly noise, which "
              f"cuts the test from {n_raw:,} to {n_adj:,} nights. The interval narrows by about half, with no "
              "extra traffic and no change to the design.")
    call("The stopping rule, the metric and the covariate are written down before the first night. Results are not "
         "checked daily, and the test is not stopped early when it looks good. The peeking problem turns most "
         "early-stopped pricing wins into false positives.")
    try:
        st.page_link("views/experiments.py", label="Method note: peeking, power and CUPED in depth",
                     icon=":material/arrow_outward:")
    except Exception:  # page run on its own, outside the navigation
        pass

# ===========================================================================
elif stage == "Run":
    stage_head(
        "Run it: monitor, detect drift, retrain on evidence",
        "After launch the copilot reports its own health every week: forecast accuracy against outcomes and the "
        "drift in how far ahead guests book. Retraining is triggered by agreed thresholds, not by the calendar or by alarm.",
    )
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        shift = a.toggle("Competitor opens, wk 12", value=True)
        mape_limit = b.slider("Accuracy alert (MAPE)", 0.08, 0.16, 0.11, 0.01, format="%.2f")
        weeks_over = c.slider("Weeks over before retrain", 1, 4, 2)
        psi_limit = d.select_slider("Drift alert (PSI)", [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 9.99], value=0.20,
                                    format_func=lambda v: "off" if v > 5 else f"{v:.2f}")
    mon = cp.monitoring(shift, mape_limit, weeks_over, psi_limit)
    trig = mon[mon["trigger"]]
    tw = int(trig["week"].iloc[0]) if len(trig) else None
    degraded = int((mon["mape"] > 0.09).sum())
    t.tiles([
        {"label": "Retrain triggered", "value": f"Week {tw}" if tw else "Not needed", "accent": True,
         "note": f"by {trig['by'].iloc[0]}" if tw else "no threshold crossed"},
        {"label": "Weeks running degraded", "value": f"{degraded}", "note": "error above 9%, the launch level plus margin"},
        {"label": "Median weekly error", "value": t.pct(mon["mape"].median(), 1), "note": "26 weeks after launch"},
        {"label": "Peak drift (PSI)", "value": f"{mon['psi'].max():.2f}", "note": "booking lead-time distribution"},
    ])
    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure(go.Scatter(x=mon["week"], y=mon["mape"] * 100, mode="lines+markers", line=dict(color=t.S1, width=2.5),
                                   marker=dict(size=6), name="Weekly forecast error",
                                   hovertemplate="Week %{x}: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=mape_limit * 100, line=dict(color=t.S2, width=1.2, dash="dot"))
        fig.add_annotation(x=1, y=mape_limit * 100, text="alert", showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=12, color=t.S2))
        if tw:
            fig.add_vline(x=tw + 0.5, line=dict(color=t.PETROL, width=1))
            fig.add_annotation(x=tw + 0.5, y=1, yref="paper", text=" retrained", showarrow=False, xanchor="left",
                               yanchor="top", font=dict(size=12, color=t.PETROL))
        fig.update_layout(title="Forecast accuracy, weekly", xaxis_title="Week since launch",
                          yaxis_title="Mean absolute error (%)", yaxis=dict(rangemode="tozero"), showlegend=False)
        t.chart(fig, height=340)
    with right:
        fig2 = go.Figure(go.Bar(x=mon["week"], y=mon["psi"],
                                marker=dict(color=[t.S2 if v > psi_limit else t.S3 for v in mon["psi"]]),
                                hovertemplate="Week %{x}: PSI %{y:.2f}<extra></extra>"))
        fig2.add_hline(y=psi_limit, line=dict(color=t.S2, width=1.2, dash="dot"))
        fig2.update_layout(title="Drift in booking lead times (PSI)", xaxis_title="Week since launch",
                           yaxis_title="Population stability index", showlegend=False)
        t.chart(fig2, height=340)
    if shift:
        acc_only = cp.monitoring(shift, mape_limit, weeks_over, 9.99)
        aw = acc_only.loc[acc_only["trigger"], "week"]
        aw = int(aw.iloc[0]) if len(aw) else None
        if tw and aw and aw > tw:
            t.insight(f"Guests changed how far ahead they book before the forecast errors arrived. The drift alert "
                      f"retrains in <b>week {tw}</b>; an accuracy-only rule would have waited until week {aw}, "
                      f"pricing on a stale model for <b>{aw - tw} more week{'s' if aw - tw > 1 else ''}</b>.")
        elif tw:
            t.insight(f"With these thresholds the accuracy alert fires first, in <b>week {tw}</b>. Lower the drift "
                      "threshold to catch the change in booking behavior earlier, at the cost of more false alarms.")
        else:
            t.insight("No threshold was crossed, so the copilot kept pricing on a model the market had moved past. "
                      "Thresholds that never fire are as costly as ones that fire constantly.")
    st.markdown("**Model card**")
    st.dataframe(pd.DataFrame([
        ("Purpose", "Recommend a nightly rate and a protection level for the next 60 nights."),
        ("Decision owner", "Revenue manager. Recommendations are advisory; every override is logged with a reason."),
        ("Training data", f"{len(hist)} nights of reservations, rates and snapshots; sold-out nights excluded from elasticity."),
        ("Performance", f"Forecast MAPE {acc['blend']:.1%} vs {acc['naive']:.1%} for last year; RevPAR lift measured by switchback."),
        ("Known limits", "No competitor rates; business elasticity is uncertain; events must be entered by hand."),
        ("Retrain policy", f"MAPE above {mape_limit:.0%} for {weeks_over} week(s), or lead-time PSI above {psi_limit:.2f}."),
        ("Fallback", "If a data check blocks, publish yesterday's rate and alert the owner. Never publish a stale recommendation."),
    ], columns=["Field", "Entry"]), hide_index=True, width="stretch")
    call("The override log is the most valuable monitoring feed. When revenue managers override in the same "
         "direction on the same kind of night, the model is missing something they know, and that becomes the "
         "next feature.")

# ===========================================================================
elif stage == "Value":
    stage_head(
        "Count the value, and what it depends on",
        "A measured lift is not the value of the product. Value is the lift times the share of recommendations the "
        "business actually follows, across the rooms it runs on, minus what it costs to build and run. Adoption "
        "usually decides the result.",
    )
    n_default = int(np.ceil(max(cp.nights_needed(resid_sd, 1.0, 0.04, rho_pre), 28) / 14) * 14)
    measured = cp.readout(cp.switchback(n_default, seed=PILOT_SEED))["cuped"]
    with st.container(border=True):
        a, b, c = st.columns(3)
        hotels = a.slider("Hotels in the portfolio", 1, 60, 4)
        adoption = b.slider("Recommendations followed", 0.2, 1.0, 0.6, 0.05,
                            help="Share of nights where the published rate follows the copilot.")
        lift_used = c.slider("RevPAR lift (from Prove)", 0.0, 0.08, float(round(measured, 3)), 0.005, format="%.3f")
        d, e = st.columns(2)
        build = d.number_input("Build cost ($)", 0, 3_000_000, 350_000, 10_000)
        run = e.number_input("Run cost per year ($)", 0, 1_000_000, 120_000, 5_000)
    room_nights = hotels * dm.CAPACITY * 365
    gross = base_revpar * lift_used * room_nights
    adopted = gross * adoption
    net = adopted - run
    payback = (build / (net / 12)) if net > 0 else float("inf")
    t.tiles([
        {"label": "Gross annual uplift", "value": t.money(gross), "note": f"{lift_used:.1%} on {room_nights:,} room-nights"},
        {"label": "After adoption", "value": t.money(adopted), "accent": True, "note": f"{adoption:.0%} of nights follow the copilot"},
        {"label": "Net of run cost", "value": t.money(net), "accent": True, "note": "per year"},
        {"label": "Payback", "value": f"{payback:.1f} months" if np.isfinite(payback) else "Never", "note": f"on a {t.money(build)} build"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["Gross uplift", "Not adopted", "Run cost", "Net value"],
            y=[gross, -(gross - adopted), -run, 0],
            text=[t.money(gross), "−" + t.money(gross - adopted), "−" + t.money(run), t.money(net)],
            connector=dict(line=dict(color=t.GRID)),
            increasing=dict(marker=dict(color=t.S1)), decreasing=dict(marker=dict(color=t.S2)),
            totals=dict(marker=dict(color=t.PETROL)),
            textposition="outside",
            hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"))
        fig.update_layout(title="Annual value, year one", yaxis=dict(tickprefix="$", tickformat="~s"), showlegend=False)
        t.chart(fig, height=360)
    with right:
        ad = np.linspace(0.2, 1.0, 17)
        fig2 = go.Figure(go.Scatter(x=ad * 100, y=base_revpar * lift_used * room_nights * ad - run, mode="lines",
                                    line=dict(color=t.S1, width=2.5), hovertemplate="%{x:.0f}% followed: %{y:$,.0f}<extra></extra>"))
        fig2.add_hline(y=0, line=dict(color=t.INK, width=1))
        fig2.add_vline(x=adoption * 100, line=dict(color=t.S2, width=1))
        fig2.update_layout(title="Net value by adoption", xaxis_title="Recommendations followed (%)",
                           yaxis=dict(tickprefix="$", tickformat="~s"), showlegend=False)
        t.chart(fig2, height=360)
    t.insight(f"Each 10 points of adoption is worth <b>{t.money(gross * 0.1)}</b> a year here. That is why the guardrails, "
              "the reasons and the override log were designed in from Frame onward rather than added at the end.")
    st.markdown("**Delivery plan**")
    cards([
        ("Weeks 1–2 · Diagnostic", "Frame the decision, audit the data, size the prize, agree guardrails and the success measure."),
        ("Weeks 3–8 · Build", "Data contract and checks, forecast and elasticity models, recommendation service with reasons."),
        ("Weeks 9–12 · Prove", "Switchback pilot on two properties, pre-registered read-out, go or no-go."),
        ("Ongoing · Run & advise", "Monitoring, retrain policy, override reviews and a quarterly value report."),
    ])
    st.markdown("**What I would do differently in production**")
    st.markdown("""
- **Unconstrain demand first** on sell-out nights, so that price response on peak nights is not understated.
- **Pool elasticities across properties** with a hierarchical model, so a new hotel starts from its peers instead of from nothing.
- **Make rate tests a standing budget**, a small share of nights every month, so price response stays measurable as the market moves.
- **Add competitor rates as a control**, and treat channel mix as the next decision, not part of this one.
""")
    call("The number that goes in front of a CFO is the net figure at measured adoption, with the interval from "
         "Prove beside it, not the gross figure at 100%. Under-promising here is what pays for the second product.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])
if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Synthetic hotel, demand, rate tests and pilot. Methods mirror standard revenue-management and experimentation practice.")
