import html as _html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import market_engine as me
from lab import strategy_engine as se
from lab import theme as t
from lab import widgets as w

DIM_COLORS = {"demand": "#CBFA7C", "access": "#B9A6EC", "risk": "#96DFCB", "economics": "#EF936F", "competition": "#A6B8AE"}
STEPS = ["1 · Frame", "2 · Explore", "3 · Decide", "4 · Economics", "5 · Brief"]
PILL = {
    "observed": '<span class="lab-pill">Observed</span>',
    "modeled": '<span class="lab-pill warn">Modeled</span>',
    "illustrative": '<span class="lab-pill off">Illustrative</span>',
}

# ---------------------------------------------------------------------------
# State that survives switching steps. Streamlit drops a widget's value when the widget is
# not drawn, so every control mirrors its value into a plain dict kept in session state.
# ---------------------------------------------------------------------------
STORE = st.session_state.setdefault("_ps_store", {})
DEFAULTS = {
    "ps_product": "purchase", "ps_area": "Metropolitan", "ps_states": [], "ps_minpop": 100_000,
    "ps_ready": "Strong", "ps_maxrisk": 85, "ps_horizon": 3, "ps_scenario": "Base",
    **{f"ps_w_{k}": v for k, v in se.DEFAULT_WEIGHTS.items()},
}


def S(key):
    return STORE.get(key, DEFAULTS.get(key))


def ctl(fn, key, default, *args, **kwargs):
    if key not in st.session_state:
        st.session_state[key] = STORE.get(key, default)
    value = fn(*args, key=key, **kwargs)
    STORE[key] = value
    return value


def reset_all():
    STORE.clear()
    for key in [k for k in st.session_state if k.startswith("ps_") and k != "ps_step"]:
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _base():
    raw, live = me.load_raw()
    prepared = me.prepare(raw)
    meta = {}
    if live and me.META_FILE.exists():
        try:
            meta = json.loads(me.META_FILE.read_text())
        except ValueError:
            meta = {}
    return prepared, live, meta


@st.cache_data(show_spinner=False)
def _scores(product: str):
    prepared, _, _ = _base()
    return se.dimension_scores(prepared, product)


@st.cache_data(show_spinner=False)
def _stress(product: str, ids: tuple, weights: tuple):
    scores = _scores(product).loc[list(ids)]
    return se.stress_test(scores, dict(zip(se.DIM_KEYS, weights)))


prepared, live, meta = _base()
by_key = prepared.set_index("market_key", drop=False)

t.header(
    "Case study · Mortgage product strategy, on public data",
    "Where should a lender expand, and what is the evidence?",
    "A hypothetical lender must choose where to expand one product line. Public Census, HMDA and FHFA data supply the "
    "evidence, your strategy weights drive the ranking, hard screens stay separate from preferences, and an executive "
    "brief writes itself from the choices you make. A research demonstration, not a real lender's decision.",
)
if not live:
    st.markdown('<div class="lab-insight"><b>Sample data.</b> This preview runs on a synthetic market table with fictional '
                "market names. The published version uses the public Census, HMDA and FHFA build.</div>", unsafe_allow_html=True)

_requested = str(st.query_params.get("step", "frame")).lower()
_by_name = {"frame": 0, "explore": 1, "decide": 2, "economics": 3, "brief": 4}
step = w.segmented("Step", STEPS, key="ps_step", default=STEPS[_by_name.get(_requested, 0)], label_visibility="collapsed")
st.write("")


# ---------------------------------------------------------------------------
# Analysis shared by every step (reads the persisted state)
# ---------------------------------------------------------------------------
def analysis():
    product = S("ps_product")
    scores = _scores(product)
    gate = se.screen(prepared, scores, area_type=S("ps_area"), states=S("ps_states"), min_population=S("ps_minpop"),
                     min_readiness=S("ps_ready"), max_risk_pct=S("ps_maxrisk"))
    ok = scores[gate.passes]
    weights = {k: S(f"ps_w_{k}") for k in se.DIM_KEYS}
    if ok.empty:
        return dict(product=product, scores=scores, gate=gate, ok=ok, weights=weights, comp=pd.Series(dtype=float),
                    order=[], stress=None)
    comp = se.composite(ok, weights).dropna().sort_values(ascending=False)
    ok = ok.loc[comp.index]
    stress = _stress(product, tuple(ok.index), tuple(se.normalise(weights)[k] for k in se.DIM_KEYS))
    return dict(product=product, scores=scores, gate=gate, ok=ok, weights=weights, comp=comp, order=list(comp.index),
                stress=stress)


def mname(key):
    return by_key.loc[key, "market_name"]


def scope_line(a):
    states = S("ps_states")
    where = ", ".join(states) if states else "all states"
    st.caption(f"{se.PRODUCTS[a['product']]} · {S('ps_area')} areas · {where} · "
               f"{len(a['order']):,} markets pass the screens. Change scope in step 1.")


# ===========================================================================
# 1 · Frame
# ===========================================================================
if step == STEPS[0]:
    st.markdown(
        f'<div class="lab-insight"><b>The strategic question.</b> A residential mortgage lender is weighing where to put its '
        f"next product investment. Market growth does not automatically mean product opportunity. The question is where "
        f"demand, access, risk, economics and competition intersect, and how sure we can be.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, pill, head, body in [
        (c1, "observed", "Observed evidence", "Public figures as published: Census population, HMDA lending activity, FHFA house prices."),
        (c2, "modeled", "Modeled estimates", "Scores and pool sizes derived from the observed figures, with the method shown."),
        (c3, "illustrative", "Illustrative scenarios", "Hypothetical pricing, cost and share assumptions that show a method. Not quotes or results."),
    ]:
        col.markdown(f'<div class="lab-card">{PILL[pill]}<h3>{head}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Five dimensions decide it")
    dim = w.segmented("Dimension", [se.DIMENSIONS[k][0] for k in se.DIM_KEYS], key="ps_dim", default="Demand",
                      label_visibility="collapsed")
    key = next(k for k in se.DIM_KEYS if se.DIMENSIONS[k][0] == dim)
    d = se.DIM_DETAIL[key]
    st.markdown(
        f'<div class="lab-card"><h3>{t.esc(dim)}</h3><p>{t.esc(se.DIMENSIONS[key][1])}</p>'
        f'<ul class="lab-checks"><li><b>Data</b> · {t.esc(d["data"])}</li><li><b>Approach</b> · {t.esc(d["approach"])}</li>'
        f'<li><b>Why it matters</b> · {t.esc(d["impact"])}</li></ul></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Set the scope")
    with st.container(border=True):
        a1, a2, a3 = st.columns([1.2, 1, 1.2])
        with a1:
            ctl(st.selectbox, "ps_product", DEFAULTS["ps_product"], "Product line", list(se.PRODUCTS),
                format_func=lambda k: se.PRODUCTS[k])
        with a2:
            ctl(st.selectbox, "ps_area", DEFAULTS["ps_area"], "Market type", ["Metropolitan", "Micropolitan"])
        with a3:
            states = sorted(prepared["primary_state"].dropna().unique())
            ctl(st.multiselect, "ps_states", [], "States", states, placeholder="All states")
        b1, b2, b3 = st.columns(3)
        with b1:
            ctl(st.select_slider, "ps_minpop", DEFAULTS["ps_minpop"], "Population floor", [0, 50_000, 100_000, 250_000, 500_000, 1_000_000],
                format_func=lambda v: "None" if v == 0 else f"{v:,}")
        with b2:
            ctl(st.select_slider, "ps_ready", DEFAULTS["ps_ready"], "Evidence required", ["Partial", "Strong", "Complete"],
                help="Complete: all five score blocks observed. Strong: four of five.")
        with b3:
            ctl(st.slider, "ps_maxrisk", DEFAULTS["ps_maxrisk"], "Risk limit (percentile)", 50, 100, step=5,
                help="Markets whose risk percentile is above this are screened out, whatever their weighted score.")
        st.caption("These are hard screens. They remove markets before any weighting happens, so a high score can never "
                   "buy back a market that fails them.")
    a = analysis()
    gate = a["gate"]
    in_type = gate[gate.reason != "Other market type"]
    t.tiles([
        {"label": "Markets in the table", "value": f"{len(prepared):,}", "note": "Metro and micro areas"},
        {"label": "Pass every screen", "value": f"{len(a['order']):,}", "note": "Ranked in step 3", "accent": True},
        {"label": "Screened out", "value": f"{int((~in_type.passes).sum()):,}", "note": "Of the chosen market type"},
    ])
    if st.button("Reset everything to baseline", icon=":material/restart_alt:"):
        reset_all()
        st.rerun()

# ===========================================================================
# 2 · Explore
# ===========================================================================
elif step == STEPS[1]:
    a = analysis()
    scope_line(a)
    if not a["order"]:
        st.warning("No markets pass the current screens. Relax them in step 1.")
        st.stop()
    ok = a["ok"]
    top10 = a["order"][:10]
    st.markdown("##### Demand against competition " + PILL["modeled"], unsafe_allow_html=True)
    frame = pd.DataFrame({
        "market": [mname(k) for k in ok.index], "demand": ok["demand"].to_numpy(), "competition": ok["competition"].to_numpy(),
        "risk": (100 - ok["risk"]).to_numpy(), "pop": by_key.loc[ok.index, "population_latest"].to_numpy(),
        "top": [k in top10 for k in ok.index],
    })
    fig = go.Figure()
    rest = frame[~frame.top]
    fig.add_trace(go.Scatter(x=rest.competition, y=rest.demand, mode="markers", name="Other markets", text=rest.market,
                             marker=dict(size=7, color="rgba(166,184,174,.45)"),
                             hovertemplate="%{text}<br>Competition openness %{x:.0f}<br>Demand %{y:.0f}<extra></extra>"))
    hi = frame[frame.top]
    fig.add_trace(go.Scatter(x=hi.competition, y=hi.demand, mode="markers", name="Top ten at your weights", text=hi.market,
                             marker=dict(size=11, color="#CBFA7C", line=dict(width=1, color="#101B1B")),
                             hovertemplate="%{text}<br>Competition openness %{x:.0f}<br>Demand %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title="Competitive openness (higher = easier to enter)", range=[0, 100])
    fig.update_yaxes(title="Demand score", range=[0, 100])
    t.chart(fig, 380)
    t.insight("<b>Why this matters.</b> Strong demand in a closed market is expensive to win. The upper-right is where activity "
              "and room to compete overlap. Lime markers are the current top ten, so you can see whether the leaders sit there or "
              "were lifted by other dimensions.")

    st.markdown("##### Compare up to three markets")
    default = [k for k in a["order"][:3]]
    kept = [k for k in (st.session_state.get("ps_compare") or STORE.get("ps_compare") or default) if k in ok.index]
    st.session_state["ps_compare"] = kept or default
    picks = ctl(st.multiselect, "ps_compare", default, "Markets", list(ok.index), format_func=mname, max_selections=3)
    picks = [p for p in picks if p in ok.index] or default
    cmp_fig = go.Figure()
    for i, key in enumerate(picks):
        cmp_fig.add_trace(go.Bar(name=mname(key)[:34], x=[se.DIMENSIONS[k][0] for k in se.DIM_KEYS],
                                 y=[ok.loc[key, k] for k in se.DIM_KEYS],
                                 marker_color=["#CBFA7C", "#B9A6EC", "#96DFCB"][i]))
    cmp_fig.update_layout(barmode="group")
    cmp_fig.update_yaxes(title="Score (0-100, higher is better)", range=[0, 100])
    t.chart(cmp_fig, 340)
    rows = []
    for key in picks:
        r = by_key.loc[key]
        rows.append({
            "Market": r.market_name, "Population": f"{r.population_latest:,.0f}",
            "Purchase originations": f"{r.purchase_originations_latest:,.0f}",
            "Origination rate": f"{r.purchase_origination_rate_pct:.0f}%", "Denial rate": f"{r.purchase_denial_rate_pct:.0f}%",
            "Average loan": t.money(r.avg_loan_amount), "Active lenders": f"{r.active_lenders:,.0f}",
            "Top-5 lender share": f"{r.top5_lender_share_pct:.0f}%",
            "Home prices, 3-yr": "n/a" if pd.isna(r.hpi_3y_cagr_pct) else f"{r.hpi_3y_cagr_pct:+.1f}%/yr",
        })
    st.markdown("Observed figures behind those scores " + PILL["observed"], unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows).set_index("Market"), width="stretch")
    st.caption("Low observed lending is not unmet demand. Scores describe activity and conditions in public records, and "
               "cannot see applicants who never applied, local lender strategy, or eligibility rules.")

# ===========================================================================
# 3 · Decide
# ===========================================================================
elif step == STEPS[2]:
    st.markdown("##### What matters most to this lender?")
    with st.container(border=True):
        cols = st.columns(5)
        for col, key in zip(cols, se.DIM_KEYS):
            with col:
                ctl(st.slider, f"ps_w_{key}", DEFAULTS[f"ps_w_{key}"], se.DIMENSIONS[key][0], 0, 100, step=5, help=se.DIMENSIONS[key][1])
        total = sum(S(f"ps_w_{k}") for k in se.DIM_KEYS)
        norm = se.normalise({k: S(f"ps_w_{k}") for k in se.DIM_KEYS})
        st.caption("Weights are normalized to 100%, so only their proportions count. Effective weights: " +
                   " · ".join(f"{se.DIMENSIONS[k][0]} {norm[k] * 100:.0f}%" for k in se.DIM_KEYS) +
                   ("" if total else " (all zero, so the default mix is used)"))
    a = analysis()
    scope_line(a)
    if not a["order"]:
        st.warning("No markets pass the current screens. Relax them in step 1.")
        st.stop()
    order, comp, ok, stress = a["order"], a["comp"], a["ok"], a["stress"]
    contrib = se.contributions(ok, a["weights"])
    top = order[:10]

    st.markdown("##### Ranking at your weights " + PILL["modeled"], unsafe_allow_html=True)
    fig = go.Figure()
    for k in se.DIM_KEYS:
        fig.add_trace(go.Bar(name=se.DIMENSIONS[k][0], y=[mname(m)[:30] for m in top[::-1]], x=[contrib.loc[m, k] for m in top[::-1]],
                             orientation="h", marker_color=DIM_COLORS[k],
                             hovertemplate="%{y}<br>" + se.DIMENSIONS[k][0] + ": %{x:.1f} pts<extra></extra>"))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title="Weighted score (points contributed by each dimension)", range=[0, 100])
    t.chart(fig, 420)

    table = pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Market": [mname(m) for m in top],
        "Score": [comp[m] for m in top],
        "Top 3 in": [stress.top3[m] for m in top],
        "Top 10 in": [stress.top10[m] for m in top],
        "Median rank": [stress.median_rank[m] for m in top],
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Score": st.column_config.NumberColumn(format="%.1f"),
        "Top 3 in": st.column_config.ProgressColumn("In top 3", format="percent", min_value=0, max_value=1),
        "Top 10 in": st.column_config.ProgressColumn("In top 10", format="percent", min_value=0, max_value=1),
        "Median rank": st.column_config.NumberColumn("Median rank", format="%.0f"),
    })
    st.caption(f"The last three columns come from a weight stress test: the ranking is rebuilt under {stress.draws} nearby "
               "weight sets, and each market shows how often it holds that place. It tests sensitivity to your weights, "
               "not uncertainty in the public data. A high score is not proof of profitability or credit eligibility.")

    leader = order[0]
    runner = order[1] if len(order) > 1 else None
    st.markdown("##### Why did " + t.esc(mname(leader)) + " rank first?", unsafe_allow_html=True)
    mean_contrib = contrib.mean()
    edge = (contrib.loc[leader] - mean_contrib).sort_values(ascending=False)
    best, worst = edge.index[0], edge.index[-1]
    fp = se.flip_point(ok, a["weights"], leader, runner) if runner else None
    lines = [
        f"<li><b>Strongest factor</b> · {se.DIMENSIONS[best][0]} ({ok.loc[leader, best]:.0f}/100), adding "
        f"{edge[best]:+.1f} points over the typical ranked market.</li>",
        f"<li><b>Most consequential risk</b> · {se.DIMENSIONS[worst][0]} is its weakest area ({ok.loc[leader, worst]:.0f}/100, "
        f"{edge[worst]:+.1f} points against the field).</li>",
        f"<li><b>Main assumption</b> · that the weights you set are what the lender values. It holds the top spot in "
        f"{stress.top1[leader] * 100:.0f}% of nearby weight sets and a top-three place in {stress.top3[leader] * 100:.0f}%.</li>",
    ]
    if fp and runner:
        move = (fp["to"] - fp["from"]) * 100
        if abs(move) < 1:
            lines.append(f"<li><b>What would change it</b> · {t.esc(mname(runner))} is effectively tied. A one-point shift in "
                         f"the {se.DIMENSIONS[fp['dimension']][0]} weight would put it first.</li>")
        else:
            verb = "Raising" if move > 0 else "Lowering"
            lines.append(f"<li><b>What would change it</b> · {verb} the {se.DIMENSIONS[fp['dimension']][0]} weight from "
                         f"{fp['from'] * 100:.0f}% to {fp['to'] * 100:.0f}% (others scaled proportionally) puts "
                         f"{t.esc(mname(runner))} first.</li>")
    st.markdown(f'<div class="lab-card"><ul class="lab-checks">{"".join(lines)}</ul></div>', unsafe_allow_html=True)

    with st.expander("Screened out before weighting"):
        counts = a["gate"][a["gate"].reason.ne("") & a["gate"].reason.ne("Other market type")].reason.value_counts()
        if counts.empty:
            st.write("Nothing in this market type was removed by the screens.")
        else:
            st.dataframe(counts.rename("Markets").to_frame(), width="stretch")
    choices = order[:25]
    current = S("ps_market") if S("ps_market") in choices else order[0]
    STORE["ps_market"] = current
    if st.session_state.get("ps_market") not in choices:
        st.session_state.pop("ps_market", None)
    ctl(st.selectbox, "ps_market", current, "Market to take into economics and the brief", choices, format_func=mname)

# ===========================================================================
# 4 · Economics
# ===========================================================================
elif step == STEPS[3]:
    a = analysis()
    scope_line(a)
    if not a["order"]:
        st.warning("No markets pass the current screens. Relax them in step 1.")
        st.stop()
    product = a["product"]
    key = S("ps_market") if S("ps_market") in a["order"] else a["order"][0]
    STORE["ps_market"] = key
    row = by_key.loc[key]
    base_loans = se.market_volume(row, product)
    dflt = se.ECON_DEFAULTS[product]
    st.markdown(f"##### Illustrative economics for {t.esc(row.market_name)} " + PILL["illustrative"], unsafe_allow_html=True)
    t.insight("<b>Read this as a method, not a forecast.</b> Prices, costs and shares below are hypothetical assumptions a "
              "lender could replace with its own. The market size comes from observed lending. Nothing here is a rate quote or a "
              "claim about any real lender's economics.")
    with st.container(border=True):
        c = st.columns(4)
        capture = ctl(c[0].slider, f"ps_e_{product}_capture", dflt["capture_pct"], "Share of market captured (%)", 0.25, 10.0, step=0.25)
        rev = ctl(c[1].slider, f"ps_e_{product}_rev", dflt["revenue_bps"], "Revenue per loan (bps)", 100, 500, step=5,
                  help="Gross revenue as basis points of the loan amount, before credit reserve.")
        cost = ctl(c[2].slider, f"ps_e_{product}_cost", dflt["cost_per_loan"], "Cost to originate ($)", 2_000, 12_000, step=250, format="$%d")
        res = ctl(c[3].slider, f"ps_e_{product}_res", dflt["reserve_bps"], "Credit and repurchase reserve (bps)", 0, 50, step=1,
                  help="Set aside per loan. Kept separate from revenue so it is not double counted.")
        d = st.columns(4)
        launch = ctl(d[0].slider, f"ps_e_{product}_launch", dflt["launch_cost"], "Launch cost ($)", 50_000, 2_000_000, step=25_000, format="$%d")
        horizon = ctl(d[1].slider, "ps_horizon", DEFAULTS["ps_horizon"], "Horizon (years)", 1, 5)
        avg_loan = ctl(d[2].number_input, f"ps_e_avg_{key}", float(round(row.avg_loan_amount, -3)), "Average loan ($)", min_value=50_000.0,
                       max_value=3_000_000.0, step=5_000.0, format="%.0f")
        scenario = d[3].radio("Scenario", list(se.SCENARIOS), index=list(se.SCENARIOS).index(S("ps_scenario")), horizontal=True,
                              key="ps_scenario_w")
        STORE["ps_scenario"] = scenario
    inputs = dict(market_loans=base_loans, avg_loan=avg_loan, capture_pct=capture, revenue_bps=rev, cost_per_loan=cost,
                  reserve_bps=res, launch_cost=launch, horizon=horizon)
    result = se.economics(**inputs, scenario=scenario)
    t.tiles([
        {"label": "Market loans per year", "value": f"{base_loans:,.0f}", "note": "Estimated from observed lending", "accent": False},
        {"label": "Funded loans at full run rate", "value": f"{result['loans_full']:,.0f}", "note": f"{scenario} case"},
        {"label": "Contribution per loan", "value": ("−" if result["margin"] < 0 else "") + f"${abs(result['margin']):,.0f}", "note": "After cost and reserve"},
        {"label": f"Net after {horizon} yr", "value": t.money(result["net"]), "note": "After launch cost", "accent": True},
        {"label": "Break-even share", "value": "n/a" if not np.isfinite(result["breakeven_share"]) else f"{result['breakeven_share']:.1f}%",
         "note": "Of the market, over the horizon"},
    ])
    years = ["Launch"] + [f"Year {i}" for i in range(1, horizon + 1)]
    fig = go.Figure()
    for name, color in [("Downside", "#EF936F"), ("Base", "#CBFA7C"), ("Upside", "#96DFCB")]:
        r = se.economics(**inputs, scenario=name)
        fig.add_trace(go.Scatter(x=years, y=r["cumulative"], mode="lines+markers", name=name, line=dict(color=color, width=3 if name == scenario else 1.5),
                                 opacity=1 if name == scenario else 0.55))
    fig.add_hline(y=0, line_color="rgba(166,184,174,.4)")
    fig.update_yaxes(title="Cumulative net ($)", tickformat="$,.0f")
    t.chart(fig, 340)
    tor = se.tornado(inputs)
    tf = go.Figure()
    tf.add_trace(go.Bar(y=tor.input, x=tor.down, orientation="h", name="Input 20% lower", marker_color="#EF936F"))
    tf.add_trace(go.Bar(y=tor.input, x=tor.up, orientation="h", name="Input 20% higher", marker_color="#CBFA7C"))
    tf.update_layout(barmode="relative", title="What moves the result most (change in net, base case)")
    tf.update_xaxes(title="Change in net ($)", tickformat="$,.0f")
    t.chart(tf, 320)
    st.caption("Funded loans = market loans × share captured × scenario volume, ramped 50% in year one. Contribution per loan = "
               "loan × (revenue − reserve) − cost to originate. Net = Σ yearly contribution − launch cost.")

# ===========================================================================
# 5 · Brief
# ===========================================================================
else:
    a = analysis()
    scope_line(a)
    if not a["order"]:
        st.warning("No markets pass the current screens. Relax them in step 1.")
        st.stop()
    product = a["product"]
    order, comp, ok, stress = a["order"], a["comp"], a["ok"], a["stress"]
    key = S("ps_market") if S("ps_market") in order else order[0]
    row = by_key.loc[key]
    rank = order.index(key) + 1
    dflt = se.ECON_DEFAULTS[product]
    inputs = dict(
        market_loans=se.market_volume(row, product), avg_loan=float(STORE.get(f"ps_e_avg_{key}", round(row.avg_loan_amount, -3))),
        capture_pct=S(f"ps_e_{product}_capture") or dflt["capture_pct"], revenue_bps=S(f"ps_e_{product}_rev") or dflt["revenue_bps"],
        cost_per_loan=S(f"ps_e_{product}_cost") or dflt["cost_per_loan"], reserve_bps=S(f"ps_e_{product}_res") if S(f"ps_e_{product}_res") is not None else dflt["reserve_bps"],
        launch_cost=S(f"ps_e_{product}_launch") or dflt["launch_cost"], horizon=S("ps_horizon"))
    base = se.economics(**inputs, scenario="Base")
    down = se.economics(**inputs, scenario="Downside")
    cls, why = se.classify(readiness=row.data_readiness, rank=rank, top3=float(stress.top3[key]), top10=float(stress.top10[key]),
                           base_net=base["net"], downside_net=down["net"], breakeven_share=base["breakeven_share"],
                           risk_pct=float(row.mortgage_risk_score) if pd.notna(row.mortgage_risk_score) else 100.0)
    contrib = se.contributions(ok, a["weights"])
    strengths = contrib.loc[key].sub(contrib.mean()).sort_values(ascending=False)
    norm = se.normalise(a["weights"])
    alts = [k for k in order[:4] if k != key][:3]

    def alt_line(k):
        gap = comp[k] - comp[key]
        better = se.DIMENSIONS[(ok.loc[k] - ok.loc[key]).idxmax()][0]
        gap_text = 'effectively tied' if abs(gap) < 0.05 else f"{'ahead' if gap > 0 else 'behind'} by {abs(gap):.1f} points"
        return f"{mname(k)} (score {comp[k]:.1f}, rank {order.index(k) + 1}): {gap_text}, stronger on {better.lower()}."

    meta_years = meta.get("hmda_years") or []
    vintage = (f"HMDA {meta_years[0]}–{meta_years[-1]}, FHFA {meta.get('fhfa_latest_year', 'latest')}, Census {meta.get('census_vintage', 'latest')}"
               if meta_years else "synthetic sample data")
    roadmap = {
        "Prioritize for diligence": ["Validate the public indicators against the lender's own pipeline and local market knowledge.",
                                     "Check product guidelines, eligibility and capacity requirements with credit and compliance.",
                                     "Build a bottom-up volume and cost estimate to replace the illustrative economics.",
                                     "Design a limited pilot with exit criteria, then decide.", "Monitor conversion, credit quality and unit cost monthly."],
        "Pilot candidate": ["Confirm the market holds up under alternative weights and a refreshed data vintage.",
                            "Scope a small pilot with explicit volume and credit-quality limits.", "Agree success and stop criteria before launch.",
                            "Review results after two quarters."],
        "Optimize current offering": ["Compare current performance in this market against the public benchmarks shown.",
                                      "Look for conversion and cost improvements before any new investment.", "Re-test if the market's rank changes."],
        "Monitor": ["Track the same indicators each data refresh.", "Revisit if risk eases, the economics improve, or the weights change."],
        "Defer pending evidence": ["Wait for fuller public data or add a better source.", "Do not allocate investment on this evidence."],
    }[cls]
    kpis = ["Application-to-funding conversion", "Credit quality and early delinquency", "Unit cost to originate", "Share of local market", "Time to decision for borrowers"]

    def li(items):
        return "".join(f"<li>{_html.escape(i)}</li>" for i in items)

    strongest = ", ".join(se.DIMENSIONS[k][0].lower() for k in strengths.index[:2])
    weakest = se.DIMENSIONS[strengths.index[-1]][0].lower()
    body = f"""
<h1>Market expansion brief</h1>
<p class="tag">Research demonstration · hypothetical lender · public data · not a real decision or an approval</p>
<h2>Strategic question</h2>
<p>Where should a lender expand its {_html.escape(se.PRODUCTS[product].lower())} activity among {_html.escape(S('ps_area').lower())} areas, given the strategy weights below?</p>
<h2>Recommended direction</h2>
<p class="big">{_html.escape(row.market_name)}: {_html.escape(cls)}</p>
<p>{_html.escape(why)} It ranks {rank} of {len(order)} with a weighted score of {comp[key]:.1f}, and holds a top-three place in {stress.top3[key] * 100:.0f}% of nearby weight sets.</p>
<h2>Supporting evidence <small>(observed and modeled)</small></h2>
<ul><li>Strongest dimensions: {strongest}; weakest: {weakest}.</li>
<li>Observed: {row.purchase_originations_latest:,.0f} purchase originations, {row.purchase_origination_rate_pct:.0f}% origination rate, {row.purchase_denial_rate_pct:.0f}% denial rate, {row.active_lenders:,.0f} active lenders, top five hold {row.top5_lender_share_pct:.0f}%.</li>
<li>Strategy weights: {", ".join(f"{se.DIMENSIONS[k][0]} {norm[k] * 100:.0f}%" for k in se.DIM_KEYS)}.</li></ul>
<h2>Business case <small>(illustrative)</small></h2>
<ul><li>At {inputs['capture_pct']:.2f}% of an estimated {inputs['market_loans']:,.0f} market loans a year: about {base['loans_full']:,.0f} funded loans at full run rate.</li>
<li>Contribution per loan about ${base['margin']:,.0f}; net after {inputs['horizon']} years {t.money(base['net'])} base, {t.money(down['net'])} downside.</li>
<li>Break-even needs {"n/a" if not np.isfinite(base['breakeven_share']) else f"{base['breakeven_share']:.1f}%"} share. All pricing, cost and launch figures are hypothetical.</li></ul>
<h2>Risks and constraints</h2>
<ul><li>Credit-risk percentile {row.mortgage_risk_score:.0f} of 100 among comparable markets (lower is safer; {_html.escape(str(row.mortgage_risk_basis))} basis); screens applied: risk limit {S('ps_maxrisk')}, evidence {_html.escape(S('ps_ready'))}, population floor {S('ps_minpop'):,}.</li>
<li>Public aggregates cannot see eligibility rules, local lender strategy or applicants who never applied.</li>
<li>Rankings reflect the weights chosen, not proof of profitability or credit eligibility.</li></ul>
<h2>Alternatives considered</h2>
<ul>{li([alt_line(k) for k in alts]) or "<li>No other markets ranked.</li>"}</ul>
<h2>Execution roadmap</h2>
<ol>{li(roadmap)}</ol>
<h2>Success metrics</h2>
<ul>{li(kpis)}</ul>
<p class="foot">Data vintage: {_html.escape(vintage)}. Built from public Census, HMDA and FHFA data by Richard Henderson as a portfolio project.</p>
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Market expansion brief</title>
<style>body{{font:15px/1.6 Georgia,serif;max-width:760px;margin:40px auto;padding:0 20px;color:#142322}}h1{{font:700 30px Helvetica,Arial,sans-serif;margin:0 0 4px}}
h2{{font:700 13px Helvetica,Arial,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#4d6a5c;margin:26px 0 6px;border-top:1px solid #ccd6d0;padding-top:14px}}
.tag{{font:12px Helvetica,Arial,sans-serif;color:#8a4b30}}.big{{font:700 22px Helvetica,Arial,sans-serif}}small{{font-weight:400;text-transform:none;letter-spacing:0}}.foot{{font-size:12px;color:#667}}</style></head><body>{body}</body></html>"""

    st.markdown(f'<div class="lab-reco"><div class="e">Recommended direction · {t.esc(cls)}</div>'
                f'<div class="t">{t.esc(row.market_name)}</div><div class="d">{t.esc(why)}</div></div>', unsafe_allow_html=True)
    st.write("")
    t.tiles([
        {"label": "Rank at your weights", "value": f"{rank} of {len(order)}", "note": f"Score {comp[key]:.1f}"},
        {"label": "Top 3 across weight sets", "value": f"{stress.top3[key] * 100:.0f}%", "note": "Stability of the result"},
        {"label": f"Net after {inputs['horizon']} yr (base)", "value": t.money(base["net"]), "note": "Illustrative", "accent": True},
        {"label": "Break-even share", "value": "n/a" if not np.isfinite(base["breakeven_share"]) else f"{base['breakeven_share']:.1f}%",
         "note": "Of the market"},
    ])
    with st.container(border=True):
        st.markdown("<style>.ps-brief h1{font-size:1.9rem!important;margin:0 0 .2rem}.ps-brief h2{font-size:.74rem!important;letter-spacing:.16em;text-transform:uppercase;color:#CBFA7C;margin:1.4rem 0 .4rem;border-top:1px solid #263E36;padding-top:.9rem}.ps-brief h2 small{text-transform:none;letter-spacing:0;color:#A6B8AE;font-size:.8rem}.ps-brief .big{font-size:1.35rem;font-weight:700}.ps-brief .tag{color:#F2A887;font-size:.85rem}.ps-brief .foot{color:#A6B8AE;font-size:.82rem}</style>" + f'<div class="ps-brief">{body}</div>', unsafe_allow_html=True)
    st.download_button("Download the brief (HTML, print to PDF)", page, file_name="market-expansion-brief.html", mime="text/html",
                       icon=":material/download:")
    with st.expander("How the recommendation class is decided"):
        for name, rule in se.CLASSES:
            st.markdown(f"**{name}.** {rule}")
        st.caption("Rules are applied in this order: evidence first, then economics and risk, then rank and stability. "
                   "The class is a research label. It implies no approval, authorization or regulatory clearance.")

with st.expander("Methodology, data and limits"):
    st.markdown(f"""
**Data.** {'Public, aggregated Census population estimates, HMDA lending records and the FHFA House Price Index, rebuilt monthly. Vintage: ' + str(meta.get('hmda_years', '')) if live else 'A synthetic sample table.'} Market definitions are Census CBSAs. No borrower-level records, lender names or private data are used.

**Scores.** Market scores use the same percentile-within-area-type method as the Mortgage market intelligence project. The five dimensions are rebuilt per product line from those blocks and from observed HMDA shares. Refinance and government-backed pools are scaled from observed purchase originations, so they are estimates.

**Decision method.** Hard screens first, then a weighted composite over the five dimensions with weights normalized to 100%, then a stress test that re-ranks under nearby weight sets. This is a multi-criteria weighting model. It does not claim optimality.

**Economics.** A small, transparent illustration with hypothetical price, cost, reserve and launch inputs. Revenue and credit reserve are separate lines so neither is counted twice.

**Limits.** Public aggregates do not show eligibility, pricing, secondary-market terms or applicant intent. Low lending is not unmet demand. Scores are relative within metro or micro areas, not absolute. This is a portfolio demonstration, not advice, and is not a model of any real lender.
""")
t.footnote("Research demonstration on public data and hypothetical assumptions.")
