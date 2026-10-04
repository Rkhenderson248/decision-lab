import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import ask
from lab import market_engine as me
from lab import theme as t
from lab import widgets as w

STATE_GRID = {
    "AK": ("Alaska", 0, 0), "ME": ("Maine", 10, 0), "VT": ("Vermont", 9, 1), "NH": ("New Hampshire", 10, 1),
    "WA": ("Washington", 0, 2), "ID": ("Idaho", 1, 2), "MT": ("Montana", 2, 2), "ND": ("North Dakota", 3, 2),
    "MN": ("Minnesota", 4, 2), "IL": ("Illinois", 5, 2), "WI": ("Wisconsin", 6, 2), "MI": ("Michigan", 7, 2),
    "NY": ("New York", 8, 2), "RI": ("Rhode Island", 9, 2), "MA": ("Massachusetts", 10, 2),
    "OR": ("Oregon", 0, 3), "NV": ("Nevada", 1, 3), "WY": ("Wyoming", 2, 3), "SD": ("South Dakota", 3, 3),
    "IA": ("Iowa", 4, 3), "IN": ("Indiana", 5, 3), "OH": ("Ohio", 6, 3), "PA": ("Pennsylvania", 7, 3),
    "NJ": ("New Jersey", 8, 3), "CT": ("Connecticut", 9, 3), "CA": ("California", 0, 4), "UT": ("Utah", 1, 4),
    "CO": ("Colorado", 2, 4), "NE": ("Nebraska", 3, 4), "MO": ("Missouri", 4, 4), "KY": ("Kentucky", 5, 4),
    "WV": ("West Virginia", 6, 4), "VA": ("Virginia", 7, 4), "MD": ("Maryland", 8, 4), "DC": ("Washington DC", 9, 4),
    "AZ": ("Arizona", 1, 5), "NM": ("New Mexico", 2, 5), "KS": ("Kansas", 3, 5), "AR": ("Arkansas", 4, 5),
    "TN": ("Tennessee", 5, 5), "NC": ("North Carolina", 6, 5), "SC": ("South Carolina", 7, 5), "DE": ("Delaware", 8, 5),
    "OK": ("Oklahoma", 3, 6), "LA": ("Louisiana", 4, 6), "MS": ("Mississippi", 5, 6), "AL": ("Alabama", 6, 6),
    "GA": ("Georgia", 7, 6), "HI": ("Hawaii", 0, 7), "TX": ("Texas", 3, 7), "FL": ("Florida", 8, 7),
}
SEQ = ["#E6F2EF", "#C3E2D9", "#9DD0C2", "#73BBA9", "#49A48F", "#1F8C76", "#0F7563", "#0F5E51", "#0F4640"]
BLOCK_SHORT = {
    "demographic_opportunity_score": "Demographic",
    "mortgage_demand_score": "Demand",
    "borrower_capacity_score": "Capacity",
    "collateral_momentum_score": "Collateral",
    "market_openness_score": "Openness",
    "affordability_pressure_score": "Affordability pressure",
}
BLOCK_TICK = {**BLOCK_SHORT, "demographic_opportunity_score": "Demo-<br>graphic",
              "affordability_pressure_score": "Afford.<br>pressure"}


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
def _clusters(prepared: pd.DataFrame):
    return me.cluster(prepared)


@st.cache_data(show_spinner=False)
def _extras():
    return me.load_years(), me.load_lenders(), me.load_history()


prepared, live, meta = _base()

t.header(
    "Data product · Market intelligence",
    "A market-sizing product that refreshes itself",
    "A scheduled pipeline pulls public Census, HMDA and FHFA data every month, checks it, scores every U.S. "
    "metro and micro area on demand, capacity, momentum and competitive openness, and records what changed. "
    "The housing-finance market is the worked example; the pattern applies to any market with public signals. "
    "Set your own strategy weights and the rankings, archetypes and briefs rebuild on the spot.",
)

if not live:
    st.markdown(
        '<div class="lab-insight"><b>Sample data.</b> This preview runs on a synthetic market table with '
        "fictional market names so the method can be explored. The published version uses the public "
        "Census, HMDA and FHFA build.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 0.9])
    with c1:
        area = w.segmented("Market type", ["Metropolitan", "Micropolitan"], key="mi_area", default="Metropolitan")
    with c2:
        evidence = w.segmented("Evidence required", ["Complete", "Strong+", "Any"], key="mi_evidence", default="Strong+",
                               help="Complete: all five blocks observed. Strong+: four or more.")
    states = c3.multiselect("States", sorted(STATE_GRID), placeholder="All states", key="mi_states")
    with c4.popover("Strategy weights", width="stretch", icon=":material/tune:"):
        st.caption("How much each block counts toward the opportunity score. Weights are normalized.")
        weights = {
            "demographic": st.slider("Demographic strength", 0, 50, 25, 5, key="w_demo"),
            "demand": st.slider("Mortgage demand", 0, 50, 25, 5, key="w_demand"),
            "capacity": st.slider("Borrower capacity", 0, 50, 20, 5, key="w_cap"),
            "collateral": st.slider("Collateral momentum", 0, 50, 20, 5, key="w_coll"),
            "openness": st.slider("Competitive openness", 0, 50, 10, 5, key="w_open"),
        }

if sum(weights.values()) == 0:
    st.warning("All weights are zero. Using the default strategy.")
    weights = {k: int(v * 100) for k, v in me.DEFAULT_WEIGHTS.items()}

scored = me.score(prepared, {k: v / 100 for k, v in weights.items()})
clusters = _clusters(prepared)
scored = scored.merge(clusters.assignments, on="market_key", how="left")

allowed = {"Complete": ["Complete"], "Strong+": ["Complete", "Strong"], "Any": ["Complete", "Strong", "Partial"]}[evidence]
view = scored[scored["area_type"].eq(area) & scored["data_readiness"].isin(allowed)
              & scored["strategic_mortgage_opportunity_score"].notna()].copy()
if states:
    view = view[view["primary_state"].isin(states)]

if view.empty:
    st.info("No markets match these filters. Widen the evidence level or clear the state filter.")
    st.stop()

custom = any(weights[k] / 100 != v for k, v in me.DEFAULT_WEIGHTS.items())
top = view.sort_values(["strategic_mortgage_opportunity_score", "population_latest"], ascending=False).iloc[0]
hmda_span = (f"{int(view['hmda_first_year'].min())}–{int(view['hmda_latest_year'].max())}"
             if view["hmda_latest_year"].notna().any() else "n/a")
t.tiles([
    {"label": f"{area} markets in view", "value": f"{len(view):,}",
     "note": f"{view['population_latest'].sum() / 1e6:,.1f}M residents"},
    {"label": "Highest opportunity", "value": f"{top['strategic_mortgage_opportunity_score']:.0f}",
     "accent": True, "note": str(top["market_name"])},
    {"label": "High-signal markets", "value": f"{(view['mortgage_signal'] == 'High').sum():,}",
     "note": "top third within market type"},
    {"label": "Data vintage", "value": hmda_span,
     "note": "HMDA years" + (f" · FHFA {int(view['fhfa_latest_year'].max())}" if view['fhfa_latest_year'].notna().any() else "")},
])
if custom:
    st.caption("Scores reflect your custom strategy weights.")

TAB_NAMES = ["Overview", "Rankings", "Market brief", "Lenders", "What changed", "Archetypes", "Signals", "What could be built"]
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
years_df, lenders_df, history = _extras()

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with T["Overview"]:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        by_state = (view.assign(w=view["population_latest"].fillna(0))
                    .groupby("primary_state")
                    .apply(lambda d: pd.Series({
                        "score": np.average(d["strategic_mortgage_opportunity_score"], weights=d["w"]) if d["w"].sum() > 0 else d["strategic_mortgage_opportunity_score"].mean(),
                        "markets": len(d), "pop": d["population_latest"].sum(),
                        "lead": d.sort_values("strategic_mortgage_opportunity_score", ascending=False)["market_name"].iloc[0],
                    }), include_groups=False))
        lo = float(np.floor(by_state["score"].quantile(0.05)))
        hi = float(np.ceil(by_state["score"].quantile(0.95)))
        if hi - lo < 5:
            lo, hi = lo - 5, hi + 5
        fig = go.Figure()
        xs, ys, texts, hovers, colors = [], [], [], [], []
        for st_code, (name, gx, gy) in STATE_GRID.items():
            row = by_state.loc[st_code] if st_code in by_state.index else None
            if row is not None:
                frac = float(np.clip((row["score"] - lo) / (hi - lo), 0, 0.999))
                fill = SEQ[int(frac * len(SEQ))]
                hovers.append(f"<b>{name}</b><br>Population-weighted opportunity {row['score']:.0f}"
                              f"<br>{int(row['markets'])} markets · lead: {row['lead']}")
                txt_color = "#FFFFFF" if frac > 0.55 else t.INK
            else:
                fill = "#F1F2EE"
                hovers.append(f"<b>{name}</b><br>No markets in view")
                txt_color = t.MUTED
            fig.add_shape(type="rect", x0=gx + 0.04, x1=gx + 0.96, y0=-gy - 0.96, y1=-gy - 0.04,
                          line=dict(color="#FFFFFF", width=2), fillcolor=fill, layer="below")
            xs.append(gx + 0.5); ys.append(-gy - 0.5); texts.append(f"<span style='color:{txt_color}'>{st_code}</span>")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=texts, textfont=dict(size=12),
                                 hovertext=hovers, hoverinfo="text", showlegend=False))
        # Color key
        for i, colr in enumerate(SEQ):
            fig.add_shape(type="rect", x0=0.2 + i * 0.42, x1=0.6 + i * 0.42, y0=-8.55, y1=-8.35,
                          fillcolor=colr, line=dict(width=0))
        fig.add_annotation(x=0.2, y=-8.75, text=f"{lo:.0f}", showarrow=False, font=dict(size=11, color=t.MUTED), xanchor="left")
        fig.add_annotation(x=0.2 + len(SEQ) * 0.42, y=-8.75, text=f"{hi:.0f}", showarrow=False,
                           font=dict(size=11, color=t.MUTED), xanchor="right")
        fig.update_layout(
            title="Population-weighted opportunity by state",
            xaxis=dict(visible=False, range=[-0.1, 11.1]), yaxis=dict(visible=False, range=[-9, 0.1], scaleanchor="x"),
            margin=dict(l=4, r=4, t=60, b=4),
        )
        t.chart(fig, height=430)

    with right:
        pts = view.dropna(subset=["mortgage_risk_score"]).copy()
        leaders = pts[pts["strategic_mortgage_opportunity_score"].ge(pts["strategic_mortgage_opportunity_score"].quantile(.9))
                      & pts["mortgage_risk_score"].le(50)].nlargest(5, "strategic_mortgage_opportunity_score")
        size = np.clip(np.sqrt(pts["population_latest"].fillna(0)) / (60 if area == "Metropolitan" else 22), 5, 28)
        fig2 = go.Figure()
        for x0, x1, y0, y1, label, ax, ay in ((50, 100, 0, 50, "Pursue", 98, 3), (50, 100, 50, 100, "Pursue with care", 98, 97),
                                               (0, 50, 0, 50, "Monitor", 2, 3), (0, 50, 50, 100, "Deprioritize", 2, 97)):
            fig2.add_annotation(x=ax, y=ay, text=label, showarrow=False, font=dict(size=12, color=t.MUTED),
                                xanchor="right" if ax > 50 else "left", yanchor="bottom" if ay < 50 else "top")
        fig2.add_shape(type="rect", x0=50, x1=100, y0=0, y1=50, fillcolor="rgba(127,208,190,.12)", line_width=0, layer="below")
        fig2.add_hline(y=50, line=dict(color=t.GRID, width=1))
        fig2.add_vline(x=50, line=dict(color=t.GRID, width=1))
        fig2.add_trace(go.Scatter(
            x=pts["strategic_mortgage_opportunity_score"], y=pts["mortgage_risk_score"], mode="markers",
            marker=dict(size=size, color=t.S1, opacity=0.55, line=dict(color="#FFFFFF", width=1)),
            customdata=np.stack([pts["market_name"], pts["population_latest"], pts["mortgage_archetype"],
                                 pts["mortgage_risk_basis"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>Opportunity %{x:.0f} · Risk %{y:.0f} (%{customdata[3]})"
                          "<br>%{customdata[1]:,.0f} residents<br>%{customdata[2]}<extra></extra>",
            showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=leaders["strategic_mortgage_opportunity_score"], y=leaders["mortgage_risk_score"], mode="markers",
            marker=dict(size=11, color=t.PETROL, line=dict(color="#FFFFFF", width=2)),
            customdata=leaders["market_name"], hovertemplate="<b>%{customdata}</b><br>Top pursue candidate<extra></extra>",
            showlegend=False,
        ))
        fig2.update_layout(
            title="Opportunity against risk · each dot a market, sized by population",
            xaxis=dict(title="Strategic opportunity (within-type percentile basis)", range=[0, 100]),
            yaxis=dict(title="Mortgage risk percentile", range=[0, 100]),
        )
        t.chart(fig2, height=430)

    pursue = pts[pts["strategic_mortgage_opportunity_score"].ge(50) & pts["mortgage_risk_score"].lt(50)]
    lead_txt = ", ".join(n.split(",")[0] for n in leaders["market_name"].head(3)) or "none at these settings"
    t.insight(
        f"<b>{len(pursue):,}</b> of {len(pts):,} {area.lower()} markets sit in the <b>Pursue</b> quadrant: "
        f"above-median opportunity with below-median risk. The strongest of them: {t.esc(lead_txt)}."
    )

# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
with T["Rankings"]:
    cols = ["market_name", "strategic_mortgage_opportunity_score", "demographic_opportunity_score",
            "mortgage_demand_score", "borrower_capacity_score", "collateral_momentum_score",
            "market_openness_score", "mortgage_risk_score", "mortgage_archetype", "population_latest"]
    names = {"market_name": "Market", "strategic_mortgage_opportunity_score": "Opportunity",
             "demographic_opportunity_score": "Demographic", "mortgage_demand_score": "Demand",
             "borrower_capacity_score": "Capacity", "collateral_momentum_score": "Collateral",
             "market_openness_score": "Openness", "mortgage_risk_score": "Risk",
             "mortgage_archetype": "Archetype", "population_latest": "Population"}
    bar = lambda: st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)
    cfg = {n: bar() for n in ["Opportunity", "Demographic", "Demand", "Capacity", "Collateral", "Openness", "Risk"]}
    cfg["Population"] = st.column_config.NumberColumn(format="localized")

    st.subheader("Top opportunities", anchor=False)
    ranked = view.sort_values(["strategic_mortgage_opportunity_score", "population_latest"], ascending=False)
    st.dataframe(ranked[cols].head(25).rename(columns=names), hide_index=True, column_config=cfg, height=420)

    st.subheader("Risk watchlist", anchor=False)
    st.caption("Highest mortgage risk, ranked within evidence basis. Full-basis and HMDA-only readings are never mixed.")
    watch = view.dropna(subset=["mortgage_risk_score"]).sort_values(
        ["mortgage_risk_basis", "mortgage_risk_score"], ascending=[True, False])
    wcols = ["market_name", "mortgage_risk_score", "mortgage_risk_basis", "affordability_pressure_score",
             "purchase_denial_rate_pct", "high_dti_share_pct", "hpi_1y_pct", "population_latest"]
    wcols = [c for c in wcols if c == "market_name" or watch[c].notna().any()]
    st.dataframe(
        watch[wcols].head(15).rename(columns={
            "market_name": "Market", "mortgage_risk_score": "Risk", "mortgage_risk_basis": "Basis",
            "affordability_pressure_score": "Affordability pressure", "purchase_denial_rate_pct": "Denial rate %",
            "high_dti_share_pct": "High-DTI share %", "hpi_1y_pct": "HPI 1-yr %", "population_latest": "Population"}),
        hide_index=True,
        column_config={"Risk": bar(), "Affordability pressure": bar(),
                       "Denial rate %": st.column_config.NumberColumn(format="%.1f"),
                       "High-DTI share %": st.column_config.NumberColumn(format="%.1f"),
                       "HPI 1-yr %": st.column_config.NumberColumn(format="%.1f"),
                       "Population": st.column_config.NumberColumn(format="localized")},
    )
    export = ranked[["market_key", "market_name", "primary_state", "area_type", "data_readiness",
                     "strategic_mortgage_opportunity_score", *me.BLOCKS.values(), "mortgage_risk_score",
                     "mortgage_risk_basis", "mortgage_archetype", "cluster_label", "population_latest"]]
    st.download_button("Download this ranking (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"market_ranking_{area.lower()}.csv", mime="text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------
with T["Archetypes"]:
    prof = clusters.profiles[clusters.profiles["area_type"].eq(area)] if not clusters.profiles.empty else pd.DataFrame()
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        if prof.empty:
            st.info("Not enough complete markets to fit archetypes for this market type.")
        else:
            z = prof[me.CLUSTER_FEATURES].to_numpy()
            ylab = [f"{r.cluster_id} · {r.markets} markets" for r in prof.itertuples()]
            figc = go.Figure(go.Heatmap(
                z=z, x=[BLOCK_TICK[c] for c in me.CLUSTER_FEATURES], y=ylab, zmid=0, zmin=-1.2, zmax=1.2,
                colorscale=[[0, t.S2], [0.5, "#F0EFEC"], [1, t.S1]], xgap=2, ygap=2,
                text=[[f"{v:+.1f}" for v in r] for r in z], texttemplate="%{text}", textfont=dict(size=12),
                colorbar=dict(title=dict(text="vs typical", font=dict(size=11, color=t.MUTED)), thickness=10,
                              outlinewidth=0, tickfont=dict(size=11, color=t.MUTED)),
                hovertemplate="%{y}<br>%{x}: %{z:+.2f} (robust z)<extra></extra>",
            ))
            figc.update_layout(title="Archetype profiles discovered by clustering (robust z-scores)",
                               yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                               xaxis=dict(showline=False, ticks="", tickangle=0, side="top", tickfont=dict(size=11)))
            t.chart(figc, height=110 + 62 * len(prof))
            sel = clusters.selection[clusters.selection["area_type"].eq(area)]
            if not sel.empty:
                best = sel[sel["viable"]].sort_values("silhouette", ascending=False).head(1)
                if not best.empty:
                    st.caption(f"k = {int(best['k'].iloc[0])} chosen by silhouette ({best['silhouette'].iloc[0]:.2f}) "
                               f"from k = 3–6, after robust scaling and PCA to 85% variance. Low silhouettes mean "
                               "soft tendencies, not hard segments, which is why archetypes describe markets but "
                               "should never define territories.")
            for r in prof.itertuples():
                st.markdown(f"**{r.cluster_id}** · {r.cluster_label}")

    with right:
        counts = view["mortgage_archetype"].value_counts().sort_values()
        figr = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h", marker=dict(color=t.S1, cornerradius=4),
            text=counts.values, textposition="outside", textfont=dict(size=12, color=t.GRAPHITE),
            hovertemplate="%{y}: %{x} markets<extra></extra>", width=0.6))
        figr.update_layout(title=dict(text="Rule-based archetypes", x=0, xref="container", xanchor="left"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                           xaxis=dict(range=[0, counts.max() * 1.25], visible=False), bargap=0.3,
                           margin=dict(l=8, r=8, t=78, b=8))
        t.chart(figr, height=110 + 44 * len(counts))
        st.caption("Rules use within-type thresholds: for example, compound growth requires demographic, demand "
                   "and collateral blocks all in their top 30%.")

# ---------------------------------------------------------------------------
# Market brief
# ---------------------------------------------------------------------------
with T["Market brief"]:
    options = ranked["market_name"].tolist()
    choice = st.selectbox("Choose a market", options, index=0, key="mi_market")
    row = view[view["market_name"].eq(choice)].iloc[0]
    peers = view
    pct_opp = (peers["strategic_mortgage_opportunity_score"] <= row["strategic_mortgage_opportunity_score"]).mean() * 100
    rank_opp = int((peers["strategic_mortgage_opportunity_score"] > row["strategic_mortgage_opportunity_score"]).sum() + 1)

    pills = [f'<span class="lab-pill">{t.esc(row["mortgage_signal"])} signal</span>',
             f'<span class="lab-pill off">{t.esc(row["mortgage_archetype"])}</span>']
    if isinstance(row.get("cluster_label"), str):
        pills.append(f'<span class="lab-pill off">{t.esc(row["cluster_id"])} · {t.esc(row["cluster_membership"])}</span>')
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    t.tiles([
        {"label": "Strategic opportunity", "value": f"{row['strategic_mortgage_opportunity_score']:.0f}", "accent": True,
         "note": f"rank {rank_opp:,} of {len(peers):,} in view"},
        {"label": "Mortgage risk", "value": f"{row['mortgage_risk_score']:.0f}" if pd.notna(row["mortgage_risk_score"]) else "n/a",
         "note": f"{row['mortgage_risk_basis']} basis"},
        {"label": "Population", "value": f"{row['population_latest']:,.0f}",
         "note": f"{row['population_cagr_pct']:+.2f}% a year" if pd.notna(row["population_cagr_pct"]) else ""},
        {"label": "Purchase originations", "value": f"{row['purchase_originations_latest']:,.0f}" if pd.notna(row["purchase_originations_latest"]) else "n/a",
         "note": f"{row['purchase_origination_cagr_pct']:+.1f}% a year" if pd.notna(row["purchase_origination_cagr_pct"]) else ""},
    ])

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        feats = list(me.BLOCKS.values())
        theta = [BLOCK_SHORT[c] for c in feats] + [BLOCK_SHORT[feats[0]]]
        med = [float(peers[c].median()) for c in feats]
        val = [float(row[c]) if pd.notna(row[c]) else None for c in feats]
        figp = go.Figure()
        figp.add_trace(go.Scatterpolar(r=med + med[:1], theta=theta, name="Peer median", line=dict(color=t.BASE, width=1.5),
                                       hovertemplate="%{theta}: %{r:.0f}<extra>Peer median</extra>"))
        figp.add_trace(go.Scatterpolar(r=val + val[:1], theta=theta, name=choice.split(",")[0], fill="toself",
                                       fillcolor="rgba(0,138,115,.16)", line=dict(color=t.S1, width=2.5),
                                       hovertemplate="%{theta}: %{r:.0f}<extra></extra>"))
        figp.update_layout(title="Market fingerprint against its peers", margin=dict(l=70, r=70, t=90, b=40),
                           polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor=t.GRID_2, linecolor=t.GRID,
                                                                                tickfont=dict(size=10, color=t.MUTED)),
                                      angularaxis=dict(gridcolor=t.GRID_2, linecolor=t.GRID, tickfont=dict(size=12, color=t.GRAPHITE))))
        t.chart(figp, height=400)
    with right:
        st.markdown("**What the evidence says**")
        st.markdown(me.driver_read(row))
        st.markdown(me.divergence_read(row["population_mortgage_gap_pp"]))
        st.markdown(me.evidence_read(row))
        metrics = [
            ("Purchase originations per 1,000 residents", "purchase_originations_per_1000_residents", "{:.1f}"),
            ("Origination rate %", "purchase_origination_rate_pct", "{:.1f}"),
            ("Denial rate %", "purchase_denial_rate_pct", "{:.1f}"),
            ("High-DTI share %", "high_dti_share_pct", "{:.1f}"),
            ("Income to property value %", "applicant_income_to_property_pct", "{:.1f}"),
            ("House prices, 1-yr %", "hpi_1y_pct", "{:+.1f}"),
            ("House prices, 3-yr CAGR %", "hpi_3y_cagr_pct", "{:+.1f}"),
            ("Government share of purchases %", "government_purchase_share_pct", "{:.1f}"),
            ("Active lenders", "active_lenders", "{:,.0f}"),
            ("Top-5 lender share %", "top5_lender_share_pct", "{:.1f}"),
        ]
        rows = []
        for label, col, fmt in metrics:
            v, m = row.get(col), peers[col].median()
            if pd.isna(m):
                continue  # input not in this build
            rows.append({"Measure": label, "This market": fmt.format(v) if pd.notna(v) else "n/a",
                         "Peer median": fmt.format(m) if pd.notna(m) else "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    tl, tr = st.columns([1.1, 1], gap="large")
    with tl:
        yr = years_df[years_df["market_key"].eq(row["market_key"])].sort_values("year") if years_df is not None else pd.DataFrame()
        if not yr.empty:
            figy = go.Figure(go.Bar(
                x=yr["year"].astype(str), y=yr["purchase_originations"], marker=dict(color=t.S1, cornerradius=4),
                text=[f"{v:,.0f}" for v in yr["purchase_originations"]], textposition="outside",
                textfont=dict(size=11, color=t.GRAPHITE), width=0.55,
                customdata=np.stack([yr["purchase_denial_rate_pct"], yr["refinance_share_pct"]], axis=1),
                hovertemplate="%{x}: %{y:,.0f} purchase originations<br>denial %{customdata[0]:.1f}% · refi share %{customdata[1]:.1f}%<extra></extra>"))
            figy.update_layout(title="Purchase originations by year (HMDA)", yaxis=dict(range=[0, yr["purchase_originations"].max() * 1.2]),
                               bargap=0.35)
            t.chart(figy, height=320)
        else:
            st.caption("Year-by-year series arrive with the next data build.")
    with tr:
        lt = lenders_df[lenders_df["market_key"].eq(row["market_key"])].nsmallest(8, "rank") if lenders_df is not None else pd.DataFrame()
        if not lt.empty:
            lt = lt.iloc[::-1]
            figl = go.Figure(go.Bar(
                y=[n.title()[:34] for n in lt["lender"]], x=lt["share_pct"], orientation="h",
                marker=dict(color=t.S3, cornerradius=4), text=[f"{v:.1f}%" for v in lt["share_pct"]], textposition="outside",
                textfont=dict(size=11, color=t.GRAPHITE), width=0.6,
                customdata=(lt["share_pct"] - lt["share_first_pct"]).round(1),
                hovertemplate="%{y}: %{x:.1f}% of HMDA records<br>change since first year %{customdata:+.1f} pts<extra></extra>"))
            figl.update_layout(title=dict(text="Largest lenders here, share of HMDA records", x=0, xref="container"),
                               xaxis=dict(visible=False, range=[0, lt["share_pct"].max() * 1.3]),
                               yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(l=8, r=8, t=60, b=8))
            t.chart(figl, height=320)
        else:
            st.caption("Lender shares arrive with the next data build.")

    st.markdown("**Ask about this market**")
    ctx = ask.context(row, peers, years_df, lenders_df)
    if ask.llm_available():
        q = st.text_input("Your question", placeholder="e.g. Is lending here keeping pace with population growth?",
                          key="mi_q")
        if q:
            with st.spinner("Reading the record…"):
                try:
                    st.markdown(f'<div class="lab-insight">{t.esc(ask.ask_llm(q, ctx))}</div>', unsafe_allow_html=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"The assistant is unavailable right now ({type(exc).__name__}).")
        st.caption("Answers come only from this market's record, shown below, and cite its numbers.")
    else:
        q = st.selectbox("Pick a question", ask.QUESTIONS, key="mi_q_rules")
        st.markdown(f'<div class="lab-insight">{t.esc(ask.ask_rules(q, row, ctx))}</div>', unsafe_allow_html=True)
        st.caption("Answers are generated from this market's record alone, so every number traces back to the data.")
    with st.expander("The record the answers are grounded in"):
        st.json(ctx, expanded=False)

# ---------------------------------------------------------------------------
# Lenders
# ---------------------------------------------------------------------------
with T["Lenders"]:
    if lenders_df is None or lenders_df.empty:
        st.info("Lender shares arrive with the next data build.")
    else:
        L = lenders_df.merge(scored[["market_key", "market_name", "area_type", "primary_state", "strategic_mortgage_opportunity_score"]],
                             on="market_key", how="inner")
        L = L[L["area_type"].eq(area)]
        national = L.groupby(["lei", "lender"], as_index=False)["records"].sum().sort_values("records", ascending=False)
        choices = national.head(300)
        label = [f"{n.title()}  ·  {int(r):,} records" for n, r in zip(choices["lender"], choices["records"])]
        sel = st.selectbox("Choose a lender (top 300 by HMDA records in this market type)", range(len(choices)),
                           format_func=lambda i: label[i], key="mi_lender")
        lei = choices.iloc[sel]["lei"]
        mine = L[L["lei"].eq(lei)].copy()
        mine["share_change_pp"] = mine["share_pct"] - mine["share_first_pct"]
        top_share = mine.sort_values("share_pct", ascending=False)
        t.tiles([
            {"label": "Markets with a top-40 position", "value": f"{len(mine):,}", "accent": True,
             "note": f"of {L['market_key'].nunique():,} {area.lower()} markets"},
            {"label": "Markets ranked #1", "value": f"{(mine['rank'] == 1).sum():,}"},
            {"label": "Median share where present", "value": f"{mine['share_pct'].median():.1f}%"},
            {"label": "Markets gaining share", "value": f"{(mine['share_change_pp'] > 0).sum():,}",
             "note": f"{(mine['share_change_pp'] < 0).sum():,} losing, since the first HMDA year"},
        ])
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            show = top_share.head(15).iloc[::-1]
            figb = go.Figure(go.Bar(y=[n.split(",")[0] for n in show["market_name"]], x=show["share_pct"], orientation="h",
                                    marker=dict(color=t.S1, cornerradius=4), text=[f"#{int(r)}" for r in show["rank"]],
                                    textposition="outside", textfont=dict(size=11, color=t.GRAPHITE), width=0.6,
                                    hovertemplate="%{y}: %{x:.1f}% share<extra></extra>"))
            figb.update_layout(title=dict(text="Strongest markets by share (label is rank)", x=0, xref="container"),
                               xaxis=dict(title="Share of HMDA records (%)", range=[0, show["share_pct"].max() * 1.2]),
                               yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(l=8, r=8, t=60, b=8))
            t.chart(figb, height=480)
        with right:
            pts = mine.dropna(subset=["strategic_mortgage_opportunity_score"])
            figs = go.Figure(go.Scatter(
                x=pts["strategic_mortgage_opportunity_score"], y=pts["share_pct"], mode="markers",
                marker=dict(size=8, color=np.where(pts["share_change_pp"] >= 0, t.S1, t.S2), opacity=0.8,
                            symbol=np.where(pts["share_change_pp"] >= 0, "circle", "diamond"), line=dict(color="#fff", width=1)),
                customdata=np.stack([pts["market_name"], pts["share_change_pp"].fillna(0)], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>opportunity %{x:.0f} · share %{y:.1f}%<br>change %{customdata[1]:+.1f} pts<extra></extra>"))
            figs.update_layout(title="Where it is strong vs where opportunity is", xaxis_title="Market opportunity score",
                               yaxis_title="Lender share (%)")
            t.chart(figs, height=480)
            st.caption("Teal circles: gaining share. Orange diamonds: losing share. High opportunity with low share is white space.")
        gap = mine[(mine["strategic_mortgage_opportunity_score"] >= 60)].nsmallest(5, "share_pct")
        if not gap.empty:
            t.insight("<b>White space:</b> high-opportunity markets where this lender's share is lowest: "
                      + t.esc(", ".join(n.split(",")[0] for n in gap["market_name"])) + ".")
        st.caption("Shares are of all HMDA records the lender filed in the market (applications of every type), not "
                   "originations or volume. Only each market's top 40 lenders are kept.")

# ---------------------------------------------------------------------------
# What changed
# ---------------------------------------------------------------------------
with T["What changed"]:
    if len(history) < 2:
        month = history[-1][0] if history else "n/a"
        st.info(f"The first monthly snapshot is {month}. Each automated build adds one; from the second, this tab ranks "
                "the markets whose scores moved most, and why.")
        st.caption("Snapshots live in data/history/ in the repository, one file per month.")
    else:
        (m0, h0), (m1, h1) = history[-2], history[-1]
        def _score(frame):
            s_ = me.score(me.prepare(frame), {k: v / 100 for k, v in weights.items()})
            return s_.set_index("market_key")[["market_name", "area_type", "strategic_mortgage_opportunity_score", "mortgage_risk_score"]]
        a0, a1 = _score(h0), _score(h1)
        d = a1.join(a0[["strategic_mortgage_opportunity_score", "mortgage_risk_score"]], rsuffix="_prev", how="inner")
        d = d[d["area_type"].eq(area)]
        d["Δ opportunity"] = d["strategic_mortgage_opportunity_score"] - d["strategic_mortgage_opportunity_score_prev"]
        d["Δ risk"] = d["mortgage_risk_score"] - d["mortgage_risk_score_prev"]
        st.markdown(f"Comparing **{m1}** with **{m0}**, using your current strategy weights.")
        c1, c2 = st.columns(2, gap="large")
        cfg = {c: st.column_config.NumberColumn(format="%+.1f") for c in ["Δ opportunity", "Δ risk"]}
        with c1:
            st.markdown("**Biggest risers**")
            st.dataframe(d.nlargest(10, "Δ opportunity")[["market_name", "Δ opportunity", "Δ risk"]].rename(columns={"market_name": "Market"}),
                         hide_index=True, column_config=cfg)
        with c2:
            st.markdown("**Biggest fallers**")
            st.dataframe(d.nsmallest(10, "Δ opportunity")[["market_name", "Δ opportunity", "Δ risk"]].rename(columns={"market_name": "Market"}),
                         hide_index=True, column_config=cfg)

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
with T["Signals"]:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Population ahead of lending**")
        st.caption("Demographic percentile at least 30 points above mortgage fundamentals. Possible unconverted demand.")
        up = view[view["population_mortgage_gap_pp"].ge(me.DIVERGENCE_PP)].nlargest(8, "population_mortgage_gap_pp")
        st.dataframe(up[["market_name", "population_mortgage_gap_pp", "demographic_percentile", "mortgage_fundamentals_percentile"]]
                     .rename(columns={"market_name": "Market", "population_mortgage_gap_pp": "Gap (pp)",
                                      "demographic_percentile": "Demographic pct", "mortgage_fundamentals_percentile": "Mortgage pct"}),
                     hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.0f") for c in ["Gap (pp)", "Demographic pct", "Mortgage pct"]})
    with b:
        st.markdown("**Lending ahead of population**")
        st.caption("Mortgage fundamentals at least 30 points above the demographic backdrop. Test durability.")
        dn = view[view["population_mortgage_gap_pp"].le(-me.DIVERGENCE_PP)].nsmallest(8, "population_mortgage_gap_pp")
        st.dataframe(dn[["market_name", "population_mortgage_gap_pp", "demographic_percentile", "mortgage_fundamentals_percentile"]]
                     .rename(columns={"market_name": "Market", "population_mortgage_gap_pp": "Gap (pp)",
                                      "demographic_percentile": "Demographic pct", "mortgage_fundamentals_percentile": "Mortgage pct"}),
                     hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.0f") for c in ["Gap (pp)", "Demographic pct", "Mortgage pct"]})

    st.markdown("**Unusual markets**")
    st.caption("IsolationForest on the archetype feature space. High scores mean a combination of signals few peers share, "
               "which is worth a look before it is worth a decision.")
    odd = view.dropna(subset=["anomaly_score"]).nlargest(8, "anomaly_score").copy()
    if not odd.empty:
        med = view[me.CLUSTER_FEATURES].median()
        def _why(r):
            dev = (r[me.CLUSTER_FEATURES] - med).dropna()
            top2 = dev.abs().sort_values(ascending=False).head(2).index
            return " · ".join(f"{'high' if dev[c] > 0 else 'low'} {BLOCK_SHORT[c].lower()}" for c in top2)
        odd["What stands out"] = odd.apply(_why, axis=1)
        st.dataframe(odd[["market_name", "anomaly_score", "What stands out", "mortgage_archetype"]]
                     .rename(columns={"market_name": "Market", "anomaly_score": "Anomaly percentile", "mortgage_archetype": "Archetype"}),
                     hide_index=True, column_config={"Anomaly percentile": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)})

# ---------------------------------------------------------------------------
# What could be built
# ---------------------------------------------------------------------------
with T["What could be built"]:
    st.markdown("This page is one view over a reusable market table. The same foundation supports a family of "
                "decision tools. Each one adds a single question and, usually, one internal data source.")
    ideas = [
        ("Territory and site planning", "Where should the next branch, team or hire go?",
         "Market scores + your footprint and production", "Contiguous territories balanced on workload, with a local ranking inside each."),
        ("Marketing budget allocator", "How should a fixed budget be split across markets?",
         "Demand, conversion and openness + cost-per-lead history", "An optimized spend plan with expected funded loans per market."),
        ("Competitive benchmarking (live in the Lenders tab)", "Where are we gaining or losing share, and to whom?",
         "Public HMDA by lender, year over year", "Share, rank and momentum against named peers in every market."),
        ("Conversational market brief (live in Market brief)", "What should I know about this market before Monday?",
         "This table as a grounded contract for an LLM", "Plain-language answers that cite the numbers and refuse to guess."),
        ("Early-warning monitor (started in What changed)", "Which markets changed enough this month to act on?",
         "Monthly FHFA and Census refresh, quarterly HMDA", "Alerts when risk, prices or demand cross agreed thresholds."),
        ("Access and fair-lending lens", "Where do outcomes differ across borrower groups?",
         "HMDA decisions by applicant group", "Reportable gaps with minimum-sample guards, for compliance review."),
        ("Product-market fit", "Which products should lead in which markets?",
         "Government share, CLTV, DTI, price-to-income", "A product emphasis map for marketing and loan-officer coaching."),
        ("Portfolio and servicing exposure", "How exposed is the book to softening markets?",
         "Market risk + servicing or MSR portfolio by geography", "Concentration and stress views by market and vintage."),
    ]
    cols3 = st.columns(2, gap="medium")
    for i, (title, q, src, out) in enumerate(ideas):
        with cols3[i % 2]:
            st.markdown(
                f'<div class="lab-card" style="margin-bottom:14px;height:auto"><h3 style="font-size:1.2rem">{t.esc(title)}</h3>'
                f'<p><b>Answers:</b> {t.esc(q)}</p><p style="margin-top:6px"><b>Built from:</b> {t.esc(src)}</p>'
                f'<p style="margin-top:6px"><b>Delivers:</b> {t.esc(out)}</p></div>',
                unsafe_allow_html=True,
            )

with st.expander("Method, sources and limits"):
    st.markdown(f"""
- **Grain.** One row per Core Based Statistical Area. Metros and micros are scored only against their own type.
- **Sources.** U.S. Census Bureau population estimates; HMDA through the FFIEC Data Browser API (applications, outcomes, loan types and active lenders by market); the FHFA House Price Index; and, where enabled, ACS 5-year income and home values. All public. A GitHub Actions pipeline rebuilds the table monthly.
- **Demographic block.** Within-type percentiles of population growth, latest-year growth, domestic and international migration, and natural change.
- **Blocks.** Each input becomes a within-type percentile. Blocks are weighted means of their inputs and need at least two observed. The opportunity score needs at least three blocks.
- **Risk.** Seven inverted inputs. Markets with house-price coverage are scored on the full basis; others on an HMDA-only basis, and the two are never ranked against each other.
- **Archetypes.** Rules on within-type thresholds, plus unsupervised clusters (robust scaling, PCA to 85% variance, KMeans with k chosen by silhouette among viable solutions). Anomalies come from IsolationForest in the same space.
- **Limits.** Percentiles show relative position, not absolute attractiveness. HMDA lags a year or more. Scores describe markets, not any lender's position in them.
""")

note = "Sample data with fictional markets." if not live else (
    f"Built from public data · {meta.get('markets', len(scored)):,} markets · generated {meta.get('generated_utc', 'n/a')[:10]}.")
t.footnote(note + " Not investment, lending or legal advice.")
