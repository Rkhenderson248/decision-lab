"""Decision platform: the engineering behind one decision, end to end, on public mortgage data.

Ingest → layered models with tests → governed metrics → train and register → serve → monitor → architecture.
Everything is computed live from the repository's data files (built monthly by pipeline/build_markets.py).
"""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.cu import ui as cui
from lab.platform import metrics as K
from lab.platform import ml
from lab.platform import models as M

cui.css()
STAGES = ["Ingest", "Model", "Metrics", "Train", "Serve", "Monitor", "Architecture"]
KEYS = {s.lower(): s for s in STAGES}
ROOT = Path(__file__).resolve().parent.parent
LAYER_COLOR = {"bronze": "#C9935A", "silver": "#A6B8AE", "gold": "#CBFA7C"}

t.header(
    "Project · data & ML engineering · public mortgage data",
    "The platform behind the decision",
    "Most portfolios show the model. This shows the system around it: a scheduled pipeline that lands public data, "
    "layered SQL models with tests, governed metrics, a model trained and promoted through a gate, a scoring service "
    "that validates its inputs, and monitoring that says when to retrain. Every number is computed live from the data.",
)


@st.cache_resource(show_spinner="Building the warehouse…")
def platform():
    run = M.build()
    versions, info = ml.train(run.con)
    return run, versions, info


RUN, VERSIONS, INFO = platform()
CHAMP = INFO["champion"]

q = str(st.query_params.get("stage", "")).lower()
if "dp_stage" not in st.session_state:
    st.session_state.dp_stage = KEYS.get(q, STAGES[0])
    st.session_state.dp_stage__last = st.session_state.dp_stage
stage = w.segmented("Stage", STAGES, key="dp_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower()
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")


def flow_diagram(active: str | None = None) -> str:
    """The whole platform as one row of boxes; the current stage is lit."""
    steps = [
        ("Ingest", "Census · FHFA · HMDA", "GitHub Actions, monthly"),
        ("Model", "Bronze → silver → gold", "SQL models + tests"),
        ("Metrics", "Governed definitions", "Owners, KPI tree"),
        ("Train", "Candidates + registry", "Promotion gate"),
        ("Serve", "Scoring service", "Schema, reason codes"),
        ("Monitor", "Drift + freshness", "Retrain rules"),
    ]
    cells = []
    for i, (k, a, b) in enumerate(steps):
        on = k == active
        cells.append(
            f'<div style="flex:1;min-width:120px;padding:12px 12px 10px;border:1px solid {"#CBFA7C" if on else "#263E36"};'
            f'background:{"rgba(203,250,124,.08)" if on else "#142322"}">'
            f'<div style="font:700 .66rem Inter,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:{"#CBFA7C" if on else "#A6B8AE"}">{i + 1:02d} · {k}</div>'
            f'<div style="font:600 .92rem \'Space Grotesk\',Inter,sans-serif;color:#F0F7ED;margin-top:4px">{a}</div>'
            f'<div style="font-size:.76rem;color:#A6B8AE">{b}</div></div>'
        )
        if i < len(steps) - 1:
            cells.append('<div style="align-self:center;color:#6E827A;font-size:1rem">→</div>')
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:.2rem 0 1rem">{"".join(cells)}</div>'


# ===========================================================================
if stage == "Ingest":
    cui.stage_head(1, "Ingest", "Where does the data come from, and how fresh is it?",
                   "A scheduled job pulls three public sources, writes one table per run, and refuses to overwrite good data "
                   "with a thin pull. This stage reads the job's own build log.", "")
    st.markdown(flow_diagram("Ingest"), unsafe_allow_html=True)
    logs = M.build_log()
    meta = logs.get("markets_meta.json", {})
    steps = logs.get("build_log.json", {}).get("steps", {})
    gen = meta.get("generated_utc", "")
    age = None
    if gen:
        try:
            age = (datetime.now(timezone.utc) - datetime.strptime(gen, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).days
        except ValueError:
            age = None
    hm = steps.get("hmda", {})
    t.tiles([
        {"label": "Last build", "value": gen[:10] or "–", "accent": True, "note": f"{age} days ago · rebuilt on the 6th of each month" if age is not None else "rebuilt monthly"},
        {"label": "Markets landed", "value": f"{meta.get('markets', 0):,}", "note": "every U.S. metro and micro area"},
        {"label": "HMDA requests", "value": f"{hm.get('requests', 0):,}", "note": f"{hm.get('activity_failures', 0)} failed and were logged, not hidden"},
        {"label": "Freshness SLA", "value": "Pass" if age is not None and age <= 40 else "Breach", "note": "data older than 40 days fails the check"},
    ])
    src = [
        ("U.S. Census Bureau", "Population estimates by CBSA", f"vintage {meta.get('census_vintage', '–')}", f"{steps.get('census_popest', {}).get('markets', '–')} markets"),
        ("FHFA", "House Price Index, metro and division", f"through {meta.get('fhfa_latest_year', '–')}", f"{steps.get('fhfa', {}).get('series', '–')} series"),
        ("FFIEC HMDA API", "Purchase and refinance activity, lenders", f"{', '.join(str(y) for y in meta.get('hmda_years', []))}", f"{hm.get('markets', '–')} markets"),
    ]
    st.dataframe(pd.DataFrame(src, columns=["Source", "What it provides", "Coverage", "Landed"]), hide_index=True, width="stretch")
    cov = meta.get("coverage", {})
    if cov:
        n = max(meta.get("markets", 1), 1)
        cdf = pd.DataFrame({"field": list(cov), "coverage": [v / n for v in cov.values()]}).sort_values("coverage")
        fig = go.Figure(go.Bar(y=cdf["field"], x=cdf["coverage"], orientation="h", marker=dict(color=[t.S1 if v >= .95 else t.S2 for v in cdf["coverage"]], cornerradius=2)))
        fig.update_layout(title="Field coverage in the latest build", xaxis=dict(tickformat=".0%", range=[0, 1.05]), margin=dict(l=8, t=60))
        t.chart(fig, height=max(320, 18 * len(cdf)))
    wf = []
    for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        txt = f.read_text()
        name = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.startswith("name:")), f.stem)
        cron = next((ln.split("cron:", 1)[1].strip().strip('"') for ln in txt.splitlines() if "cron:" in ln), "")
        trig = ", ".join(x for x in ("schedule" if cron else "", "push" if "push:" in txt else "", "manual" if "workflow_dispatch" in txt else "") if x)
        wf.append({"Workflow": name, "File": f".github/workflows/{f.name}", "Triggers": trig, "Schedule (cron, UTC)": cron or "–"})
    if wf:
        st.markdown("**The jobs that run it** (read from the repository)")
        st.dataframe(pd.DataFrame(wf), hide_index=True, width="stretch")
    cui.call("Fail loudly, keep the last good table. If HMDA coverage comes back thin, the build keeps the previous file and marks the "
             "run failed, so a bad pull never reaches a decision quietly.")

# ===========================================================================
elif stage == "Model":
    cui.stage_head(2, "Model", "How does raw data become a table you can trust?",
                   "Three layers, each a SQL model that declares its inputs: bronze as landed, silver cleaned one source at a "
                   "time, gold joined and ready for analysis. Tests run after every build, the way dbt does it.", "")
    st.markdown(flow_diagram("Model"), unsafe_allow_html=True)
    mdf, tdf = RUN.models, RUN.tests
    npass, nwarn, nfail = (tdf["status"] == "pass").sum(), (tdf["status"] == "warn").sum(), (tdf["status"] == "fail").sum()
    t.tiles([
        {"label": "Models built", "value": str(len(mdf)), "note": f"{(mdf.layer == 'bronze').sum()} bronze · {(mdf.layer == 'silver').sum()} silver · {(mdf.layer == 'gold').sum()} gold"},
        {"label": "Build time", "value": f"{mdf['ms'].sum() / 1000:.2f}s", "note": "in-memory warehouse, this session"},
        {"label": "Data tests", "value": f"{npass} of {len(tdf)} pass", "accent": nfail == 0, "note": f"{nwarn} warnings · {nfail} failures"},
        {"label": "Rows in gold", "value": f"{int(mdf.loc[mdf.layer == 'gold', 'rows'].sum()):,}", "note": "across the gold tables"},
    ])
    # Lineage graph
    order = M.order()
    layer_of = dict(zip(mdf["model"], mdf["layer"]))
    cols = {"bronze": 0, "silver": 1, "gold": 2}
    pos, counts = {}, {"bronze": 0, "silver": 0, "gold": 0}
    for n in order:
        L = layer_of[n]
        pos[n] = (cols[L], counts[L])
        counts[L] += 1
    fig = go.Figure()
    for a, b in M.lineage():
        (x0, y0), (x1, y1) = pos[a], pos[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[-y0, -y1], mode="lines", line=dict(color="#3A5A4E", width=1.5), hoverinfo="skip", showlegend=False))
    for L in ("bronze", "silver", "gold"):
        ns = [n for n in order if layer_of[n] == L]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[-pos[n][1] for n in ns], mode="markers+text", text=ns, textposition="top center",
                                 marker=dict(size=16, color=LAYER_COLOR[L], line=dict(color="#101B1B", width=2)), name=L.title(),
                                 hovertext=[f"{n}: {int(mdf.loc[mdf.model == n, 'rows'].iloc[0]):,} rows" for n in ns], hoverinfo="text"))
    fig.update_layout(title="Lineage: what each model is built from", xaxis=dict(visible=False, range=[-.4, 2.4]), yaxis=dict(visible=False, range=[-3.2, .6]),
                      legend=dict(orientation="h", y=1.08), margin=dict(l=8, r=8, t=80))
    t.chart(fig, height=380)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("**Models**")
        show = mdf.assign(rows=mdf["rows"].map("{:,}".format), ms=mdf["ms"].map("{:.0f} ms".format))[["model", "layer", "grain", "rows", "ms", "owner"]]
        st.dataframe(show.rename(columns=str.title), hide_index=True, width="stretch", height=360)
    with right:
        st.markdown("**Tests**")
        st.dataframe(tdf[["model", "test", "kind", "status", "failing_rows"]].rename(columns={"failing_rows": "Failing rows"}).rename(columns=str.title),
                     hide_index=True, width="stretch", height=360)
    pick = st.selectbox("Read a model's SQL", [m.name for m in M.MODELS], index=len(M.MODELS) - 2)
    m = next(x for x in M.MODELS if x.name == pick)
    st.caption(m.description)
    st.code("\n".join(ln[8:] if ln.startswith("        ") else ln for ln in m.sql.strip("\n").splitlines()), language="sql")
    cui.call("Warnings are information, failures stop the line. Micro areas have no FHFA index, so house-price coverage is a warning "
             "with a threshold; a duplicate market key would be a failure that blocks every downstream table.")

# ===========================================================================
elif stage == "Metrics":
    cui.stage_head(3, "Metrics", "Which numbers does the business see, and who owns them?",
                   "Six governed metrics over the gold layer, each with one definition and one owner. Ratios come from summed "
                   "numerators and denominators, so a national rate is never an average of market rates.", "")
    st.markdown(flow_diagram("Metrics"), unsafe_allow_html=True)
    con = RUN.con
    states = ["All states"] + sorted(pd.read_sql_query("SELECT DISTINCT state FROM gold_market_year_features WHERE state IS NOT NULL", con)["state"])
    c1, c2 = st.columns([1, 1])
    st_pick = c1.selectbox("State", states, key="dp_state")
    area = c2.radio("Area type", ["All", "Metro", "Micro"], horizontal=True, key="dp_area")
    cond = []
    if st_pick != "All states":
        cond.append(f"state = '{st_pick.replace(chr(39), '')}'")
    if area != "All":
        cond.append(f"area_type = '{area}'")
    where = " AND ".join(cond)
    by_year = K.compute(con, by="year", where=where)
    last, prev = by_year.iloc[-1], by_year.iloc[-2]
    t.tiles([
        {"label": f"Purchase originations, {int(last.year)}", "value": K.fmt("originations", last.originations), "accent": True,
         "note": f"{(last.originations / prev.originations - 1):+.1%} vs {int(prev.year)}"},
        {"label": "Origination rate", "value": K.fmt("origination_rate", last.origination_rate), "note": f"{(last.origination_rate - prev.origination_rate) * 100:+.1f} pts"},
        {"label": "Denial rate", "value": K.fmt("denial_rate", last.denial_rate), "note": f"{(last.denial_rate - prev.denial_rate) * 100:+.1f} pts"},
        {"label": "Average purchase loan", "value": K.fmt("avg_loan", last.avg_loan), "note": f"{(last.avg_loan / prev.avg_loan - 1):+.1%}"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=by_year["year"].astype(str), y=by_year["applications"], name="Applications", marker=dict(color="#3A5A4E", cornerradius=2)))
        fig.add_trace(go.Bar(x=by_year["year"].astype(str), y=by_year["originations"], name="Originations", marker=dict(color=t.S1, cornerradius=2)))
        fig.update_layout(title="Purchase applications and originations", barmode="overlay", legend=dict(orientation="h", y=1.08), margin=dict(t=80))
        t.chart(fig, height=340)
    with right:
        st.markdown("**KPI tree**")
        st.markdown(
            f'<div style="font-size:.92rem;line-height:1.9;color:#C9D6CE">'
            f'<b style="color:#CBFA7C">Originations</b> {K.fmt("originations", last.originations)}<br>'
            f'&nbsp;&nbsp;= Applications {K.fmt("applications", last.applications)}<br>'
            f'&nbsp;&nbsp;× Origination rate {K.fmt("origination_rate", last.origination_rate)}<br>'
            f'<span style="color:#A6B8AE">&nbsp;&nbsp;&nbsp;&nbsp;driven by denial rate {K.fmt("denial_rate", last.denial_rate)} and withdrawals</span><br>'
            f'<b style="color:#CBFA7C">Volume</b> ≈ Originations × average loan {K.fmt("avg_loan", last.avg_loan)}</div>',
            unsafe_allow_html=True)
        st.caption("A change in originations always decomposes into demand (applications) and conversion (origination rate).")
    st.markdown("**Definitions**")
    st.dataframe(pd.DataFrame([{"Metric": m.label, "Definition": m.definition, "SQL": m.sql, "Owner": m.owner} for m in K.METRICS]),
                 hide_index=True, width="stretch")
    top = pd.read_sql_query(
        "SELECT state, SUM(originations) AS originations, SUM(originations)*1.0/SUM(applications) AS origination_rate "
        f"FROM gold_market_year_features WHERE year = (SELECT MAX(year) FROM gold_market_year_features){(' AND area_type = ' + repr(area)) if area != 'All' else ''} "
        "GROUP BY state ORDER BY originations DESC LIMIT 12", con)
    fig = go.Figure(go.Bar(y=top["state"], x=top["originations"], orientation="h", marker=dict(color=t.S1, cornerradius=2),
                           text=[f"{v:.0%}" for v in top["origination_rate"]], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Top states by purchase originations (label: origination rate)", yaxis=dict(autorange="reversed"), margin=dict(l=8, t=70))
    t.chart(fig, height=380)
    cui.call("Define once, use everywhere. These definitions are the same ones a BI semantic model or the Analytics copilot would use, "
             "so a dashboard, an analyst's query and an AI answer all report the same number.")

# ===========================================================================
elif stage == "Train":
    cui.stage_head(4, "Train", "Is the model good enough to promote?",
                   "The decision: which markets will grow purchase lending next year. Four candidates train on 2023 features "
                   "to predict 2024 growth and are judged out of time, on 2024 features predicting 2025. Every run is "
                   "registered with its data hash, and only a model that clears the gate is promoted.", "")
    st.markdown(flow_diagram("Train"), unsafe_allow_html=True)
    tr, te = INFO["train"], INFO["test"]
    t.tiles([
        {"label": "Training rows", "value": f"{len(tr):,}", "note": f"markets with ≥ {ml.MIN_APPS} applications, {ml.TRAIN_YEAR}"},
        {"label": "Out-of-time rows", "value": f"{len(te):,}", "note": f"{ml.TEST_YEAR} features, {ml.TEST_YEAR + 1} outcomes"},
        {"label": "Champion", "value": CHAMP.version, "accent": True, "note": CHAMP.name},
        {"label": "Top-decile lift", "value": f"{CHAMP.lift * 100:+.1f} pts", "note": "growth of the top 10% picked vs all markets"},
    ])
    reg = pd.DataFrame([{
        "Version": v.version, "Model": v.name, "Params": v.params, "Stage": v.stage,
        "Rank corr": "–" if np.isnan(v.spearman) else f"{v.spearman:.2f}", "Top-decile lift": f"{v.lift * 100:+.1f} pts",
        "Error (wMAE)": f"{v.mae * 100:.1f} pts", "Data hash": v.data_hash, "Gate": v.gate} for v in VERSIONS])
    st.markdown("**Model registry**")
    st.dataframe(reg, hide_index=True, width="stretch", column_config={"Gate": st.column_config.TextColumn(width="large")})
    left, right = st.columns([1, 1], gap="large")
    with left:
        p = CHAMP.model.predict(te[list(ml.FEATURES)])
        dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"decile": dec + 1, "y": te[ml.TARGET].values}).groupby("decile")["y"].mean()
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker=dict(color=[t.S1 if i == 10 else "#3A5A4E" for i in g.index], cornerradius=2)))
        fig.add_hline(y=float(te[ml.TARGET].mean()), line=dict(color=t.S2, dash="dot"), annotation_text="all markets", annotation_font_color=t.S2)
        fig.update_layout(title=f"Actual {ml.TEST_YEAR + 1} growth by predicted decile ({CHAMP.version})", xaxis_title="Predicted decile (10 = highest)",
                          yaxis=dict(tickformat=".0%"), margin=dict(t=70))
        t.chart(fig, height=340)
    with right:
        if hasattr(CHAMP.model, "steps"):
            ridge = CHAMP.model.steps[-1][1]
            coef = pd.Series(ridge.coef_, index=[ml.FEATURES[k][0] for k in ml.FEATURES]).sort_values()
            fig = go.Figure(go.Bar(y=coef.index, x=coef.values, orientation="h", marker=dict(color=[t.S1 if v > 0 else t.S2 for v in coef.values], cornerradius=2)))
            fig.update_layout(title="What moves the prediction (standardized coefficients)", margin=dict(l=8, t=70))
            t.chart(fig, height=340)
        st.markdown("**Model card**")
        st.markdown(
            f"- **Decision:** which markets to prioritize for purchase lending next year\n"
            f"- **Owner:** Market strategy · **Version:** {CHAMP.version} ({CHAMP.name}), data hash `{CHAMP.data_hash}`\n"
            f"- **Use:** ranking markets. **Not for:** forecasting a market's exact growth\n"
            f"- **Known limit:** error is higher than the median baseline's ({CHAMP.mae * 100:.1f} vs {INFO['baseline_mae'] * 100:.1f} pts): it ranks well and misses the level\n"
            f"- **Data:** public HMDA aggregates; no applicant-level or employer data")
    cui.call("Gate on what the decision needs. The question is which markets to prioritize, so the gate is ranking: rank correlation and "
             "top-decile lift. Gradient boosting has lower error but cannot rank, so it fails; the ridge model ranks, so it ships, "
             "with its forecast error written on the model card instead of hidden.")

# ===========================================================================
elif stage == "Serve":
    cui.stage_head(5, "Serve", "What happens when another system asks for a score?",
                   "The production model sits behind a scoring contract: a request is validated against a schema, scored, "
                   "and returned with the model version and reason codes. A bad request gets a 422, never a guess.", "")
    st.markdown(flow_diagram("Serve"), unsafe_allow_html=True)
    sc = INFO["score"].merge(pd.read_sql_query("SELECT market_key, market_name FROM gold_markets", RUN.con), on="market_key", how="left")
    sc = sc.sort_values("applications", ascending=False)
    names = sc["market_name"].fillna(sc["market_key"].astype(str)).tolist()
    pick = st.selectbox("Market to score (its latest HMDA year as the request)", names, index=0, key="dp_mkt")
    row = sc.iloc[names.index(pick)]
    payload = {k: round(float(row[k]), 4) for k in ml.FEATURES}
    broken = st.toggle("Send a malformed request", key="dp_bad", help="Drops one field and puts another out of range, to show the contract rejecting it.")
    if broken:
        payload = dict(payload)
        payload.pop("denial_rate", None)
        payload["origination_rate"] = 1.7
    resp = ml.score(CHAMP, payload, INFO["train"])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Request** · `POST /v1/markets/score`")
        st.code(json.dumps({"market": pick, "features": payload}, indent=2), language="json")
        st.markdown("**Schema**")
        st.dataframe(pd.DataFrame([{"Field": k, "Meaning": v[0], "Allowed": f"{v[1]} to {v[2]}"} for k, v in ml.FEATURES.items()]),
                     hide_index=True, width="stretch")
    with right:
        st.markdown(f"**Response** · `{resp['status']}`")
        st.code(json.dumps(resp, indent=2), language="json")
        if resp["status"] == 200:
            g = resp["prediction"]["next_year_origination_growth"]
            st.markdown(f'<div class="lab-reco"><div class="e">Score</div><div class="t">{html.escape(pick)}: {g:+.1%} predicted origination growth</div>'
                        f'<div class="d">Use it to rank, not to forecast: compare it with other markets\' scores.</div></div>', unsafe_allow_html=True)
    st.markdown(f"**Batch scoring** · every market's {ml.SCORE_YEAR} features scored for {ml.SCORE_YEAR + 1}")
    pred = CHAMP.model.predict(sc[list(ml.FEATURES)])
    batch = pd.DataFrame({"Market": names, "State": sc["state"].values, "Applications": sc["applications"].map("{:,.0f}".format).values,
                          "Predicted growth": pred}).sort_values("Predicted growth", ascending=False)
    batch["Rank"] = range(1, len(batch) + 1)
    batch["Predicted growth"] = batch["Predicted growth"].map("{:+.1%}".format)
    st.dataframe(batch[["Rank", "Market", "State", "Applications", "Predicted growth"]].head(25), hide_index=True, width="stretch", height=400)
    cui.call("The contract outlives the model. Callers depend on the schema and the response shape, so a new model version can be "
             "promoted without anyone downstream changing code, and every response says which version answered.")

# ===========================================================================
elif stage == "Monitor":
    cui.stage_head(6, "Monitor", "How do you know when the model has gone stale?",
                   "Three signals, checked on every build: are the inputs still shaped like the training data (drift), is the "
                   "data fresh, and do the tests still pass. Each has a rule that says what happens next.", "")
    st.markdown(flow_diagram("Monitor"), unsafe_allow_html=True)
    d = ml.drift(INFO["train"], INFO["score"])
    n_re = int((d["status"] == "Retrain").sum())
    tdf = RUN.tests
    meta = M.build_log().get("markets_meta.json", {})
    gen = meta.get("generated_utc", "")
    try:
        age = (datetime.now(timezone.utc) - datetime.strptime(gen, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).days
    except ValueError:
        age = None
    t.tiles([
        {"label": "Retrain trigger", "value": "Fired" if n_re else "Clear", "accent": bool(n_re), "note": f"{n_re} feature(s) with PSI ≥ 0.25"},
        {"label": "Data freshness", "value": f"{age} days" if age is not None else "–", "note": "limit 40 days"},
        {"label": "Data tests", "value": f"{(tdf.status == 'pass').sum()} of {len(tdf)}", "note": "failures block scoring"},
        {"label": "Serving version", "value": CHAMP.version, "note": CHAMP.name},
    ])
    fig = go.Figure(go.Bar(y=d["feature"], x=d["psi"], orientation="h",
                           marker=dict(color=[t.S2 if s == "Retrain" else ("#D9C24A" if s == "Watch" else t.S1) for s in d["status"]], cornerradius=2)))
    fig.add_vline(x=0.1, line=dict(color="#6E827A", dash="dot"), annotation_text="watch", annotation_position="bottom right", annotation_font_color="#A6B8AE")
    fig.add_vline(x=0.25, line=dict(color=t.S2, dash="dot"), annotation_text="retrain", annotation_position="top right", annotation_font_color=t.S2)
    fig.update_layout(title=f"Input drift: training ({ml.TRAIN_YEAR}) vs scoring ({ml.SCORE_YEAR}), population stability index",
                      xaxis=dict(type="log", title="PSI (log scale)"), margin=dict(l=8, t=70))
    t.chart(fig, height=340)
    feat = st.selectbox("Compare one feature's distribution", [k for k in ml.FEATURES if k != "is_metro"], format_func=lambda k: ml.FEATURES[k][0], key="dp_feat")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=INFO["train"][feat], name=f"Training ({ml.TRAIN_YEAR})", marker_color="#3A5A4E", opacity=.85, nbinsx=40))
    fig.add_trace(go.Histogram(x=INFO["score"][feat], name=f"Scoring ({ml.SCORE_YEAR})", marker_color=t.S1, opacity=.6, nbinsx=40))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.08), margin=dict(t=60))
    t.chart(fig, height=300)
    st.markdown("**Runbook**")
    st.dataframe(pd.DataFrame([
        ("PSI ≥ 0.25 on any input", "Retrain on the latest labeled year, re-run the gate, promote only on a pass"),
        ("PSI 0.10 – 0.25", "Watch: review at the next monthly build"),
        ("Data older than 40 days", "Page the owner; scoring keeps the last good table"),
        ("A data test fails", "Block the build; no new scores until it is fixed"),
        ("Gate fails after retraining", "Keep the current production version; open a model review"),
    ], columns=["Signal", "Action"]), hide_index=True, width="stretch")
    t.insight(f"The retrain trigger is firing for a real reason: {ml.TRAIN_YEAR} was the year after the rate shock, when origination growth "
              "collapsed almost everywhere, so this year's growth and refinance share look nothing like the training year. "
              "A model trained on one regime and scored in another is exactly what monitoring exists to catch.")
    cui.call("Monitor the inputs, not just the accuracy. Outcomes arrive a year late here, so input drift is the early warning; by the "
             "time accuracy drops, a year of decisions has already used the stale model.")

# ===========================================================================
elif stage == "Architecture":
    cui.stage_head(7, "Architecture", "What does this look like inside a real organization?",
                   "The same design, mapped from this demo's lightweight parts to the enterprise tools that would replace them, "
                   "with the choices that matter: what to build, what to buy, and where the risks sit.", "")
    st.markdown(flow_diagram(None), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ("Ingest", "GitHub Actions, Python requests", "Airflow, Dagster or Databricks Workflows; Fivetran for SaaS sources", "Buy connectors, build the domain logic"),
        ("Store", "CSV in the repository", "Lakehouse (Delta, Iceberg) on Databricks or Snowflake", "Buy"),
        ("Model", "SQL models in SQLite with tests", "dbt models and tests, with CI on every pull request", "Build on a standard tool"),
        ("Metrics", "Python metric definitions", "Semantic layer: dbt metrics, Power BI or Looker models", "Build definitions, buy the layer"),
        ("Train + registry", "scikit-learn, in-memory registry", "MLflow tracking and model registry", "Buy the registry, build the gate"),
        ("Serve", "Validated scoring function", "Model serving endpoint behind an API gateway", "Buy serving, own the contract"),
        ("Monitor", "PSI, freshness and test checks", "Lakehouse monitoring or Evidently, alerts to the owner", "Buy tooling, own the thresholds"),
    ], columns=["Layer", "In this demo", "In production", "Build or buy"]), hide_index=True, width="stretch")
    cui.cards([
        ("Security", "Read-only service accounts per layer; aggregates only in this domain; secrets in a vault, never in code."),
        ("Cost", "Monthly batch, not streaming: the decision changes monthly, so the platform should too. Compute scales to zero between runs."),
        ("Ownership", "Every model, metric and test has a named owner, and a change to any of them is reviewed like code."),
        ("Failure modes", "Thin source pulls, silent schema changes, regime shifts. Each has a check above that stops the line."),
        ("Where AI fits", "The governed metrics are what an AI assistant should query. The Analytics copilot shows that layer in use."),
        ("Time to value", "Ingest, gold tables and three metrics first; the model and monitoring follow once a decision owner is using the numbers."),
    ])
    cui.call("Architecture follows the decision. A monthly market-prioritization decision does not need streaming or a feature store; it "
             "needs reliable monthly data, definitions people agree on, and a model that is retrained when the world moves.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.dp_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Public data: U.S. Census Bureau population estimates, FHFA House Price Index and HMDA via the FFIEC Data Browser API. "
           "The model and thresholds are illustrative.")
