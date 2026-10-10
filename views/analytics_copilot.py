"""Analytics copilot: ask the warehouse in plain English, get governed SQL, a chart and an answer you can check."""

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG

from lab import theme as t
from lab import widgets as w
from lab.askdata import parser as P
from lab.askdata import semantic as S
from lab.cu import ui as cui

cui.css()
STAGES = ["Ask", "Metrics", "Evaluate", "Govern"]
KEYS = {s.lower(): s for s in STAGES}

t.header(
    "AI project · Corvane Connect (fictional)",
    "Ask the warehouse, get an answer you can check",
    "Type a business question. The copilot maps it to governed metric definitions, writes the SQL, runs it against a "
    "synthetic subscriber warehouse and returns a chart, the SQL and a one-line answer. Anything outside the definitions, "
    "and anything about individual customers, is declined.",
)

q = str(st.query_params.get("stage", "")).lower()
if "ac_stage" not in st.session_state:
    st.session_state.ac_stage = KEYS.get(q, STAGES[0])
    st.session_state.ac_stage__last = st.session_state.ac_stage
stage = w.segmented("Stage", STAGES, key="ac_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower()
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")

FMT = {"int": lambda v: f"{v:,.0f}", "pct": lambda v: f"{v:.2%}", "money": lambda v: f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}",
       "dec": lambda v: f"{v:,.1f}"}
EXAMPLES = {
    "Churn by region": "What is our churn rate by region?",
    "Net adds by month": "Net adds by month this year",
    "Business ARPU": "ARPU by plan for small business customers",
    "Contact reasons": "Top 3 contact reasons",
    "Promo churn trend": "Churn rate for promo switchers by month",
    "Emails (declined)": "List the names and emails of customers who churned",
}


def nice_period(df: pd.DataFrame) -> pd.DataFrame:
    if "period" in df and df["period"].astype(str).str.match(r"^\d{4}-\d{2}-01$").all():
        df = df.assign(period=pd.to_datetime(df["period"]).dt.strftime("%b %Y"))
    return df


def answer_sentence(it: S.Intent, df: pd.DataFrame) -> str:
    m = S.METRICS[it.metric]
    f = FMT[m.fmt]
    period = f"{S.month_label(it.start)} – {S.month_label(it.end)}" if it.start != it.end else S.month_label(it.end)
    filt = (" for " + ", ".join(f"{v}" for v in it.filters.values())) if it.filters else ""
    if df.empty or df["value"].isna().all():
        return "No rows match that question."
    if it.grain and not it.dims:
        first, last = df.iloc[0], df.iloc[-1]
        ch = (last["value"] / first["value"] - 1) if first["value"] else 0
        return (f"{m.label}{filt} went from <b>{f(first['value'])}</b> in {first['period']} to <b>{f(last['value'])}</b> in {last['period']} "
                f"({ch:+.0%}).")
    if it.dims and not it.grain:
        d = it.dims[0]
        g = df.groupby(d)["value"].sum() if len(it.dims) > 1 else df.set_index(d)["value"]
        hi, lo = g.idxmax(), g.idxmin()
        if len(g) == 1:
            return f"{m.label}{filt}, {period}: <b>{f(g.iloc[0])}</b> for {hi}."
        return (f"{m.label}{filt}, {period}: highest for <b>{hi}</b> ({f(g.max())}), lowest for <b>{lo}</b> ({f(g.min())}).")
    if not it.dims and not it.grain:
        when = S.month_label(it.end) if m.snapshot else period
        return f"{m.label}{filt}, {when}: <b>{f(df['value'].iloc[0])}</b>."
    return f"{m.label}{filt} by {S.DIMS[it.dims[0]]['label'].lower()} over time, {period}."


def chart(it: S.Intent, df: pd.DataFrame):
    m = S.METRICS[it.metric]
    yfmt = {"pct": ".1%", "money": "$,.0f", "int": ",.0f", "dec": ",.1f"}[m.fmt]
    palette = ["#CBFA7C", "#EF936F", "#8FA2F0", "#E07AA0", "#D9C24A", "#5CC3EE", "#A6B8AE"]
    fig = go.Figure()
    if it.grain:
        if it.dims:
            for i, (k, g) in enumerate(df.groupby(it.dims[0])):
                fig.add_trace(go.Scatter(x=g["period"].astype(str), y=g["value"], name=str(k), mode="lines+markers", line=dict(color=palette[i % 7], width=2)))
        else:
            fig.add_trace(go.Scatter(x=df["period"].astype(str), y=df["value"], mode="lines+markers", line=dict(color=t.S1, width=2.5), name=m.label))
        fig.update_layout(xaxis_title=it.grain.title())
    elif it.dims:
        d = it.dims[0]
        if len(it.dims) == 2:
            d2 = it.dims[1]
            for i, (k, g) in enumerate(df.groupby(d2)):
                fig.add_trace(go.Bar(y=g[d], x=g["value"], name=str(k), orientation="h", marker=dict(color=palette[i % 7])))
            fig.update_layout(barmode="group")
        else:
            fig.add_trace(go.Bar(y=df[d], x=df["value"], orientation="h", marker=dict(color=t.S1, cornerradius=3),
                                 text=[FMT[m.fmt](v) for v in df["value"]], textposition="outside", cliponaxis=False))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=yfmt))
    else:
        return None
    if it.grain:
        fig.update_layout(yaxis=dict(tickformat=yfmt, rangemode="tozero" if m.fmt != "pct" else None))
    fig.update_layout(title=m.label, legend=dict(orientation="h", y=1.02), margin=dict(t=90, l=8))
    return fig


# ===========================================================================
if stage == "Ask":
    cui.stage_head(1, "Ask", "What do you want to know?",
                   "Nine governed metrics over three years of monthly data: subscribers, sign-ups, churn, revenue, care contacts. "
                   "Break them down by region, channel, plan, segment or customer type, over any period.", "")
    st.markdown("**Try a question**")
    cols = st.columns(len(EXAMPLES))
    for c, (label, ex) in zip(cols, EXAMPLES.items()):
        if c.button(label, key=f"acx_{label}", width="stretch"):
            st.session_state["ac_q"] = ex
    question = st.text_input("Your question", key="ac_q", placeholder="e.g. How many new subscribers did we get by channel last year?")
    if question:
        mi = P.model_parse(question)
        it = mi or P.parse(question)
        left, right = st.columns([1.25, 1], gap="large")
        if it.refusal:
            with left:
                st.markdown('<div class="lab-reco" style="background:#2A3631;border-left-color:#EF936F"><div class="e">Declined</div>'
                            f'<div class="t">Not answerable from the governed data.</div><div class="d">{html.escape(it.refusal)}</div></div>',
                            unsafe_allow_html=True)
            with right:
                st.markdown("**Interpreted as**")
                st.code(json.dumps(it.as_dict(), indent=2, ensure_ascii=False), language="json")
        else:
            df = nice_period(S.run(it))
            sql = S.compile_sql(it)
            with left:
                st.markdown(f'<div class="lab-reco"><div class="e">Answer · governed metric</div>'
                            f'<div class="d" style="color:#F0F7ED;font-size:1.05rem">{answer_sentence(it, df)}</div></div>', unsafe_allow_html=True)
                fig = chart(it, df)
                if fig is not None:
                    t.chart(fig, height=380)
                show = df.copy()
                m = S.METRICS[it.metric]
                show["value"] = show["value"].map(FMT[m.fmt])
                show = show.rename(columns={"value": m.label, "period": "Period", **{d: S.DIMS[d]["label"] for d in S.DIMS}})
                st.dataframe(show, hide_index=True, width="stretch", height=min(38 * (len(show) + 1) + 4, 320))
                for n_ in it.notes:
                    st.caption(n_)
            with right:
                st.markdown("**SQL that ran**")
                st.code(sql, language="sql")
                st.markdown("**Interpreted as**")
                st.code(json.dumps(it.as_dict(), indent=2, ensure_ascii=False), language="json")
                st.markdown(f"**Definition** · {html.escape(m.label)} (owner: {html.escape(m.owner)})")
                st.caption(m.description)
        st.caption("Model mode: Claude mapped the question to the semantic layer; the intent was validated before the SQL was compiled." if mi else "Rules mode: phrasing is mapped to the semantic layer by deterministic rules, so the same question always gives the same SQL. "
                   "Add an API key and Claude handles freer phrasing; its output is validated against the same definitions before anything runs.")

# ===========================================================================
elif stage == "Metrics":
    cui.stage_head(2, "Metrics", "What does each number mean, exactly?",
                   "The semantic layer: every metric the copilot can use, its definition, its SQL and its owner. The copilot "
                   "can only compose these, which is what makes two people asking the same question get the same answer.", "")
    rows = []
    for m in S.METRICS.values():
        expr = m.numerator if not m.denominator else f"{m.numerator} / {m.denominator}"
        rows.append({"Metric": m.label, "SQL": expr + ("  · end of period" if m.snapshot else ""), "Owner": m.owner,
                     "Breakdowns": ", ".join(S.DIMS[d]["label"].lower() for d in m.dims)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=372)
    with st.expander("Definitions in full"):
        for m in S.METRICS.values():
            st.markdown(f"**{m.label}** · {m.description} *Also understood as: {', '.join(m.synonyms[:5])}.*")
    con = S.warehouse()
    tbls = []
    for name, desc in (("fact_subscriptions", "Monthly counts and revenue by region, channel, plan, segment and customer type"),
                       ("fact_care_contacts", "Monthly care contacts by region, segment and reason"),
                       ("dim_month", "Calendar: month, quarter, year and period-end flags")):
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        cols_ = [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]
        tbls.append({"Table": name, "Rows": f"{n:,}", "Contents": desc, "Columns": ", ".join(cols_)})
    st.markdown("**The warehouse**")
    st.dataframe(pd.DataFrame(tbls), hide_index=True, width="stretch")
    t.insight("Two metrics are <b>semi-additive</b>: active subscribers and revenue are snapshots, so a quarter reports its last month, "
              "never the sum of three. Ratios are computed from summed numerators and denominators, never by averaging rates. "
              "These rules live in the layer once, instead of in every analyst's query.")
    cui.call("The metric definitions are the product; the language model is the interface. Finance owns revenue, Marketing owns gross "
             "adds, Care owns contacts, and a change to any definition is reviewed like a code change.")

# ===========================================================================
elif stage == "Evaluate":
    cui.stage_head(3, "Evaluate", "How often does it get the question right?",
                   "A test set of questions with the intent each should produce, including ones it must decline, plus a "
                   "held-out set written after the rules were tuned. Every change runs both before release.", "")
    ev, ho = P.evaluate(), P.evaluate_heldout()
    first_ok, first_n = P.HELDOUT_FIRST_RUN
    t.tiles([
        {"label": "Test set, exact intent", "value": t.pct(ev["exact"].mean()), "accent": True, "note": f"{len(ev)} questions, metric, breakdown, grain and filters all right"},
        {"label": "Declines correct", "value": t.pct(ev.loc[ev["should_refuse"], "exact"].mean()), "note": f"{int(ev['should_refuse'].sum())} questions it must refuse"},
        {"label": "Held-out, first run", "value": f"{first_ok} of {first_n}", "accent": True, "note": "phrasings it had never seen"},
        {"label": "Held-out, today", "value": t.pct(ho["exact"].mean()), "note": "after fixing the one miss"},
    ])
    show = pd.concat([ev.assign(set="Test"), ho.assign(set="Held-out")])
    show["Result"] = show["exact"].map({True: "Correct", False: "Wrong"})
    st.dataframe(show[["set", "question", "expected", "got", "Result"]].rename(columns={"set": "Set", "question": "Question", "expected": "Expected", "got": "Got"}),
                 hide_index=True, width="stretch", height=420, column_config={"Set": st.column_config.TextColumn(width="small"),
                                                                             "Result": st.column_config.TextColumn(width="small")})
    t.insight("The held-out set did its job on the first run: “churn by zip code” was answered as overall churn, silently dropping a "
              "breakdown the warehouse does not have. That is the most dangerous kind of error, a confident answer to a different "
              "question. The fix declines any breakdown that is not defined, and the case joined the test set.")
    cui.call("Score the copilot on intents, not on prose. If the intent is right the SQL is right by construction, because SQL is "
             "compiled from definitions, not written by the model.")

# ===========================================================================
elif stage == "Govern":
    cui.stage_head(4, "Govern", "What keeps it safe to put in front of the business?",
                   "Guardrails are designed in, not bolted on: read-only access, aggregates only, definitions with owners, every "
                   "question logged with the SQL it produced.", "")
    cui.cards([
        ("Read-only, aggregates only", "The copilot queries monthly fact tables, never row-level customer records, and refuses any request to change data."),
        ("Compiled, not generated, SQL", "The model's only job is to pick a metric, breakdowns, filters and period. SQL is compiled from the semantic layer."),
        ("Declines over guesses", "Unknown metrics, unknown breakdowns and personal data are declined with a list of what is available."),
        ("Owned definitions", "Each metric has an owner. Definition changes go through review and are versioned like code."),
        ("Logged and replayable", "Every question is stored with its intent, SQL and result, so any answer can be reproduced later."),
        ("Measured before release", "A test set and a held-out set run on every change; misses become new test cases."),
    ])
    st.markdown("""
**Why not let the model write SQL directly?** It can, and often well, but a model writing SQL against raw tables will eventually average a rate, sum a snapshot or join on the wrong grain, and its answer will look exactly as confident as a right one. Compiling SQL from governed definitions makes those mistakes impossible rather than unlikely.

**What would change at scale.** The same semantic layer in dbt or a BI tool, permissions inherited from the warehouse, usage analytics on which questions get asked (and declined) to decide which metrics to define next.
""")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.ac_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Corvane Connect is fictional and the warehouse is synthetic, built from the subscriber lab. Nothing here is real customer data.")
