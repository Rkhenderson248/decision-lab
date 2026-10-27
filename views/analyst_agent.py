"""Analyst agent: a model answers business questions by writing SQL against a read-only warehouse. Real runs are
recorded once and replayed here step by step; no model is called when this page is viewed."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.ai import agent as A
from lab.ai import evals as E
from lab.ai import replay as R
from lab.ai import ui as U
from lab.ai import warehouse as W
from lab.cu import ui

ui.css()
STAGES = ["The warehouse", "Guardrails", "Replay", "Cost & speed"]

t.header(
    "AI engineering · Larkspur Market (fictional)",
    "Analyst agent: questions in, SQL out, every step on the record",
    "A model gets four tools (list tables, describe a table, run a read-only query, give a final answer) and a "
    "grocery chain's sales warehouse. It has to find the right tables, write the SQL, check its own errors and "
    "refuse what it should not do. The runs below were recorded once against the real API and are replayed exactly.",
)

data = R.load()
models = R.analyst_models()


@st.cache_resource
def _con():
    return W.connect()


con = _con()

q = str(st.query_params.get("stage", "")).lower().replace("-", "").replace(" ", "").replace("&", "")
lookup = {s.lower().replace(" ", "").replace("&", ""): s for s in STAGES}
lookup.update({"warehouse": "The warehouse", "trace": "Replay", "cost": "Cost & speed"})
if "aa_stage" not in st.session_state:
    st.session_state.aa_stage = lookup.get(q, STAGES[0])
    st.session_state.aa_stage__last = st.session_state.aa_stage
stage = w.segmented("Stage", STAGES, key="aa_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace(" & ", "-").replace(" ", "-")
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")


def graded(model: str) -> list[dict]:
    out = []
    for c in E.CASES:
        for tr in data["analyst"].get(model, {}).get(c.id, [])[:1]:
            out.append({**E.grade(c, tr), "trace": tr, "case": c})
    return out


# ===========================================================================
if stage == "The warehouse":
    ui.stage_head(1, "The warehouse", "What can the agent see?",
                  "Four tables built from the forecasting lab's grocery chain: stores, categories, three years of "
                  "weekly sales by store and category, and a contacts table with personal data the agent must never "
                  "repeat. The agent sees only what its tools return.", "build")
    fr = W.frames()
    t.tiles([
        {"label": "Tables", "value": f"{len(fr)}", "note": "SQLite, in memory, rebuilt from a seed"},
        {"label": "Sales rows", "value": f"{len(fr['weekly_sales']):,}", "note": "156 weeks × 12 stores × 6 categories"},
        {"label": "Tools", "value": f"{len(A.TOOLS)}", "accent": True, "note": "list, describe, query, answer"},
        {"label": "Rows per query", "value": f"{W.MAX_ROWS}", "note": "cap on what comes back to the model"},
    ])
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.markdown("##### Schema")
        U.table([{"Table": n, "Rows": f"{len(df):,}", "Columns": ", ".join(df.columns)} for n, df in fr.items()])
        st.markdown("##### The tools, as the model receives them")
        U.table([{"Tool": tl["name"], "Inputs": ", ".join(tl["input_schema"]["properties"]) or "none", "What it does": tl["description"]} for tl in A.TOOLS])
    with b:
        st.markdown("##### The system prompt, word for word")
        st.code(A.SYSTEM, language="markdown", wrap_lines=True)
        st.caption(f"Prompt and tools hash {A.PROMPT_HASH}. A recording made with a different prompt shows a different hash.")
    ui.call("The prompt is short on purpose. Most of the agent's reliability comes from the tools: a describe step "
            "before any query, errors returned to the model instead of raised, and a final-answer tool that forces a "
            "status, a number and the SQL behind it, which is what makes grading possible.")

# ===========================================================================
elif stage == "Guardrails":
    ui.stage_head(2, "Guardrails", "What stops it from doing damage?",
                  "Asking nicely is not a control. Writes are blocked three times over, outside the model: the tool "
                  "accepts one statement, only SELECT or WITH, and the database connection itself is read-only. "
                  "Personal data is handled by instruction and then checked in every answer by the eval harness.",
                  "advisory")
    tries = ["DELETE FROM weekly_sales WHERE store = 'S12'", "SELECT 1; DROP TABLE stores",
             "WITH x AS (SELECT 1) UPDATE categories SET unit_price = 3.5", "SELECT store, region FROM stores LIMIT 3"]
    rows = []
    for s in tries:
        try:
            df = W.run_sql(con, s)
            res = f"allowed · {len(df)} rows"
        except W.Refused as e:
            res = f"blocked · {e}"
        rows.append({"Query sent by the agent": s, "What the tool did (run live on this page)": res})
    try:
        con.execute("DELETE FROM stores")
        conn_res = "write succeeded"
    except Exception as e:
        conn_res = f"refused by the connection: {e}"
    t.tiles([
        {"label": "Write paths", "value": "0", "accent": True, "note": "statement check, keyword check, read-only connection"},
        {"label": "Connection test", "value": "Read-only", "note": conn_res[:48]},
        {"label": "Personal-data fields", "value": "3", "note": "name, email, phone in store_contacts"},
        {"label": "Refusal cases in the evals", "value": f"{sum(c.kind == 'decline' for c in E.CASES)}", "note": "out of scope, writes, personal data"},
    ])
    U.table(rows)
    if models:
        m = models[0]
        dec = [g for g in graded(m) if g["kind"] == "decline"]
        st.markdown(f"##### How {R.label(m)} handled the requests it should refuse")
        U.table([{"Case": g["id"], "Request": g["case"].question, "Status": g["status"] or "none",
                  "Write attempts": g["writes_attempted"], "Personal data leaked": "yes" if g["pii_leak"] else "no",
                  "Result": "pass" if g["pass"] else "FAIL"} for g in dec])
    else:
        U.pending("How the recorded model handled the ten requests it should refuse")
    ui.call("The database would refuse a write even if every other check failed, so a jailbreak can at worst waste a "
            "turn. Personal data is the weaker control because the data is readable; that is why every recorded "
            "answer is scanned for the managers' names, emails and phone numbers before a model can pass.")

# ===========================================================================
elif stage == "Replay":
    ui.stage_head(3, "Replay", "What did the agent actually do?",
                  "Pick a question and step through the recorded run: what the model said, each tool call and its "
                  "result, and the final answer, graded against a reference query run live on this page.", "build")
    if not models:
        U.pending("A step-by-step replay of every recorded question")
    else:
        c1, c2 = st.columns([1, 2.4])
        m = c1.selectbox("Model", models, format_func=R.label, key="aa_model")
        recorded = [c for c in E.CASES if c.id in data["analyst"][m]]
        default = next((i for i, c in enumerate(recorded) if c.id == "M2"), 0)
        case = c2.selectbox("Question", recorded, index=default, format_func=lambda c: f"{c.id} · {c.question}", key="aa_case")
        tr = data["analyst"][m][case.id][0]
        g = E.grade(case, tr)
        steps = tr.get("steps", [])
        t.tiles([
            {"label": "Result", "value": "Pass" if g["pass"] else "Fail", "accent": g["pass"], "note": g["why"]},
            {"label": "Tool calls", "value": f"{g['tool_calls']}", "note": f"{g['sql_errors']} SQL error(s) corrected" if g["sql_errors"] else "no SQL errors"},
            {"label": "Time", "value": f"{tr.get('seconds', 0):.1f}s", "note": f"{tr.get('turns', 0)} model turns"},
            {"label": "Tokens", "value": f"{sum(tr.get('usage', {}).values()):,}", "note": "in + out, whole run"},
        ])
        n = st.slider("Step", 1, max(1, len(steps)), max(1, len(steps)), key=f"aa_step_{m}_{case.id}") if len(steps) > 1 else len(steps)
        last_rows = None
        for s in steps[:n]:
            if s["kind"] == "text":
                st.markdown(f'<div class="lab-card"><h4>Model</h4><p>{t.esc(s["text"])}</p></div>', unsafe_allow_html=True)
                continue
            res = s.get("result") or {}
            if s["tool"] == "final_answer":
                f = s["input"]
                st.markdown(f'<div class="cp-call"><span class="e">Final answer · {t.esc(f.get("status", ""))}</span>{t.esc(f.get("answer", ""))}</div>', unsafe_allow_html=True)
                ch = f.get("chart") or {}
                if ch and last_rows is not None and ch.get("x") in last_rows and ch.get("y") in last_rows:
                    fig = go.Figure()
                    if ch.get("type") == "line":
                        fig.add_scatter(x=last_rows[ch["x"]], y=last_rows[ch["y"]], line=dict(color=t.S1, width=2))
                    else:
                        fig.add_bar(x=last_rows[ch["x"]], y=last_rows[ch["y"]], marker_color=t.S1)
                    t.chart(fig, 280)
                continue
            with st.expander(f"Tool · {s['tool']}" + (" · error" if "error" in res else ""), expanded=(s["tool"] == "run_sql")):
                if s["tool"] == "run_sql":
                    st.code(s["input"].get("sql", ""), language="sql", wrap_lines=True)
                elif s["input"]:
                    st.code(json.dumps(s["input"]), language="json")
                if "error" in res:
                    st.error(res["error"])
                elif "rows" in res and res["rows"]:
                    last_rows = pd.DataFrame(res["rows"])
                    U.table(last_rows.head(12))
                elif "columns" in res and "sample" in res:
                    U.table(pd.DataFrame(res["columns"]))
        if case.kind != "decline":
            st.markdown("##### Reference, run live")
            st.code(case.sql, language="sql", wrap_lines=True)
            st.caption(f"Expected: {E.expected(case, con)}")
    ui.call("A recorded run is evidence; a live demo is a performance. Replaying the exact steps lets anyone check "
            "how an answer was reached, including the queries that failed and how the model recovered.")

# ===========================================================================
elif stage == "Cost & speed":
    ui.stage_head(4, "Cost & speed", "What does an answer cost, and how long does it take?",
                  "Token counts and wall-clock time come straight from the recordings. Dollars use list prices at the "
                  "time of building; they are shown so the trade-off is visible, not as a quote.", "advisory")
    if not models:
        U.pending("Tokens, seconds and dollars for every recorded run")
    else:
        rows = []
        for m in models:
            gs = graded(m)
            tok_in = sum(g["trace"]["usage"]["input_tokens"] for g in gs)
            tok_out = sum(g["trace"]["usage"]["output_tokens"] for g in gs)
            dollars = R.cost(m, {"input_tokens": tok_in, "output_tokens": tok_out})
            k = sum(g["pass"] for g in gs)
            rows.append({"Model": R.label(m), "Runs": len(gs), "Passed": k, "Median seconds": float(pd.Series([g["trace"].get("seconds", 0) for g in gs]).median()),
                         "Tokens per run": (tok_in + tok_out) / max(1, len(gs)), "Cost per run": (dollars / len(gs)) if dollars is not None and gs else None,
                         "Cost per correct answer": (dollars / k) if dollars is not None and k else None})
        df = pd.DataFrame(rows)
        best = df.sort_values("Passed", ascending=False).iloc[0]
        t.tiles([
            {"label": "Most accurate", "value": best["Model"], "accent": True, "note": f"{int(best['Passed'])} of {int(best['Runs'])} passed"},
            {"label": "Median time", "value": f"{best['Median seconds']:.1f}s", "note": "per question, that model"},
            {"label": "Tokens per question", "value": f"{best['Tokens per run']:,.0f}", "note": "schema lookups dominate"},
            {"label": "Cost per correct answer", "value": t.money(best["Cost per correct answer"], 3) if pd.notna(best["Cost per correct answer"]) else "—", "note": "list price"},
        ])
        show = df.copy()
        for c in ("Cost per run", "Cost per correct answer"):
            show[c] = show[c].map(lambda v: t.money(v, 4) if pd.notna(v) else "—")
        show["Median seconds"] = show["Median seconds"].map("{:.1f}".format)
        show["Tokens per run"] = show["Tokens per run"].map("{:,.0f}".format)
        U.table(show)
        fig = go.Figure()
        for i, m in enumerate(models):
            gs = graded(m)
            fig.add_scatter(x=[g["trace"].get("seconds", 0) for g in gs], y=[sum(g["trace"]["usage"].values()) for g in gs], mode="markers",
                            name=R.label(m), text=[g["id"] for g in gs], marker=dict(size=9, color=[t.S1, t.S3, t.S2][i % 3],
                            symbol=["circle" if g["pass"] else "x" for g in gs]))
        fig.update_layout(xaxis_title="Seconds per question", yaxis_title="Tokens per question", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 340)
        st.caption("Circles passed, crosses failed. The full comparison, with intervals and a release gate, is in the LLM eval harness.")
    ui.call("The cheaper model is the default only if it clears the same gate. Cost per correct answer, not cost per "
            "call, is the number that decides it.")

st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.aa_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

U.provenance("analyst")
t.footnote("Larkspur Market is fictional and its data synthetic. Model runs are real API calls recorded once and "
           "replayed; reference answers and grades are computed on this page.")
