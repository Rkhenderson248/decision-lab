"""Document intelligence: forty synthetic supplier invoices, extracted by a rules baseline (run live) and by a model
(recorded once, replayed), then put through deterministic controls that decide what can be paid without a person."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.ai import docs as D
from lab.ai import replay as R
from lab.ai import ui as U
from lab.cu import ui

ui.css()
STAGES = ["Inbox", "Extraction", "Controls", "Review queue", "Economics"]
ISSUE_NAMES = {"mismatch": "total does not add up", "missing_po": "no purchase order", "due_before_issue": "due before issued", "duplicate": "duplicate"}

t.header(
    "AI engineering · Larkspur Market accounts payable (fictional)",
    "Document intelligence: invoices in, payable records out, people only where needed",
    "Forty supplier invoices in four layouts, from a clean form to a forwarded email to a faxed copy with scanning "
    "errors, and ten planted problems. Ten fields are extracted, checked by rules an accounts-payable team already "
    "trusts, and either paid straight through or sent to a person with the reason.",
)


@st.cache_resource
def docs() -> list:
    return D.generate()


DOCS = docs()
data = R.load()
models = R.doc_models()


@st.cache_data(show_spinner=False)
def rules_eval() -> dict:
    return D.evaluate(DOCS, {d.id: D.rules_extract(d.text) for d in DOCS})


def model_ext(m: str) -> dict:
    return {k: v.get("extraction") for k, v in data["docs"][m]["docs"].items() if v.get("extraction") is not None}


def model_eval(m: str) -> dict:
    return D.evaluate(DOCS, model_ext(m))


RULES = rules_eval()
EV = {m: model_eval(m) for m in models}
LEAD = max(models, key=lambda m: EV[m]["field_accuracy"]) if models else None

q = str(st.query_params.get("stage", "")).lower().replace("-", "").replace(" ", "")
lookup = {s.lower().replace(" ", ""): s for s in STAGES}
lookup.update({"review": "Review queue", "extract": "Extraction"})
if "di_stage" not in st.session_state:
    st.session_state.di_stage = lookup.get(q, STAGES[0])
    st.session_state.di_stage__last = st.session_state.di_stage
stage = w.segmented("Stage", STAGES, key="di_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace(" ", "-")
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")


def paper(text: str) -> None:
    st.html(f'<pre style="background:#F3EFE4;color:#1d2321;padding:18px 20px;margin:0;border-radius:2px;font:12.5px/1.5 ui-monospace,'
            f'Menlo,Consolas,monospace;white-space:pre-wrap;box-shadow:0 6px 24px rgba(0,0,0,.35)">{t.esc(text)}</pre>')


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


# ===========================================================================
if stage == "Inbox":
    ui.stage_head(1, "Inbox", "What arrives in accounts payable?",
                  "Suppliers do not share a template. The same ten facts arrive as a form, a statement, a sentence in "
                  "an email or a fax where a 0 reads as an O. Some invoices are wrong on their own terms, and two are "
                  "copies of earlier ones.", "diagnostic")
    t.tiles([
        {"label": "Invoices", "value": f"{len(DOCS)}", "note": "8 fictional suppliers"},
        {"label": "Layouts", "value": "4", "note": "form, statement, email, fax"},
        {"label": "Planted problems", "value": f"{sum(bool(d.issues) for d in DOCS)}", "accent": True, "note": "mismatch, no PO, dates, duplicates"},
        {"label": "Fields per invoice", "value": f"{len(D.FIELDS)}", "note": "vendor to line count"},
    ])
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        pick = st.selectbox("Invoice", DOCS, index=3, format_func=lambda d: f"{d.id} · {d.layout}" + (f" · {', '.join(ISSUE_NAMES[i] for i in d.issues)}" if d.issues else ""), key="di_doc")
        paper(pick.text)
    with b:
        st.markdown("##### What it really says")
        U.table([{"Field": k, "Value": fmt(v)} for k, v in D.truth(pick).items()])
        if pick.issues:
            ui.cards([("Planted problem", ", ".join(ISSUE_NAMES[i] for i in pick.issues))])
    ui.call("The ground truth is the document as written, not the document as it should be. An extractor that "
            "quietly fixes a wrong total hides the very problem the controls exist to catch.")

# ===========================================================================
elif stage == "Extraction":
    ui.stage_head(2, "Extraction", "How many fields come out right?",
                  "The rules baseline is what a team writes before it has a model: regular expressions per layout, run "
                  "live on this page. The model gets the text, a short instruction and a JSON schema; its outputs were "
                  "recorded once and are graded here field by field.", "build")
    t.tiles([
        {"label": "Rules · field accuracy", "value": f"{RULES['field_accuracy']:.0%}", "note": f"{RULES['doc_accuracy']:.0%} of invoices fully right"},
        {"label": f"{R.label(LEAD)} · field accuracy" if LEAD else "Model · field accuracy", "value": f"{EV[LEAD]['field_accuracy']:.0%}" if LEAD else "—", "accent": bool(LEAD),
         "note": f"{EV[LEAD]['doc_accuracy']:.0%} of invoices fully right" if LEAD else "not recorded yet"},
        {"label": "Rules · email layout", "value": f"{RULES['by_layout']['email']:.0%}", "note": "prose defeats patterns"},
        {"label": "Rules · fax layout", "value": f"{RULES['by_layout']['fax']:.0%}", "note": "scanning errors"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        fig.add_bar(x=D.LAYOUTS, y=[RULES["by_layout"][k] for k in D.LAYOUTS], name="Rules", marker_color=t.S2)
        for i, m in enumerate(models):
            fig.add_bar(x=D.LAYOUTS, y=[EV[m]["by_layout"][k] for k in D.LAYOUTS], name=R.label(m), marker_color=[t.S1, t.S3][i % 2])
        fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%", range=[0, 1.05]), title=dict(text="Field accuracy by layout", font=dict(size=13)), legend=dict(orientation="h", y=1.15))
        t.chart(fig, 330)
    with b:
        fig = go.Figure()
        fig.add_bar(y=D.FIELDS, x=[RULES["by_field"][k] for k in D.FIELDS], name="Rules", orientation="h", marker_color=t.S2)
        for i, m in enumerate(models):
            fig.add_bar(y=D.FIELDS, x=[EV[m]["by_field"][k] for k in D.FIELDS], name=R.label(m), orientation="h", marker_color=[t.S1, t.S3][i % 2])
        fig.update_layout(barmode="group", xaxis=dict(tickformat=".0%", range=[0, 1.05]), yaxis=dict(autorange="reversed"), title=dict(text="Field accuracy by field", font=dict(size=13)), showlegend=False)
        t.chart(fig, 330)
    if not models:
        U.pending("The model's extraction of every invoice")
    else:
        m = st.selectbox("Compare on one invoice", models, format_func=R.label, key="di_m")
        d = st.selectbox("Invoice", DOCS, index=2, format_func=lambda d: f"{d.id} · {d.layout}", key="di_doc2")
        ext = (data["docs"][m]["docs"].get(d.id) or {}).get("extraction") or {}
        rx = D.rules_extract(d.text)
        tr = D.truth(d)
        U.table([{"Field": k, "Truth": fmt(tr[k]), "Rules": fmt(rx.get(k)) + ("" if D.field_ok(k, rx.get(k), tr[k]) else "  ✗"),
                  R.label(m): fmt(ext.get(k)) + ("" if D.field_ok(k, ext.get(k), tr[k]) else "  ✗")} for k in D.FIELDS])
        with st.expander("The extraction prompt and schema"):
            st.code(D.EXTRACT_SYSTEM, language="markdown", wrap_lines=True)
            st.caption(f"Prompt and schema hash {D.PROMPT_HASH}.")
    ui.call("Rules are perfect on the layouts they were written for and useless on the next supplier. That, not "
            "average accuracy, is the case for a model: the long tail of formats nobody wrote a pattern for.")

# ===========================================================================
elif stage == "Controls":
    ui.stage_head(3, "Controls", "Which invoices can be paid without a person?",
                  "Five deterministic checks run on whatever was extracted: required fields present, a purchase order, "
                  "subtotal plus tax equals total, due date after issue date, and no earlier invoice with the same "
                  "vendor and number. Anything flagged goes to review with the reason.", "advisory")
    opts = ["Rules"] + models
    rows = []
    for o in opts:
        e = RULES if o == "Rules" else EV[o]
        rows.append({"Extractor": "Rules baseline" if o == "Rules" else R.label(o), "Straight through": f"{e['straight_through']:.0%}",
                     "Planted problems caught": f"{e['issue_recall']:.0%}", "Clean invoices held": e["false_holds"], "Fully correct": f"{e['doc_accuracy']:.0%}"})
    best = EV[LEAD] if LEAD else None
    t.tiles([
        {"label": "Planted problems caught", "value": f"{best['issue_recall']:.0%}" if best else f"{RULES['issue_recall']:.0%}", "accent": True, "note": R.label(LEAD) if LEAD else "rules baseline"},
        {"label": "Clean invoices held", "value": f"{best['false_holds']}" if best else f"{RULES['false_holds']}", "note": "work a person did not need to do"},
        {"label": "Straight through", "value": f"{best['straight_through']:.0%}" if best else f"{RULES['straight_through']:.0%}", "note": "best possible: 75% (30 of 40 are clean)"},
        {"label": "Rules baseline holds", "value": f"{RULES['false_holds']} clean", "note": "it cannot read them, so it holds them"},
    ])
    U.table(rows)
    if not models:
        U.pending("The model's routing through the same controls")
    ui.call("The controls do the deciding, not the model. A model that extracts well makes the controls quiet on clean "
            "invoices; it never gets to wave a wrong total through, because the arithmetic is checked outside it.")

# ===========================================================================
elif stage == "Review queue":
    ui.stage_head(4, "Review queue", "What does the reviewer see?",
                  "Each held invoice arrives with the reason and the fields that triggered it, so the reviewer checks "
                  "one thing instead of re-keying the document.", "build")
    opts = (["Rules"] + models) if models else ["Rules"]
    o = st.selectbox("Extractor", opts, index=len(opts) - 1 if models else 0, format_func=lambda x: "Rules baseline" if x == "Rules" else R.label(x), key="di_q")
    e = RULES if o == "Rules" else EV[o]
    held = [r for r in e["rows"] if r["held"]]
    t.tiles([
        {"label": "In the queue", "value": f"{len(held)}", "note": f"of {len(DOCS)} invoices"},
        {"label": "Real problems", "value": f"{sum(r['planted'] for r in held)}", "accent": True, "note": "planted issues found"},
        {"label": "Clean but held", "value": f"{sum(not r['planted'] for r in held)}", "note": "extraction gaps"},
        {"label": "Missed problems", "value": f"{sum(r['planted'] and not r['held'] for r in e['rows'])}", "note": "paid without review"},
    ])
    U.table([{"Invoice": r["id"], "Layout": r["layout"], "Flags": "; ".join(r["flags"]),
              "Planted": ", ".join(ISSUE_NAMES[i] for i in r["issues"]) or "none", "Real problem": "yes" if r["planted"] else "no"} for r in held], height=420)
    if not models:
        U.pending("The model's review queue")
    ui.call("A short queue of real problems is the goal. A long queue of clean invoices the extractor could not read "
            "is the rules baseline's cost, paid in reviewer hours.")

# ===========================================================================
elif stage == "Economics":
    ui.stage_head(5, "Economics", "What is it worth per thousand invoices?",
                  "Reviewer time, model cost and the cost of paying a bad invoice, per thousand invoices like these. "
                  "Change the assumptions; the rates come from this page's results.", "advisory")
    c1, c2, c3 = st.columns(3)
    mins = c1.slider("Minutes to key or review one invoice", 1.0, 15.0, 6.0, .5, key="di_min")
    rate = c2.slider("Loaded cost per hour, $", 20, 120, 55, 5, key="di_rate")
    miss = c3.slider("Average cost of a problem paid unreviewed, $", 50, 2000, 400, 50, key="di_miss")
    per = rate / 60 * mins
    planted_share = sum(bool(d.issues) for d in DOCS) / len(DOCS)
    rows = [{"Process": "Manual keying", "Reviewed per 1,000": 1000, "Missed problems per 1,000": 0.0, "Model $ per 1,000": 0.0}]
    for o in ["Rules"] + models:
        e = RULES if o == "Rules" else EV[o]
        held = sum(r["held"] for r in e["rows"]) / len(DOCS)
        missed = sum(r["planted"] and not r["held"] for r in e["rows"]) / len(DOCS)
        mc = 0.0
        if o != "Rules":
            u = [v.get("usage", {}) for v in data["docs"][o]["docs"].values()]
            c = R.cost(o, {"input_tokens": sum(x.get("input_tokens", 0) for x in u), "output_tokens": sum(x.get("output_tokens", 0) for x in u)})
            mc = (c or 0) / max(1, len(u)) * 1000
        rows.append({"Process": "Rules + controls" if o == "Rules" else f"{R.label(o)} + controls", "Reviewed per 1,000": held * 1000,
                     "Missed problems per 1,000": missed * 1000, "Model $ per 1,000": mc})
    df = pd.DataFrame(rows)
    df["Total $ per 1,000"] = df["Reviewed per 1,000"] * per + df["Missed problems per 1,000"] * miss + df["Model $ per 1,000"]
    best = df.iloc[1:].sort_values("Total $ per 1,000").iloc[0]
    t.tiles([
        {"label": "Manual", "value": t.money(df.iloc[0]["Total $ per 1,000"]), "note": "per 1,000 invoices"},
        {"label": "Best automated", "value": t.money(best["Total $ per 1,000"]), "accent": True, "note": best["Process"]},
        {"label": "Saving", "value": t.money(df.iloc[0]["Total $ per 1,000"] - best["Total $ per 1,000"]), "note": "per 1,000 invoices"},
        {"label": "Problem rate", "value": f"{planted_share:.0%}", "note": "in this inbox; set by design"},
    ])
    show = df.copy()
    for c in ("Reviewed per 1,000", "Missed problems per 1,000"):
        show[c] = show[c].map("{:,.0f}".format)
    for c in ("Model $ per 1,000", "Total $ per 1,000"):
        show[c] = show[c].map(lambda v: t.money(v, 2 if v < 100 else 0))
    U.table(show)
    if not models:
        U.pending("The model's line in this table")
    ui.call("Model cost is a rounding error next to reviewer time. The number that moves the answer is how many clean "
            "invoices the extractor can read well enough for the controls to stay quiet.")

st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.di_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

U.provenance("docs")
t.footnote("Larkspur Market and its suppliers are fictional and every invoice is synthetic. Model extractions are real "
           "API calls recorded once and replayed; the rules baseline, grading and controls run on this page.")
