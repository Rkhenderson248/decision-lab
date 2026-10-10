"""Credit policy assistant: grounded answers with citations, an honest refusal, and the evaluation behind it."""

import html

import plotly.graph_objects as go
import streamlit as st

from lab import theme as t
from lab.policy import engine as E
from lab.policy.manual import SECTIONS

THRESHOLD = 0.125


def cards(items):
    body = "".join(f'<div class="lab-card"><h4>{html.escape(k)}</h4><p>{html.escape(v)}</p></div>' for k, v in items)
    st.markdown(f'<div class="cp-grid">{body}</div>', unsafe_allow_html=True)


t.header(
    "AI project · Kestrel Valley Credit Union (fictional)",
    "Ask the credit policy, get an answer you can check",
    "A retrieval-grounded assistant over a 25-section consumer lending policy manual. Every answer cites the "
    "section it came from, anything the manual does not cover is declined rather than guessed, and an "
    "evaluation suite measures both before any change ships.",
)

ask, evaluation, built = st.tabs(["Ask the policy", "Evaluation", "How it's built"])

EXAMPLES = {
    "Exception authority": "Who can approve an exception on a $40,000 loan?",
    "Bonus and overtime income": "How do we count bonus and overtime income?",
    "HELOC on a rental": "Can a HELOC be opened on an investment property?",
    "Fraud queue rules": "When does an application go to the fraud queue?",
    "Online banking (not covered)": "How do I reset my online banking password?",
}

with ask:
    st.markdown("**Try a question**")
    cols = st.columns(len(EXAMPLES))
    for c, (label, ex) in zip(cols, EXAMPLES.items()):
        if c.button(label, key=f"ex_{label}", width="stretch"):
            st.session_state["pa_q"] = ex
    q = st.text_input("Your question", key="pa_q", placeholder="e.g. What is the maximum term on a used car loan?")
    if q:
        hits = E.retrieve(q, k=3)
        top = float(hits["score"].iloc[0])
        ok = E.should_answer(q, hits, THRESHOLD)
        answer = E.model_answer(q, hits) if ok else None
        mode = "model" if answer else "extractive"
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            if not ok:
                st.markdown(
                    '<div class="lab-reco" style="background:#2A3631;border-left-color:#EF936F"><div class="e">Declined</div>'
                    '<div class="t">The policy manual doesn\'t cover this.</div>'
                    f'<div class="d">The closest section matched with a score of {top:.2f} and shares too little with the '
                    'question to answer it safely (see Evaluation). A guess here would sound confident and could be wrong, '
                    'so the question is routed to Credit Policy instead.</div></div>', unsafe_allow_html=True)
            elif answer:
                body = html.escape(answer)
                for sid in set(E.cited_ids(answer)):
                    body = body.replace(f"[{sid}]", f'<span class="lab-pill">{sid}</span>')
                st.markdown(f'<div class="lab-reco"><div class="e">Answer · cited</div><div class="d" style="color:#F0F7ED;font-size:1.02rem">{body}</div></div>',
                            unsafe_allow_html=True)
            else:
                parts = E.extractive_answer(q, hits)
                body = " ".join(f'{html.escape(s)} <span class="lab-pill">{sid}</span>' for s, sid in parts)
                st.markdown(f'<div class="lab-reco"><div class="e">Answer · quoted from the policy</div>'
                            f'<div class="d" style="color:#F0F7ED;font-size:1.02rem">{body}</div></div>', unsafe_allow_html=True)
            st.caption("Model mode: Claude writes the answer from the retrieved sections and must cite them." if mode == "model"
                       else "Quoted mode: the answer is the policy's own sentences, so it cannot drift from the source. "
                            "Add an API key to switch on written answers.")
        with right:
            st.markdown("**Sources retrieved**")
            for r in hits.itertuples():
                with st.expander(f"{r.id} · {r.title}  —  match {r.score:.2f}", expanded=(r.Index == 0 and ok)):
                    st.markdown(r.text.replace("$", "\\$"))

with evaluation:
    ev = E.evaluate()
    ins = ev[ev["in_scope"]]
    curve = E.threshold_curve(ev)
    th = st.slider("Refusal threshold (minimum match score to answer)", 0.0, 0.4, THRESHOLD, 0.005, format="%.3f", key="pa_th")
    answered = (ev["top_score"] >= th) & (ev["coverage"] >= 2)
    correct = (answered & ev["in_scope"] & ev["rank"].eq(1)).sum() / ev["in_scope"].sum()
    refused = (~answered & ~ev["in_scope"]).sum() / (~ev["in_scope"]).sum()
    t.tiles([
        {"label": "Right section ranked first", "value": t.pct((ins["rank"] == 1).mean()), "accent": True, "note": f"{len(ins)} in-scope test questions"},
        {"label": "Right section in the top three", "value": t.pct(ins["rank"].notna().mean()), "note": "what the model actually reads"},
        {"label": "In-scope answered correctly", "value": t.pct(correct), "accent": True, "note": f"at threshold {th:.3f}"},
        {"label": "Out-of-scope declined", "value": t.pct(refused), "note": f"{(~ev['in_scope']).sum()} questions the manual can't answer"},
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["in_scope_answered_correctly"] * 100, name="In-scope answered correctly",
                                 line=dict(color=t.S1, width=2.5)))
        fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["out_of_scope_refused"] * 100, name="Out-of-scope declined",
                                 line=dict(color=t.S2, width=2.5)))
        fig.add_vline(x=th, line=dict(color=t.INK, width=1))
        fig.update_layout(title="Answer more, or refuse more: the threshold decides", xaxis_title="Refusal threshold",
                          yaxis_title="%", yaxis=dict(range=[0, 105]), legend=dict(orientation="h", y=1.02), margin=dict(t=100))
        t.chart(fig, height=360)
    with right:
        fig2 = go.Figure()
        for flag, name, color in ((True, "In scope", t.S1), (False, "Not covered", t.S2)):
            d = ev[ev["in_scope"] == flag]
            fig2.add_trace(go.Box(x=d["top_score"], name=name, marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
                                  hovertext=d["question"], hoverinfo="text+x"))
        fig2.add_vline(x=th, line=dict(color=t.INK, width=1))
        fig2.update_layout(title="Best match score by question type", xaxis_title="Top retrieval score", showlegend=False)
        t.chart(fig2, height=360)
    show = ev.copy()
    show["result"] = [
        ("Declined (correct)" if not r.in_scope else "Declined (missed)") if (r.top_score < th or r.coverage < 2)
        else ("Answered from the right section" if r.in_scope and r.rank == 1 else
              "Answered, wrong top section" if r.in_scope else "Answered (should have declined)")
        for r in show.itertuples()]
    st.dataframe(show[["question", "expected", "retrieved", "top_score", "result"]].rename(columns={
        "question": "Test question", "expected": "Correct section", "retrieved": "Retrieved (top three)", "top_score": "Match", "result": "Outcome"}),
        hide_index=True, width="stretch", column_config={"Match": st.column_config.NumberColumn(format="%.2f")})
    t.insight("No threshold is perfect: answerable and unanswerable questions overlap. Raising it declines more questions the "
              "manual can't answer but turns away real ones too. One failure no threshold fixes: the <b>dividend-rate</b> "
              "question shares enough words with the pricing sections to get through. That is why the test set exists: each "
              "miss becomes a new test case, a synonym or a scope rule before the next release.")

with built:
    cards([
        ("Source of truth", f"A {len(SECTIONS)}-section policy manual, versioned. The assistant can only quote or paraphrase it."),
        ("Retrieval", "TF-IDF over titles and text with domain synonyms (DTI, LTV, HELOC). Top three sections go to the answer step."),
        ("Answering", "Quoted mode returns the policy's own sentences. Model mode writes an answer from the same sections and must cite each claim."),
        ("Refusal", "If the best section scores too low, or shares fewer than two content words with the question, the assistant declines and routes it."),
        ("Evaluation", "A fixed set of real questions, including ones the manual can't answer, scored before every change."),
        ("In production", "Access follows existing permissions, every answer is logged with its sources, and misses feed the test set."),
    ])
    st.markdown(f"""
**Why quoted answers are the default.** For policy questions, the policy's own wording is the safest possible answer: it cannot drift, and an underwriter can check it in seconds. A written answer reads better but must be held to the same sources, so model mode requires a citation for every claim.

**What would change at scale.** Embedding-based retrieval alongside keywords, section-level access control, a larger evaluation set drawn from real questions, and a second check that every number in an answer appears in its cited section.
""")

t.footnote("Kestrel Valley Credit Union and its policy manual are fictional and illustrative; nothing here is legal, regulatory or credit-policy guidance.")
