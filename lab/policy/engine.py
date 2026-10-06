"""Retrieval, grounded answering and evaluation for the policy assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from lab.policy.manual import GOLDEN, SECTIONS

SYNONYMS = {
    "heloc": "heloc home equity line", "dti": "dti debt-to-income", "ltv": "loan-to-value ltv",
    "cltv": "combined loan-to-value cltv", "bankrupt": "bankruptcy", "bk": "bankruptcy",
    "ot": "overtime", "car": "auto vehicle", "cars": "auto vehicle", "fico": "credit score",
    "decline": "declined adverse action", "denied": "declined adverse action", "late": "delinquency",
    "military": "servicemembers active-duty", "job": "hardship job loss",
}


def _expand(q: str) -> str:
    words = re.findall(r"[a-z0-9\-']+", q.lower())
    return " ".join(SYNONYMS.get(w, w) for w in words)


@dataclass
class Index:
    vec: TfidfVectorizer
    mat: object
    ids: list
    titles: list
    texts: list


@st.cache_resource(show_spinner=False)
def index() -> Index:
    ids = [s[0] for s in SECTIONS]
    titles = [s[1] for s in SECTIONS]
    texts = [s[2] for s in SECTIONS]
    docs = [f"{t} {t} {x}" for t, x in zip(titles, texts)]  # titles count double
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    mat = vec.fit_transform([_expand(d) for d in docs])
    return Index(vec, mat, ids, titles, texts)


def retrieve(question: str, k: int = 3) -> pd.DataFrame:
    ix = index()
    qv = ix.vec.transform([_expand(question)])
    sims = (ix.mat @ qv.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    return pd.DataFrame({"id": [ix.ids[i] for i in order], "title": [ix.titles[i] for i in order],
                         "text": [ix.texts[i] for i in order], "score": sims[order]})


STOP = {"the", "a", "an", "of", "for", "to", "is", "what", "how", "can", "do", "we", "on", "in", "and", "or", "i", "my",
        "does", "be", "are", "it", "with", "who", "when", "much", "many", "get", "someone", "our", "there", "which", "use"}


def coverage(question: str, text: str) -> int:
    """How many distinct content words of the question appear in a section (a cheap guard against one-word matches)."""
    q = {w for w in _expand(question).split() if w not in STOP and len(w) > 2}
    body = _expand(text)
    return sum(1 for w in q if w in body)


def should_answer(question: str, hits: pd.DataFrame, threshold: float) -> bool:
    top = hits.iloc[0]
    return float(top["score"]) >= threshold and coverage(question, f"{top['title']} {top['text']}") >= 2


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", text) if s.strip()]


def extractive_answer(question: str, hits: pd.DataFrame, n: int = 2) -> list[tuple[str, str]]:
    """The sentences from the retrieved sections that best match the question, each with its citation."""
    q_terms = set(_expand(question).split()) - {"the", "a", "an", "of", "for", "to", "is", "what", "how", "can", "do", "we", "on"}
    scored = []
    for rank, row in enumerate(hits.itertuples()):
        for s in _sentences(row.text):
            terms = set(_expand(s).split())
            overlap = len(q_terms & terms)
            scored.append((overlap + 0.6 * row.score * 10 - 0.4 * rank, s, row.id))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, s, sid in scored:
        if s not in seen:
            out.append((s, sid))
            seen.add(s)
        if len(out) == n:
            break
    return out


SYSTEM = (
    "You answer questions about Kestrel Valley Credit Union's consumer lending policy. Use ONLY the policy "
    "sections provided. Cite every claim with its section id in square brackets, e.g. [UW-2.1]. If the sections "
    "do not answer the question, reply exactly: \"The policy manual doesn't cover this. Please ask Credit "
    "Policy.\" Be concise: two to four sentences. Never invent numbers."
)


def model_answer(question: str, hits: pd.DataFrame) -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        model = st.secrets.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    except Exception:  # noqa: BLE001 - no secrets file at all
        key, model = None, None
    if not key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        sources = "\n\n".join(f"[{r.id}] {r.title}: {r.text}" for r in hits.itertuples())
        msg = client.messages.create(
            model=model, max_tokens=350, system=SYSTEM,
            messages=[{"role": "user", "content": f"Policy sections:\n{sources}\n\nQuestion: {question}"}])
        return msg.content[0].text.strip()
    except Exception:  # noqa: BLE001 - fall back to extractive mode on any API problem
        return None


def cited_ids(answer: str) -> list[str]:
    return re.findall(r"\[([A-Z]{2,3}-\d\.\d)\]", answer)


@st.cache_data(show_spinner=False)
def evaluate(k: int = 3) -> pd.DataFrame:
    rows = []
    for q, gold in GOLDEN:
        hits = retrieve(q, k=k)
        ids = list(hits["id"])
        top = hits.iloc[0]
        rows.append({"question": q, "expected": gold or "Not covered", "top": ids[0], "top_score": float(hits["score"].iloc[0]),
                     "coverage": coverage(q, f"{top['title']} {top['text']}"),
                     "retrieved": ", ".join(ids), "rank": (ids.index(gold) + 1) if gold in ids else None, "in_scope": gold is not None})
    return pd.DataFrame(rows)


def threshold_curve(ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for th in np.linspace(0.0, 0.4, 41):
        answered = (ev["top_score"] >= th) & (ev["coverage"] >= 2)
        ins, oos = ev["in_scope"], ~ev["in_scope"]
        out.append({"threshold": th,
                    "in_scope_answered_correctly": ((answered & ins) & ev["rank"].eq(1)).sum() / ins.sum(),
                    "out_of_scope_refused": (~answered & oos).sum() / max(oos.sum(), 1)})
    return pd.DataFrame(out)
