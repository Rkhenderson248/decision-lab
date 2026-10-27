"""Shared bits for the three AI engineering pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lab import theme as t
from lab.ai import replay as R


def pending(what: str) -> None:
    st.markdown(
        f'<div class="cp-call"><span class="e">Not recorded yet</span>{t.esc(what)} appears here once real model runs '
        f'have been recorded. They are recorded once, with the owner\'s API key, by <code>{R.COMMAND}</code>, and '
        'replayed from then on: no model is called when this page is viewed, and nothing on it is simulated.</div>',
        unsafe_allow_html=True,
    )


def provenance(kind: str) -> None:
    man = R.load()["manifest"]
    if not man:
        return
    models = ", ".join(R.label(m) for m in man.get("models", []))
    h = man.get("analyst_prompt_hash" if kind == "analyst" else "docs_prompt_hash", "")
    t.footnote(f"Recorded {man.get('recorded_at', '')[:10]} with {man.get('sdk', '')} · models: {models} · prompt {h}. "
               "Replayed exactly as recorded; grading is rule-based and runs on this page.")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return max(0.0, c - h), min(1.0, c + h)


def table(rows: list[dict] | pd.DataFrame, height: int | None = None) -> None:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    kw = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, width="stretch", **kw)
