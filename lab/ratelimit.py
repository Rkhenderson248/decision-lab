"""Spend guard for the optional Claude calls.

Limits per visitor session (a burst and a total) and across the whole app per day, so a bot or a stuck
loop cannot run up the API bill. Counts live in memory (they reset when the app restarts, which is fine
for a cap). Limits can be changed in Streamlit secrets: LLM_SESSION_MAX, LLM_PER_MINUTE, LLM_DAILY_MAX.
When a limit is hit the callers fall back to the rule-based answers.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import streamlit as st


class RateLimited(Exception):
    """Raised by callers that cannot fall back silently."""


@st.cache_resource
def _global() -> dict:
    return {"day": "", "n": 0}


def _cfg(name: str, default: int) -> int:
    try:
        return int(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001 - no secrets file
        return default


def allow_llm() -> bool:
    """True if one more Claude call is allowed right now (and counts it)."""
    now = time.time()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    g = _global()
    if g["day"] != today:
        g["day"], g["n"] = today, 0
    if g["n"] >= _cfg("LLM_DAILY_MAX", 300):
        return False
    calls = [t for t in st.session_state.get("_llm_calls", []) if now - t < 3600]
    if len(calls) >= _cfg("LLM_SESSION_MAX", 20):
        return False
    if sum(1 for t in calls if now - t < 60) >= _cfg("LLM_PER_MINUTE", 4):
        return False
    calls.append(now)
    st.session_state["_llm_calls"] = calls
    g["n"] += 1
    return True
