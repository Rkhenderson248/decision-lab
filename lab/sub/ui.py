"""Layout helpers for the subscriber lab (own module, see lab/widgets.py for why)."""

from __future__ import annotations

import streamlit as st

from lab import theme as t
from lab.sub import models as M


def chip(segment: str) -> str:
    c = M.SEG_COLORS.get(segment, t.BASE)
    return f'<span class="cu-chip"><i style="background:{c}"></i>{t.esc(segment)}</span>'


def note(html: str) -> None:
    st.markdown(f'<div class="cu-member-note"><span class="e">Following this subscriber</span>{html}</div>', unsafe_allow_html=True)


def brief_html(items: dict) -> str:
    """A self-contained one-page executive brief (downloadable)."""
    rows = "".join(f"<tr><td>{t.esc(k)}</td><td>{v}</td></tr>" for k, v in items["numbers"])
    recs = "".join(f"<li><b>{t.esc(h)}</b> {b}</li>" for h, b in items["recommendations"])
    risks = "".join(f"<li>{r}</li>" for r in items["risks"])
    asks = "".join(f"<li>{a}</li>" for a in items["asks"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{t.esc(items['title'])}</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;color:#0E1311;max-width:760px;margin:40px auto;padding:0 24px;line-height:1.5}}
h1{{font-family:Georgia,serif;color:#0F4640;font-weight:400;font-size:30px;margin:.2em 0}}h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#0F4640;margin:28px 0 8px}}
.e{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#6B756E}}table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #E1E4DE;padding:7px 4px}}
td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}li{{margin:6px 0}}.lede{{font-size:17px}}footer{{margin-top:36px;font-size:11px;color:#6B756E}}</style></head>
<body><div class="e">{t.esc(items['eyebrow'])}</div><h1>{t.esc(items['title'])}</h1><p class="lede">{items['lede']}</p>
<h2>Recommendations</h2><ol>{recs}</ol><h2>The numbers</h2><table>{rows}</table>
<h2>Risks and guardrails</h2><ul>{risks}</ul><h2>Decisions needed</h2><ul>{asks}</ul>
<footer>{t.esc(items['footer'])}</footer></body></html>"""
