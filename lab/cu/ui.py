"""Small layout helpers for the lending lab, kept in their own module (see lab/widgets.py for why)."""

from __future__ import annotations

import streamlit as st

from lab import theme as t
from lab.cu import models as M

# Fixed segment order and colors (adjacent pairs validated; names always shown beside the color).
SEGMENT_ORDER = list(M.SEGMENT_NAMES)
SEG_COLORS = dict(zip(SEGMENT_ORDER, ["#CBFA7C", "#EF936F", "#8FA2F0", "#E07AA0", "#D9C24A", "#5CC3EE"]))

SERVICES = {
    "diagnostic": "Decision diagnostic",
    "build": "Build · decision products",
    "advisory": "Measurement & AI advisory",
}


def stage_head(n: int, title: str, question: str, why: str, service: str) -> None:
    st.markdown(
        f'<div class="cp-head"><h2>{t.esc(title)}</h2>'
        '</div>'
        f'<p class="cu-q"><span>{n:02d}</span>{t.esc(question)}</p>'
        f'<p class="cp-why">{t.esc(why)}</p>',
        unsafe_allow_html=True,
    )


def call(text: str) -> None:
    st.markdown(f'<div class="cp-call"><span class="e">The judgment call</span>{text}</div>', unsafe_allow_html=True)


def member_note(html: str) -> None:
    st.markdown(f'<div class="cu-member-note"><span class="e">Following this member</span>{html}</div>', unsafe_allow_html=True)


def cards(items: list[tuple[str, str]]) -> None:
    body = "".join(f'<div class="lab-card"><h4>{t.esc(k)}</h4><p>{t.esc(v)}</p></div>' for k, v in items)
    st.markdown(f'<div class="cp-grid">{body}</div>', unsafe_allow_html=True)


def chip(segment: str) -> str:
    c = SEG_COLORS.get(segment, t.BASE)
    return f'<span class="cu-chip"><i style="background:{c}"></i>{t.esc(segment)}</span>'


_CSS = """
<style>
.cu-q{display:flex;gap:12px;align-items:baseline;font-family:"Space Grotesk",Inter,sans-serif;font-weight:700;letter-spacing:-.03em;font-size:1.25rem;color:#CBFA7C;margin:.2rem 0 .1rem}
.cu-q span{font-family:Inter,sans-serif;font-size:.72rem;letter-spacing:.12em;color:#A6B8AE}
.cu-member-note{border-left:2px solid #EF936F;background:#1B2A26;padding:12px 16px;margin:.4rem 0 1.2rem;font-size:.95rem;line-height:1.55;color:#F0F7ED}
.cu-member-note .e{display:block;font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#CBFA7C;margin-bottom:4px}
.cu-chip{display:inline-flex;align-items:center;gap:7px;font-size:.85rem;font-weight:500;color:#F0F7ED;white-space:nowrap}
.cu-chip i{width:9px;height:9px;border-radius:50%;display:inline-block}
.cu-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border:1px solid #263E36;background:#142322;margin:.2rem 0 1rem}
.cu-strip > div{padding:10px 14px;border-right:1px solid #1E302B}
.cu-strip > div:last-child{border-right:0}
.cu-strip .k{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#A6B8AE}
.cu-strip .v{font-size:.98rem;font-weight:600;color:#F0F7ED;margin-top:3px;line-height:1.3}
.cu-strip .v.bad{color:#F2A887}
.cu-seg{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:.4rem 0 1.2rem}
.cu-seg .lab-card{padding:16px 18px}
.cu-seg .lab-card h4{font-family:"Space Grotesk",Inter,sans-serif;font-weight:700;letter-spacing:-.03em;font-size:1.15rem;margin:.1rem 0 .5rem;display:flex;gap:8px;align-items:center}
.cu-seg .lab-card h4 i{width:10px;height:10px;border-radius:50%;flex:none}
.cu-seg dl{display:grid;grid-template-columns:1fr auto;gap:3px 12px;margin:0;font-size:.86rem}
.cu-seg dt{color:#A6B8AE}.cu-seg dd{margin:0;text-align:right;color:#F0F7ED;font-variant-numeric:tabular-nums}
.cu-seg .lab-card.is-member{box-shadow:0 0 0 2px #EF936F inset}
.cu-reasons{list-style:none;margin:.4rem 0 0;padding:0;font-size:.9rem}
.cu-reasons li{padding:6px 0;border-bottom:1px solid #1E302B;display:flex;justify-content:space-between;gap:12px}
.cu-reasons li span:last-child{color:#A6B8AE;font-variant-numeric:tabular-nums;white-space:nowrap}
</style>
"""


def css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def strip(items: list[tuple[str, str, bool]]) -> None:
    cells = "".join(f'<div><div class="k">{t.esc(k)}</div><div class="v{" bad" if bad else ""}">{v}</div></div>' for k, v, bad in items)
    st.markdown(f'<div class="cu-strip">{cells}</div>', unsafe_allow_html=True)
