"""Shared look and feel for every Decision Lab page.

The palette mirrors richardhenderson.io: porcelain ground, petrol authority,
aqua as light. The three chart hues were run through a colour-vision check
(adjacent and all-pairs) against the light surface.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PETROL = "#0F4640"
PETROL_2 = "#15544D"
LUME = "#7FD0BE"
INK = "#0E1311"
GRAPHITE = "#4A534D"
MUTED = "#6B756E"
GRID = "#E1E4DE"
GRID_2 = "#EDEFEA"
SURFACE = "#FAFAF8"
MIST = "#F1F2EE"

# Categorical series, fixed order. Validated: CVD ΔE >= 9.4, normal-vision ΔE >= 17.
S1 = "#008A73"  # teal, the model / the recommended path
S2 = "#D9772B"  # orange, the incumbent / the comparison
S3 = "#5B6CB8"  # indigo, a third series when one is needed
BASE = "#A3ABA5"  # neutral reference (random, no model)

SITE_URL = "https://richardhenderson.io"

FONT_TEXT = "Schibsted Grotesk, Helvetica Neue, Arial, sans-serif"
FONT_DISPLAY = "Bodoni Moda, Didot, Georgia, serif"


def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_TEXT, size=13, color=GRAPHITE),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[S1, S2, S3],
        margin=dict(l=8, r=16, t=78, b=8),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=GRID,
            font=dict(family=FONT_TEXT, size=13, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(size=12.5, color=GRAPHITE),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            automargin=True,
            showgrid=False,
            zeroline=False,
            linecolor=GRID,
            ticks="outside",
            tickcolor=GRID,
            ticklen=4,
            title=dict(font=dict(size=12.5, color=MUTED)),
            tickfont=dict(size=12, color=MUTED),
        ),
        yaxis=dict(
            automargin=True,
            gridcolor=GRID_2,
            gridwidth=1,
            zeroline=False,
            linecolor="rgba(0,0,0,0)",
            title=dict(font=dict(size=12.5, color=MUTED)),
            tickfont=dict(size=12, color=MUTED),
        ),
        title=dict(font=dict(family=FONT_TEXT, size=14, color=INK), x=0, xanchor="left", xref="paper", y=0.985, yanchor="top", yref="container"),
    )
    pio.templates["atelier"] = template
    pio.templates.default = "atelier"


_register_plotly_template()

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_CSS = """
<style>
:root{--petrol:%(petrol)s;--lume:%(lume)s;--ink:%(ink)s;--graphite:%(graphite)s;--muted:%(muted)s;--grid:%(grid)s;--mist:%(mist)s}
[data-testid="stAppViewContainer"] .block-container{padding-top:4.6rem;padding-bottom:3rem;max-width:1180px}
.lab-solo{display:none}
header[data-testid="stHeader"]{background:rgba(250,250,248,.86);backdrop-filter:blur(8px)}
#MainMenu, footer{visibility:hidden}
h1,h2,h3{letter-spacing:-.018em}
h1{font-size:clamp(2rem,4.2vw,3rem)!important;line-height:1.05!important}
.lab-eyebrow{font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--petrol);margin:0 0 .55rem;font-weight:500}
.lab-lede{font-size:1.06rem;color:var(--graphite);max-width:62ch;line-height:1.6;margin:.4rem 0 1.2rem}
.lab-rule{height:1px;border:0;margin:.4rem 0 1.4rem;background:linear-gradient(90deg,var(--petrol),var(--lume) 40%%,transparent)}
.lab-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0;border-top:1px solid var(--grid);border-bottom:1px solid var(--grid);margin:.4rem 0 1.4rem}
.lab-tile{padding:16px 18px 14px 0}
.lab-tile + .lab-tile{border-left:1px solid var(--grid);padding-left:18px}
.lab-tile .k{font-size:.8rem;color:var(--muted);line-height:1.35}
.lab-tile .v{font-family:"Bodoni Moda",Georgia,serif;font-weight:600;font-size:clamp(1.6rem,3vw,2.15rem);line-height:1.05;color:var(--ink);margin-top:6px;letter-spacing:-.01em}
.lab-tile.is-accent .v{color:var(--petrol)}
.lab-tile .v .sg{font-family:"Schibsted Grotesk",sans-serif;font-weight:500;margin-right:.04em}
.lab-tile .n{font-size:.8rem;color:var(--muted);margin-top:5px;line-height:1.4}
.lab-card{background:#fff;border:1px solid var(--grid);padding:20px 22px;position:relative;height:100%%}
.lab-card::before{content:"";position:absolute;left:0;right:0;top:-1px;height:2px;background:linear-gradient(90deg,var(--petrol),var(--lume))}
.lab-card h3{font-family:"Bodoni Moda",Georgia,serif;font-size:1.35rem;margin:.2rem 0 .5rem}
.lab-card p{color:var(--graphite);font-size:.95rem;margin:0}
.lab-pill{display:inline-block;font-size:.72rem;padding:3px 8px;background:#E2F0E7;color:#1F5A47;border-radius:3px;margin-right:6px}
.lab-pill.warn{background:#FBEBDD;color:#8A4514}
.lab-pill.off{background:#EEF0EC;color:#59625C}
.lab-note{font-size:.82rem;color:var(--muted);border-top:1px solid var(--grid);padding-top:.8rem;margin-top:2rem;line-height:1.55}
.lab-note a{color:var(--petrol)}
.lab-insight{border-left:2px solid var(--petrol);background:#fff;padding:12px 16px;margin:.4rem 0 1.2rem;color:var(--ink);font-size:.98rem;line-height:1.55}
.lab-insight b{color:var(--petrol)}
.lab-reco{background:var(--petrol);color:#FAFAF8;padding:22px 24px;position:relative}
.lab-reco .e{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#C3E2D5}
.lab-reco .t{font-family:"Bodoni Moda",Georgia,serif;font-size:1.7rem;font-weight:600;line-height:1.1;margin:.35rem 0 .45rem}
.lab-reco .d{color:#C8D9CD;font-size:.95rem;line-height:1.5}
.lab-checks{list-style:none;margin:.3rem 0 0;padding:0;font-size:.9rem}
.lab-checks li{padding:6px 0;border-bottom:1px solid var(--grid);color:var(--graphite);display:flex;gap:8px}
.lab-checks li:last-child{border-bottom:0}
.lab-checks .ok{color:#0F7563;font-weight:600;min-width:1.1em}
.lab-checks .no{color:#B4561B;font-weight:600;min-width:1.1em}
div[data-testid="stExpander"] details{background:#fff}
@media (max-width:640px){.lab-tile + .lab-tile{border-left:0;padding-left:0;border-top:1px solid var(--grid)}}
</style>
""" % {
    "petrol": PETROL,
    "lume": LUME,
    "ink": INK,
    "graphite": GRAPHITE,
    "muted": MUTED,
    "grid": GRID,
    "mist": MIST,
}


def is_solo() -> bool:
    """True when the page is embedded as a single demo on the website."""
    value = st.query_params.get("solo", "")
    return str(value).lower() in {"1", "true", "yes"}


_SOLO_CSS = """
<style>
[data-testid="stAppViewContainer"] .block-container{padding-top:1.6rem}
header[data-testid="stHeader"]{display:none}
</style>
"""


def apply() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    if is_solo():
        st.markdown(_SOLO_CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<p class="lab-eyebrow">{esc(eyebrow)}</p>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="lab-lede">{esc(lede)}</p><hr class="lab-rule">', unsafe_allow_html=True)


def tiles(items: list[dict]) -> None:
    """A row of stat tiles. Each item: label, value, note (optional), accent (optional)."""
    cells = []
    for item in items:
        klass = "lab-tile is-accent" if item.get("accent") else "lab-tile"
        note = f'<div class="n">{esc(item["note"])}</div>' if item.get("note") else ""
        value = esc(item["value"])
        if value[:1] in {"+", "−", "-"}:
            value = f'<span class="sg">{value[0]}</span>{value[1:]}'

        cells.append(
            f'<div class="{klass}"><div class="k">{esc(item["label"])}</div>'
            f'<div class="v">{value}</div>{note}</div>'
        )
    st.markdown(f'<div class="lab-tiles">{"".join(cells)}</div>', unsafe_allow_html=True)


def insight(html_text: str) -> None:
    """A single highlighted sentence. Caller is responsible for escaping inputs."""
    st.markdown(f'<div class="lab-insight">{html_text}</div>', unsafe_allow_html=True)


def footnote(text: str) -> None:
    st.markdown(
        f'<div class="lab-note">{esc(text)} '
        f'Built by <a href="{SITE_URL}" target="_blank" rel="noopener">R.K. Henderson</a> · '
        "Applied decision science.</div>",
        unsafe_allow_html=True,
    )


def money(value: float, decimals: int = 0) -> str:
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:,.2f}M"
    if value >= 10_000:
        return f"{sign}${value / 1_000:,.0f}K"
    return f"{sign}${value:,.{decimals}f}"


def pct(value: float, decimals: int = 0, signed: bool = False) -> str:
    text = f"{abs(value) * 100:.{decimals}f}%"
    if signed:
        return ("+" if value >= 0 else "−") + text
    return ("−" if value < 0 else "") + text


def chart(fig: go.Figure, height: int = 360) -> None:
    fig.update_layout(template="atelier", height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG)
