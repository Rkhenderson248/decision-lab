"""Shared look and feel for every Decision Lab page.

The palette mirrors richardhenderson.io (NIGHTSHIFT): obsidian ground, lime as
the signal, coral as the warm accent. The three chart hues were checked for
color-vision deficiency on the dark surface: lime and coral separate by
lightness, periwinkle by hue.

Token names are kept from the light theme (PETROL is the accent, INK the main
text), so every page picks up the new look without changes.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PETROL = "#CBFA7C"    # accent: lime
PETROL_2 = "#B9EE62"
LUME = "#EF936F"      # warm accent: coral
INK = "#F0F7ED"       # main text: porcelain cream
GRAPHITE = "#C9D6CE"  # body text
MUTED = "#A6B8AE"     # secondary text: sage
GRID = "#263E36"      # rules and borders: forest
GRID_2 = "#1E302B"    # faint gridlines
SURFACE = "#101B1B"   # page: obsidian
MIST = "#1B2A26"      # panels: graphite green
CARD = "#142322"      # cards: deep forest

# Categorical series, fixed order.
S1 = "#CBFA7C"  # lime, the model / the recommended path
S2 = "#EF936F"  # coral, the incumbent / the comparison
S3 = "#8FA2F0"  # periwinkle, a third series when one is needed
BASE = "#6E827A"  # neutral reference (random, no model)

SITE_URL = "https://richardhenderson.io"

FONT_TEXT = "Inter, Helvetica Neue, Arial, sans-serif"
FONT_DISPLAY = "Space Grotesk, Inter, Arial, sans-serif"


def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_TEXT, size=13, color=GRAPHITE),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[S1, S2, S3, "#E07AA0", "#D9C24A", "#5CC3EE"],
        margin=dict(l=8, r=16, t=78, b=8),
        hoverlabel=dict(
            bgcolor=CARD,
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
        title=dict(font=dict(family=FONT_DISPLAY, size=14.5, color=INK), x=0, xanchor="left", xref="paper", y=0.985, yanchor="top", yref="container"),
        annotationdefaults=dict(font=dict(color=INK), arrowcolor=MUTED),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=MUTED), outlinewidth=0)),
    )
    template.data.heatmap = [go.Heatmap(colorbar=dict(tickfont=dict(color=MUTED), outlinewidth=0))]
    pio.templates["atelier"] = template
    pio.templates.default = "atelier"


_register_plotly_template()

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_CSS = """
<style>
:root{--petrol:%(petrol)s;--lume:%(lume)s;--ink:%(ink)s;--graphite:%(graphite)s;--muted:%(muted)s;--grid:%(grid)s;--mist:%(mist)s;--card:%(card)s;--surface:%(surface)s}
[data-testid="stAppViewContainer"] .block-container{padding-top:4.6rem;padding-bottom:3rem;max-width:1180px}
.lab-solo{display:none}
header[data-testid="stHeader"]{background:rgba(16,27,27,.86);backdrop-filter:blur(8px)}
#MainMenu, footer{visibility:hidden}
h1,h2,h3{letter-spacing:-.035em}
h1{font-size:clamp(2rem,4.2vw,3.1rem)!important;line-height:1.02!important}
.lab-eyebrow{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--petrol);margin:0 0 .55rem;font-weight:700}
.lab-lede{font-size:1.06rem;color:var(--graphite);max-width:62ch;line-height:1.6;margin:.4rem 0 1.2rem}
.lab-rule{height:1px;border:0;margin:.4rem 0 1.4rem;background:linear-gradient(90deg,var(--petrol),var(--lume) 40%%,transparent)}
.lab-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0;border-top:1px solid var(--grid);border-bottom:1px solid var(--grid);margin:.4rem 0 1.4rem}
.lab-tile{padding:16px 18px 14px 0}
.lab-tile + .lab-tile{border-left:1px solid var(--grid);padding-left:18px}
.lab-tile .k{font-size:.8rem;color:var(--muted);line-height:1.35}
.lab-tile .v{font-family:"Space Grotesk",Inter,sans-serif;font-weight:700;font-size:clamp(1.6rem,3vw,2.15rem);line-height:1.05;color:var(--ink);margin-top:6px;letter-spacing:-.045em}
.lab-tile.is-accent .v{color:var(--petrol)}
.lab-tile .v .sg{font-family:"Space Grotesk",Inter,sans-serif;font-weight:600;margin-right:.03em}
.lab-tile .n{font-size:.8rem;color:var(--muted);margin-top:5px;line-height:1.4}
.lab-card{background:var(--card);border:1px solid var(--grid);padding:20px 22px;position:relative;height:100%%}
.lab-card::before{content:"";position:absolute;left:0;right:0;top:-1px;height:2px;background:linear-gradient(90deg,var(--petrol),var(--lume))}
.lab-card h3{font-family:"Space Grotesk",Inter,sans-serif;font-weight:700;letter-spacing:-.035em;font-size:1.35rem;margin:.2rem 0 .5rem;color:var(--ink)}
.lab-card p{color:var(--graphite);font-size:.95rem;margin:0}
.lab-pill{display:inline-block;font-size:.72rem;padding:3px 8px;background:rgba(203,250,124,.12);color:#CBFA7C;border:1px solid rgba(203,250,124,.3);margin-right:6px}
.lab-pill.warn{background:rgba(239,147,111,.12);color:#F2A887;border-color:rgba(239,147,111,.35)}
.lab-pill.off{background:transparent;color:var(--muted);border-color:var(--grid)}
.lab-note{font-size:.82rem;color:var(--muted);border-top:1px solid var(--grid);padding-top:.8rem;margin-top:2rem;line-height:1.55}
.lab-note a{color:var(--petrol)}
.lab-insight{border-left:2px solid var(--petrol);background:var(--card);padding:12px 16px;margin:.4rem 0 1.2rem;color:var(--ink);font-size:.98rem;line-height:1.55}
.lab-insight b{color:var(--petrol)}
.lab-reco{background:#263E36;color:var(--ink);padding:22px 24px;position:relative;border-left:3px solid var(--petrol)}
.lab-reco .e{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--petrol);font-weight:700}
.lab-reco .t{font-family:"Space Grotesk",Inter,sans-serif;font-size:1.7rem;font-weight:700;letter-spacing:-.04em;line-height:1.1;margin:.35rem 0 .45rem}
.lab-reco .d{color:var(--graphite);font-size:.95rem;line-height:1.5}
.lab-checks{list-style:none;margin:.3rem 0 0;padding:0;font-size:.9rem}
.lab-checks li{padding:6px 0;border-bottom:1px solid var(--grid);color:var(--graphite);display:flex;gap:8px}
.lab-checks li:last-child{border-bottom:0}
.lab-checks .ok{color:#CBFA7C;font-weight:700;min-width:1.1em}
.lab-checks .no{color:#EF936F;font-weight:700;min-width:1.1em}
.cp-rail{display:grid;grid-template-columns:repeat(7,1fr);gap:0;margin:.2rem 0 1.1rem;border-top:1px solid var(--grid)}
.cp-rail span{font-size:.72rem;letter-spacing:.06em;color:var(--muted);padding:8px 6px 0 0;border-top:2px solid transparent;margin-top:-1px;line-height:1.3}
.cp-rail span.done{border-top-color:#6E827A;color:var(--graphite)}
.cp-rail span.on{border-top-color:var(--petrol);color:var(--petrol);font-weight:700}
.cp-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:10px 24px;margin:.6rem 0 .2rem}
.cp-head h2{font-family:"Space Grotesk",Inter,sans-serif;font-weight:700;letter-spacing:-.04em;font-size:clamp(1.5rem,2.8vw,2rem)!important;margin:0!important;padding:0!important}
.cp-proves{font-size:.78rem;color:var(--petrol);border:1px solid rgba(203,250,124,.35);background:rgba(203,250,124,.06);padding:4px 10px;white-space:nowrap}
.cp-why{color:var(--graphite);max-width:70ch;line-height:1.6;margin:.3rem 0 1rem}
.cp-call{border:1px solid var(--grid);border-left:2px solid var(--lume);background:var(--card);padding:14px 18px;margin:.6rem 0 1.2rem;font-size:.95rem;line-height:1.55;color:var(--ink)}
.cp-call .e{display:block;font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#F2A887;margin-bottom:4px;font-weight:700}
.cp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:.4rem 0 1.2rem}
.cp-grid .lab-card h4{font-family:Inter,sans-serif!important;font-size:.72rem;line-height:1.4;padding:0!important;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .4rem;font-weight:700}
.cp-grid .lab-card p{font-size:.95rem}
@media (max-width:640px){.cp-rail{grid-template-columns:repeat(4,1fr)}}
div[data-testid="stExpander"] details{background:var(--card)}
button[kind="primary"],button[data-testid="stBaseButton-primary"],[data-testid="stDownloadButton"] button[kind="primary"]{color:#101B1B!important;font-weight:700!important}
button[kind="primary"] p,button[data-testid="stBaseButton-primary"] p{color:#101B1B!important;font-weight:700!important}
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:#F0F7ED!important;border-color:#F0F7ED!important}
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
    "card": CARD,
    "surface": SURFACE,
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
        "Decision science for pricing, risk and customer value.</div>",
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

