import streamlit as st

from lab import theme as t

t.header(
    "R.K. Henderson · Decision Lab",
    "Decision products, case studies and method notes",
    "One flagship product followed end to end, a public-data product rebuilt every month, and short interactive "
    "notes on the methods behind them. Each piece is built around one decision and explains its own reasoning.",
)

SHELVES = {
    "Products": [
        ("views/copilot.py", "Flagship · end to end", "Pricing & demand copilot",
         "Frame, data, model, decide, prove, run and value: one pricing decision taken from question to measured ROI."),
        ("views/market_intel.py", "Data product · public data", "Market intelligence",
         "Every U.S. metro scored from Census, HMDA and FHFA data, rebuilt monthly by an automated pipeline, with grounded briefs."),
    ],
    "Case studies": [
        ("views/lending.py", "Prioritisation & recommendation", "Lending decisions",
         "Who to call first under fixed capacity, and which product to offer, both with reason codes."),
    ],
    "Method notes": [
        ("views/experiments.py", "Experimentation", "Will this test tell you anything?",
         "Power, the peeking problem, CUPED and a proper read-out."),
        ("views/decision_value.py", "Decision economics", "What is a model worth?",
         "Turn AUC, capacity, cost and adoption into net value."),
        ("views/human_ai.py", "Human–AI collaboration", "When should people trust the model?",
         "A ten-round judge–advisor experiment that gives you your own reliance profile."),
        ("views/goodhart.py", "Measurement", "When a measure becomes a target",
         "Watch the reported number part ways with the real outcome under pressure."),
    ],
}

for shelf, demos in SHELVES.items():
    st.subheader(shelf, anchor=False)
    cols = st.columns(4 if len(demos) == 4 else 3, gap="medium")
    for i, (path, eyebrow, title, blurb) in enumerate(demos):
        with cols[i % len(cols)]:
            st.markdown(
                f'<div class="lab-card"><p class="lab-eyebrow">{t.esc(eyebrow)}</p>'
                f"<h3>{t.esc(title)}</h3><p>{t.esc(blurb)}</p></div>",
                unsafe_allow_html=True,
            )
            st.page_link(path, label="Open the demo", icon=":material/arrow_outward:")
    st.write("")

t.footnote("Market data is public and rebuilt monthly; everything else is synthetic. Rules and rates are illustrative, not advice.")
