import streamlit as st

from lab import theme as t

t.header(
    "R.K. Henderson · Decision Lab",
    "Working demos of decision tools",
    "Small applications, each built around one decision. They run on synthetic or public data, they explain "
    "themselves, and they show the idea behind the practice: intelligence is only worth what it changes.",
)

DEMOS = [
    ("views/market_intel.py", "Market intelligence", "Where is the mortgage market opening up?",
     "Every U.S. metro and micro area scored from public Census, HMDA and FHFA data, rebuilt monthly, with your own strategy weights."),
    ("views/lending.py", "Lending decisions", "Who to call first, and which product to offer",
     "A capacity-constrained prioritisation model with reason codes, and an explainable product recommender with hard eligibility rules."),
    ("views/decision_value.py", "Decision economics", "What is a model actually worth?",
     "Turn AUC, capacity, cost and adoption into net value, and find where the next dollar comes from."),
    ("views/goodhart.py", "Measurement", "When a measure becomes a target",
     "Simulate a team under metric pressure and watch the reported number part ways with the real outcome."),
]

cols = st.columns(2, gap="medium")
for i, (path, eyebrow, title, blurb) in enumerate(DEMOS):
    with cols[i % 2]:
        st.markdown(
            f'<div class="lab-card"><p class="lab-eyebrow">{t.esc(eyebrow)}</p>'
            f"<h3>{t.esc(title)}</h3><p>{t.esc(blurb)}</p></div>",
            unsafe_allow_html=True,
        )
        st.page_link(path, label="Open the demo", icon=":material/arrow_outward:")
        st.write("")

t.footnote("Every dataset here is synthetic. Rules and rates are illustrative, not advice.")
