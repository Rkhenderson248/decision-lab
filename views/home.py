import streamlit as st

from lab import theme as t

t.header(
    "R.K. Henderson · Decision Lab",
    "Working demos of decision tools",
    "Small applications, each built around one decision, on public or synthetic data. They explain themselves "
    "and they show the idea behind the practice: intelligence is only worth what it changes.",
)

SHELVES = {
    "Predict & prioritise": [
        ("views/market_intel.py", "Market intelligence", "Where is the mortgage market opening up?",
         "Every U.S. metro and micro area scored from public data, rebuilt monthly, with lender benchmarking and grounded briefs."),
        ("views/lending.py", "Lending decisions", "Who to call first, and which product to offer",
         "A capacity-constrained prioritisation model with reason codes, and an explainable product recommender."),
        ("views/demand_pricing.py", "Forecasting & pricing", "How many will sell, and at what price?",
         "Booking-pace forecasts backtested against last year, then capacity-constrained pricing and protection levels."),
    ],
    "Decide": [
        ("views/decision_value.py", "Decision economics", "What is a model actually worth?",
         "Turn AUC, capacity, cost and adoption into net value, and find where the next dollar comes from."),
        ("views/experiments.py", "Experimentation", "Will this test actually tell you anything?",
         "Power and duration, the peeking problem, CUPED variance reduction and a proper read-out."),
    ],
    "Measure & trust": [
        ("views/human_ai.py", "Human–AI collaboration", "Do you know when to trust the model?",
         "Take a ten-deal judge–advisor experiment and get your own reliance profile."),
        ("views/goodhart.py", "Measurement", "When a measure becomes a target",
         "Simulate a team under metric pressure and watch the reported number part ways with the real outcome."),
    ],
}

for shelf, demos in SHELVES.items():
    st.subheader(shelf, anchor=False)
    cols = st.columns(3, gap="medium")
    for i, (path, eyebrow, title, blurb) in enumerate(demos):
        with cols[i % 3]:
            st.markdown(
                f'<div class="lab-card"><p class="lab-eyebrow">{t.esc(eyebrow)}</p>'
                f"<h3>{t.esc(title)}</h3><p>{t.esc(blurb)}</p></div>",
                unsafe_allow_html=True,
            )
            st.page_link(path, label="Open the demo", icon=":material/arrow_outward:")
    st.write("")

t.footnote("Market data is public and rebuilt monthly; everything else is synthetic. Rules and rates are illustrative, not advice.")
