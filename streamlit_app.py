"""Decision Lab: interactive demos for richardhenderson.io.

Run locally:   streamlit run streamlit_app.py
Embed a page:  https://<your-app>.streamlit.app/<page>?embed=true&solo=1
"""

import streamlit as st

from lab import theme

st.set_page_config(
    page_title="Decision Lab · R.K. Henderson",
    page_icon="static/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.apply()

home = st.Page("views/home.py", title="Decision Lab", icon=":material/science:", default=True)
shelves = {
    "": [home],
    "Products": [
        st.Page("views/lending_lab.py", title="Lending decision lab", icon=":material/account_balance:", url_path="lending-lab"),
        st.Page("views/policy_assistant.py", title="Credit policy assistant", icon=":material/quick_reference_all:", url_path="policy-assistant"),
        st.Page("views/copilot.py", title="Pricing & demand copilot", icon=":material/insights:", url_path="pricing-copilot"),
        st.Page("views/market_intel.py", title="Mortgage market intelligence", icon=":material/map:", url_path="market-intelligence"),
    ],
    "Case studies": [
        st.Page("views/lending.py", title="Lending decisions", icon=":material/format_list_numbered:", url_path="lending"),
    ],
    "Method notes": [
        st.Page("views/experiments.py", title="Experimentation", icon=":material/science:", url_path="experiments"),
        st.Page("views/decision_value.py", title="Model value", icon=":material/payments:", url_path="model-value"),
        st.Page("views/human_ai.py", title="Human–AI decisions", icon=":material/handshake:", url_path="human-ai"),
        st.Page("views/goodhart.py", title="Goodhart simulator", icon=":material/trending_down:", url_path="goodhart"),
    ],
}

# On the website each demo is embedded on its own, so the navigation is hidden.
nav = st.navigation(shelves, position="hidden" if theme.is_solo() else "top")
nav.run()
