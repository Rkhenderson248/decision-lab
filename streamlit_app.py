"""Decision Lab: interactive demos for richardhenderson.io.

Run locally:   streamlit run streamlit_app.py
Embed a page:  https://<your-app>.streamlit.app/pipeline?embed=true&solo=1
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

pages = [
    st.Page("views/home.py", title="Decision Lab", icon=":material/science:", default=True),
    st.Page("views/pipeline.py", title="Pipeline prioritizer", icon=":material/format_list_numbered:", url_path="pipeline"),
    st.Page("views/product_fit.py", title="Product fit", icon=":material/rule:", url_path="product-fit"),
    st.Page("views/decision_value.py", title="Model value", icon=":material/payments:", url_path="model-value"),
    st.Page("views/goodhart.py", title="Goodhart simulator", icon=":material/trending_down:", url_path="goodhart"),
]

# On the website each demo is embedded on its own, so the navigation is hidden.
nav = st.navigation(pages, position="hidden" if theme.is_solo() else "top")
nav.run()
