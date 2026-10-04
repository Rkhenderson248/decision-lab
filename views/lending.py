import streamlit as st

from lab import theme as t
from lab import widgets as w
from lab import view_pipeline, view_product_fit

VIEWS = {
    "Prioritize the pipeline": ("pipeline", view_pipeline.render),
    "Recommend the product": ("product", view_product_fit.render),
}
BY_KEY = {key: label for label, (key, _) in VIEWS.items()}

requested = str(st.query_params.get("view", "pipeline")).lower()
default = BY_KEY.get(requested, "Prioritize the pipeline")

st.markdown('<p class="lab-eyebrow">Lending decisions · Two production patterns, rebuilt on synthetic data</p>',
            unsafe_allow_html=True)
choice = w.segmented("Decision", list(VIEWS), key="lending_view", default=default, label_visibility="collapsed")
st.query_params["view"] = VIEWS[choice][0]
st.write("")
VIEWS[choice][1]()
