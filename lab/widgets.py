"""Widgets shared across pages.

Kept apart from lab.theme on purpose: Streamlit Cloud can hot-reload a page
while an already-imported module stays cached, so new helpers live in a module
of their own and every page imports it directly.
"""

import streamlit as st


def segmented(label: str, options: list, key: str, default=None, **kwargs):
    """A segmented control that never deselects.

    Streamlit's segmented control returns None when the selected option is
    clicked again, which made pages fall back to their first option. This keeps
    the last choice instead, and restores the highlight on the control.
    """
    last_key = f"{key}__last"
    if key not in st.session_state:
        st.session_state[key] = default if default in options else options[0]

    def _keep() -> None:
        if st.session_state.get(key) is None:
            st.session_state[key] = st.session_state.get(last_key, default or options[0])

    value = st.segmented_control(label, options, key=key, on_change=_keep, **kwargs)
    if value is None:  # first render after an external reset
        value = st.session_state.get(last_key, default or options[0])
    st.session_state[last_key] = value
    return value
