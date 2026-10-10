"""A thin 'Stage 3 of 9' progress bar under each lab's stage control (own module, see lab/widgets.py)."""

import streamlit as st


def bar(idx: int, total: int, name: str) -> None:
    pct = (idx + 1) / total * 100
    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin:-.2rem 0 .4rem">'
        '<div style="flex:1;height:3px;background:#263E36;border-radius:2px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#CBFA7C,#EF936F);transition:width .5s ease"></div></div>'
        f'<span style="font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;color:#A6B8AE;white-space:nowrap">'
        f'Stage {idx + 1} of {total} · {name}</span></div>',
        unsafe_allow_html=True,
    )
