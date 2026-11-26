"""How the model works: how the model is built, checked and served, for a visitor who wants to learn the ML.

The outline and its text live in `app/how_it_works.py`; this page only lays them out."""

import streamlit as st

from app.frame import page
from app.how_it_works import INTRO, LAYERS, SECTIONS

with page("How the model works", INTRO):
    for section in SECTIONS:
        st.subheader(section.title)
        for idea in section.ideas:
            st.markdown(f"**{idea.name}**: {idea.summary}")
            for field, label in LAYERS.items():
                text = getattr(idea, field)
                if text:
                    with st.expander(label):
                        st.markdown(text)
