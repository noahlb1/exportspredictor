"""How the Model Works: how the model is built, checked and served, for a visitor who wants to learn the ML.

The outline and its text are worded in the package (`exportspredictor.how_it_works`) for the served model and read from
the API (`/how-it-works`); this page only lays them out."""

import streamlit as st

from app import client
from app.frame import page
from app.ui import fetch

text = fetch("/how-it-works", client.api_url())

with page("How the Model Works", text["intro"]):
    for section in text["sections"]:
        st.subheader(section["title"])
        for idea in section["ideas"]:
            st.markdown(f"**{idea['name']}**: {idea['summary']}")
            for field, label in text["layers"].items():
                if idea[field]:
                    with st.expander(label):
                        st.markdown(idea[field])
