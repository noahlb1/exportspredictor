"""Shared page furniture: cached API reads and the persistent banner."""

import streamlit as st

from app import client


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> dict:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def banner() -> dict | None:
    """Show the Baseline-year and caveat banner. Returns health, or None if the API is down."""
    try:
        health = client.get("/health")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error} Start it with `uv run uvicorn api.main:app`.")
        return None
    st.info(
        f"Baseline year {health['baseline_year']} (provisional data). "
        "Figures are associations the model has learned, not causes. "
        f"Model version {health['artefact_version']}."
    )
    return health
