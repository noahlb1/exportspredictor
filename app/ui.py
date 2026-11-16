"""Shared page furniture: cached API reads and the persistent banner."""

import streamlit as st

from app import client


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> dict | list:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def banner() -> dict | None:
    """Show the Baseline-year and caveat banner. Returns health, or None if the API is down."""
    try:
        health = client.get("/health")
    except client.ApiUnavailable as error:
        st.error(
            f"The API is unavailable: {error} "
            "Start everything with `podman compose up`, or for development run `uv run uvicorn api.main:app`. "
            "Then reload this page."
        )
        return None
    st.info(
        f"Baseline year {health['baseline_year']} (provisional data). "
        "Figures are associations the model has learned, not causes. "
        f"Model version {health['artefact_version']}."
    )
    return health


def product_picker() -> str | None:
    """Search Products by HS6 code or words and pick one. Returns the chosen hs6, or None."""
    query = st.text_input(
        "Search Products by HS6 code (2012 edition) or words, e.g. 851712 or smartphone",
        key="product_query",
    ).strip()
    if not query:
        return None
    try:
        found = fetch("/products", client.api_url(), q=query)
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        return None
    if not found:
        st.warning(f"No Product matches '{query}'. Try an HS6 code or another word.")
        return None
    descriptions = {product["hs6"]: product["description"] for product in found}
    return st.selectbox(
        "Product",
        list(descriptions),
        format_func=lambda hs6: f"{hs6} – {descriptions[hs6]}",
        key="hs6",
    )
