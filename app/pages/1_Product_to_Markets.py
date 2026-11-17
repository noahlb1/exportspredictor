"""Product -> Markets: Markets ranked by the UK's Export Gap for one Product."""

import streamlit as st

from app import client
from app.ui import banner, fetch, product_picker, ranking_filters, show_ranking

st.title("Product → Markets")

if banner() is not None:
    try:
        filters = ranking_filters()
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        hs6 = product_picker(filters.get("sector"))
        if hs6:
            try:
                body = fetch(f"/opportunities/by-product/{hs6}", client.api_url(), **filters)
            except client.NotFound as error:
                st.warning(str(error))
            except client.ApiUnavailable as error:
                st.error(f"The API is unavailable: {error}")
            else:
                product = body["product"]
                st.subheader(f"{product['hs6']} – {product['description']}")
                show_ranking(body, {"market": "Market", "region": "Region"}, file_name=f"markets_for_{hs6}")
