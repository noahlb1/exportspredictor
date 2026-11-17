"""Product -> Markets: Markets ranked by the UK's Export Gap, or by Blocked Opportunity value, for one Product."""

import streamlit as st

from app import client
from app.ui import (
    banner,
    excluded_products_list,
    fetch,
    product_picker,
    ranking_filters,
    show_ranking,
)

st.title("Product → Markets")

if banner() is not None:
    try:
        filters = ranking_filters()
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        excluded_products_list()
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
                st.caption("Open **Why?** on a row to see which factors raised or lowered its Export Potential.")
                show_ranking(body, {"market": "Market", "region": "Region"}, file_name=f"markets_for_{hs6}", hs6=hs6)
