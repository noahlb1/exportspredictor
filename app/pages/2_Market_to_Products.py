"""Market -> Products: Products ranked by the UK's Export Gap in one Market."""

import streamlit as st

from app import client
from app.ui import banner, fetch, market_picker, ranking_filters, show_ranking

st.title("Market → Products")

if banner() is not None:
    try:
        filters = ranking_filters()
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        iso3 = market_picker(filters.get("region"))
        if iso3:
            try:
                body = fetch(f"/opportunities/by-market/{iso3}", client.api_url(), **filters)
            except client.NotFound as error:
                st.warning(str(error))
            except client.ApiUnavailable as error:
                st.error(f"The API is unavailable: {error}")
            else:
                market = body["market"]
                st.subheader(f"{market['name']} ({market['region']})")
                # Every row of one Market shares its UK Trade Agreement status; fixed per Market.
                in_force = any(row["trade_agreement"] for row in body["opportunities"])
                st.markdown(
                    f"The UK has a Trade Agreement in force with {market['name']} in {body['baseline_year']}."
                    if in_force
                    else f"There is no UK Trade Agreement in force with {market['name']} in {body['baseline_year']}."
                )
                show_ranking(
                    body,
                    {"hs6": "HS6", "description": "Product", "sector": "Sector"},
                    file_name=f"products_for_{iso3}",
                )
