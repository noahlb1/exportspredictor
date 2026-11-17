"""Market -> Products: Products ranked by the UK's Export Gap in one Market."""

import streamlit as st

from app import client
from app.ui import banner, excluded_products_list, fetch, market_picker, ranking_filters, show_ranking

st.title("Market → Products")

if banner() is not None:
    try:
        filters = ranking_filters()
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        excluded_products_list()
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
                # Every row of one Market shares its UK Trade Agreement status (none when filtered out).
                rows = body["opportunities"]
                if rows:
                    st.markdown(
                        f"The UK has a Trade Agreement in force with {market['name']} in {body['baseline_year']}."
                        if rows[0]["trade_agreement"]
                        else f"There is no UK Trade Agreement in force with {market['name']} in {body['baseline_year']}."
                    )
                st.caption("Open **Why?** on a row to see which factors raised or lowered its Export Potential.")
                show_ranking(
                    body,
                    {"hs6": "HS6", "description": "Product", "sector": "Sector"},
                    file_name=f"products_for_{iso3}",
                    iso3=iso3,
                )
