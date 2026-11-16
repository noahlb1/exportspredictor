"""Product -> Markets: Markets ranked by the UK's Export Gap for one Product."""

import pandas as pd
import streamlit as st

from app import client
from app.ui import banner, fetch

st.title("Product → Markets")

if banner() is not None:
    hs6 = st.text_input("Product (HS6 code, 2012 edition)", key="hs6", max_chars=6).strip()
    if hs6:
        try:
            body = fetch(f"/opportunities/by-product/{hs6}", client.api_url())
        except client.NotFound as error:
            st.warning(str(error))
        except client.ApiUnavailable as error:
            st.error(f"The API is unavailable: {error}")
        else:
            product = body["product"]
            st.subheader(f"{product['hs6']} – {product['description']}")
            st.caption(f"Markets ranked by Export Gap, Baseline year {body['baseline_year']}.")
            table = pd.DataFrame(body["opportunities"]).rename(
                columns={
                    "market": "Market",
                    "region": "Region",
                    "export_potential_usd": "Export Potential (USD)",
                    "actual_exports_usd": "Actual Exports (USD)",
                    "export_gap_usd": "Export Gap (USD)",
                }
            )
            st.dataframe(
                table[["Market", "Region", "Export Potential (USD)", "Actual Exports (USD)", "Export Gap (USD)"]],
                hide_index=True,
            )
