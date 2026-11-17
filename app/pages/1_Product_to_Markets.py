"""Product -> Markets: Markets ranked by the UK's Export Gap, or by Blocked Opportunity value, for one Product."""

import pandas as pd
import streamlit as st

from app import client
from app.ui import RANK_BY, banner, fetch, product_picker, why_column, why_link

st.title("Product → Markets")

if banner() is not None:
    hs6 = product_picker()
    if hs6:
        rank_by = st.radio("Rank by", list(RANK_BY), format_func=RANK_BY.get, key="rank_by", horizontal=True)
        try:
            body = fetch(f"/opportunities/by-product/{hs6}", client.api_url(), rank_by=rank_by)
        except client.NotFound as error:
            st.warning(str(error))
        except client.ApiUnavailable as error:
            st.error(f"The API is unavailable: {error}")
        else:
            product = body["product"]
            st.subheader(f"{product['hs6']} – {product['description']}")
            st.caption(
                f"Markets ranked by {RANK_BY[body['rank_by']]}, Baseline year {body['baseline_year']}. "
                "Open **Why?** on a row to see which factors raised or lowered its Export Potential."
            )
            if body["rank_by"] == "blocked":
                st.caption(
                    "Blocked Opportunity value: the Scenario Effect the model associates with adding a Trade "
                    "Agreement where none exists (0 where one does). Flagged rows clear 20% of Baseline Export "
                    "Potential and $1M."
                )
            table = pd.DataFrame(body["opportunities"])
            table["Why?"] = [why_link(product["hs6"], iso3) for iso3 in table["iso3"]]
            table = table.rename(
                columns={
                    "market": "Market",
                    "region": "Region",
                    "export_potential_usd": "Export Potential (USD)",
                    "actual_exports_usd": "Actual Exports (USD)",
                    "export_gap_usd": "Export Gap (USD)",
                    "blocked_effect_usd": "Blocked Opportunity value (USD)",
                    "blocked_opportunity": "Blocked Opportunity",
                }
            )
            st.dataframe(
                table[
                    [
                        "Market",
                        "Region",
                        "Export Potential (USD)",
                        "Actual Exports (USD)",
                        "Export Gap (USD)",
                        "Blocked Opportunity value (USD)",
                        "Blocked Opportunity",
                        "Why?",
                    ]
                ],
                hide_index=True,
                column_config={"Why?": why_column()},
            )
