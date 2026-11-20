"""Why?: which factors raised or lowered one UK Product–Market pair's Export Potential.

Ranking rows link here with `?hs6=...&iso3=...` (`app.ui.why_link`); without them the
Advisor picks a Product and a Market.
"""

import pandas as pd
import streamlit as st

from app import client
from app.format import format_usd, years_text
from app.frame import page
from app.ui import (
    fetch,
    labels,
    potential_range_bar,
    show_caveats,
    usd_unit,
)

TOP_FACTORS = 8


def _pair() -> tuple[str | None, str | None]:
    """The pair from the link, or the Advisor's pick."""
    params = st.query_params
    if params.get("hs6") and params.get("iso3"):
        return params["hs6"], params["iso3"]
    st.caption("Open Why? from a ranking row, or pick a Product and a Market.")
    query = st.text_input("Search Products by HS6 code or words, e.g. 851712 or smartphone", key="why_product_query")
    found = fetch("/products", client.api_url(), q=query.strip()) if query.strip() else []
    if query.strip() and not found:
        st.warning(f"No Product matches '{query.strip()}'. Try an HS6 code or another word.")
    if not found:
        return None, None
    descriptions = {product["hs6"]: product["description"] for product in found}
    hs6 = st.selectbox("Product", list(descriptions), format_func=lambda code: f"{code} – {descriptions[code]}", key="why_hs6")
    markets = fetch("/markets", client.api_url())
    names = {market["iso3"]: market["name"] for market in markets}
    iso3 = st.selectbox("Market", list(names), format_func=names.get, index=None, key="why_market")
    return hs6, iso3


with page("Why?", "Which factors raised or lowered one UK Product–Market pair's Export Potential."):
    hs6, iso3 = _pair()
    if hs6 and iso3:
        try:
            body = fetch(f"/explain/{hs6}/{iso3}", client.api_url())
        except client.NotFound as error:
            st.warning(f"No explanation for Product {hs6} in Market {iso3}: {error}")
        except client.ApiUnavailable as error:
            st.error(f"The API is unavailable: {error}")
        else:
            product, market = body["product"], body["market"]
            st.subheader(f"{product['hs6']} – {product['description']} → {market['name']}")
            show_caveats(body)
            if body["tariff_caveat"]:
                st.warning(body["tariff_caveat"])
            averaged = body["gap_period_label"]
            st.markdown(
                f"{averaged[0].upper()}{averaged[1:]}: "
                f"Export Potential **{format_usd(body['export_potential_usd'])}**, "
                f"Actual Exports **{format_usd(body['actual_exports_usd'])}**, "
                f"Export Gap **{format_usd(body['export_gap_usd'])}**. "
                f"Opportunity Type: **{body['opportunity_type']}**. "
                f"Baseline year {body['baseline_year']} Export Potential, which the factors below explain: "
                f"**{format_usd(body['baseline_export_potential_usd'])}**."
            )
            potential_range_bar(body)
            st.markdown(body["potential_range_text"])
            st.markdown(
                "Each factor below is something the model associates with higher or lower UK exports "
                "of this Product to this Market. These are associations the model has learned, not causes: "
                "a factor shown as raising Export Potential is not proof that changing it would raise exports."
            )
            factors = pd.DataFrame(body["factors"])
            factors["effect"] = [
                f"{row['direction']} ×{row['multiplier']:,.2f}" if row["direction"] != "no effect" else "no effect"
                for row in factors.to_dict("records")
            ]
            top = factors.head(TOP_FACTORS)
            raised = [f for f in top.to_dict("records") if f["direction"] == "raised"][:3]
            lowered = [f for f in top.to_dict("records") if f["direction"] == "lowered"][:3]
            if raised:
                st.markdown("**Most associated with higher Export Potential:** " + "; ".join(
                    f"{f['label']} ({f['value_text']})" for f in raised))
            if lowered:
                st.markdown("**Most associated with lower Export Potential:** " + "; ".join(
                    f"{f['label']} ({f['value_text']})" for f in lowered))
            st.dataframe(
                factors.rename(
                    columns={
                        "label": "Factor",
                        "value_text": "Value for this pair",
                        "effect": "Association with Export Potential",
                    }
                )[["Factor", "Value for this pair", "Association with Export Potential"]],
                hide_index=True,
            )
            st.caption(
                "×1.50 means the model associates this factor with 1.5 times the Export Potential it expects on average "
                "(SHAP values); ×0.70 means 30% lower. The factors multiply together into the Export Potential. "
                f"Computed with SHAP from fold model {body['fold']}, the same model that scored this pair's Baseline."
            )

            st.subheader("Year by year")
            baseline_year = body["baseline_year"]
            points = {point["year"]: point for point in body["history"]}
            years = list(range(baseline_year - 5, baseline_year + 1))
            empty = {
                "actual_exports_usd": 0.0, "export_potential_usd": None, "export_gap_usd": None, "tariff_imputed": None
            }
            yearly = [points.get(year, empty) for year in years]
            unit = usd_unit()
            # Each year's Applied Tariff mark (ADR 0006); blank for a year with no row.
            tariff_marks = {True: f"Carried forward from {labels()['last_tariff_year']}", False: "From tariff data"}
            st.dataframe(
                pd.DataFrame(
                    {
                        "Year": years,
                        f"Export Potential ({unit})": [p["export_potential_usd"] for p in yearly],
                        f"Actual Exports ({unit})": [p["actual_exports_usd"] for p in yearly],
                        f"Export Gap ({unit})": [p["export_gap_usd"] for p in yearly],
                    },
                    dtype=float,
                ).astype({"Year": int}).assign(
                    **{"Applied Tariff": [tariff_marks.get(p["tariff_imputed"], "") for p in yearly]}
                ),
                hide_index=True,
            )
            span = years_text(body["gap_years"])
            st.caption(
                f"The Export Gap above is the {averaged} of the yearly Export Gaps of {span}, the Gap Period; "
                "each year's Export Potential is scored from that year's conditions. Earlier years show UK Actual "
                f"Exports only. {baseline_year} figures are provisional. A year with no recorded UK exports shows 0."
            )
