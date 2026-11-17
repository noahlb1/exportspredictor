"""Why?: which factors raised or lowered one UK Product–Market pair's Export Potential.

Ranking rows link here with `?hs6=...&iso3=...` (`app.ui.why_link`); without them the
Advisor picks a Product and a Market.
"""

import pandas as pd
import streamlit as st

from app import client
from app.ui import (
    banner,
    fetch,
    gap_period_label,
    potential_range_bar,
    product_picker,
    show_caveats,
)
from exportspredictor.conformal import potential_range_text
from exportspredictor.currency import USD_UNIT, format_usd

TOP_FACTORS = 8

st.title("Why?")


def _pair() -> tuple[str | None, str | None]:
    """The pair from the link, or the Advisor's pick."""
    params = st.query_params
    if params.get("hs6") and params.get("iso3"):
        return params["hs6"], params["iso3"]
    st.caption("Open Why? from a ranking row, or pick a Product and a Market.")
    hs6 = product_picker()
    if not hs6:
        return None, None
    markets = fetch("/markets", client.api_url())
    names = {market["iso3"]: market["name"] for market in markets}
    iso3 = st.selectbox("Market", list(names), format_func=names.get, index=None, key="why_market")
    return hs6, iso3


if banner() is not None:
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
            averaged = gap_period_label(body["gap_years"])
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
            st.markdown(
                potential_range_text(
                    body["potential_range_low_usd"],
                    body["potential_range_high_usd"],
                    body["actual_exports_usd"],
                    uncertain=body["uncertain"],
                )
            )
            st.markdown(
                "Each factor below is something the model associates with higher or lower UK exports "
                "of this Product to this Market. These are associations the model has learned, not causes: "
                "a factor shown as raising Export Potential is not proof that changing it would raise exports."
            )
            factors = pd.DataFrame(body["factors"])
            factors["effect"] = [
                f"{row.direction} ×{row.multiplier:,.2f}" if row.direction != "no effect" else "no effect"
                for row in factors.itertuples()
            ]
            top = factors.head(TOP_FACTORS)
            raised = [f for f in top.itertuples() if f.direction == "raised"][:3]
            lowered = [f for f in top.itertuples() if f.direction == "lowered"][:3]
            if raised:
                st.markdown("**Most associated with higher Export Potential:** " + "; ".join(
                    f"{f.label} ({f.value_text})" for f in raised))
            if lowered:
                st.markdown("**Most associated with lower Export Potential:** " + "; ".join(
                    f"{f.label} ({f.value_text})" for f in lowered))
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
            empty = {"actual_exports_usd": 0.0, "export_potential_usd": None, "export_gap_usd": None}
            yearly = [points.get(year, empty) for year in years]
            st.dataframe(
                pd.DataFrame(
                    {
                        "Year": years,
                        f"Export Potential ({USD_UNIT})": [p["export_potential_usd"] for p in yearly],
                        f"Actual Exports ({USD_UNIT})": [p["actual_exports_usd"] for p in yearly],
                        f"Export Gap ({USD_UNIT})": [p["export_gap_usd"] for p in yearly],
                    },
                    dtype=float,
                ).astype({"Year": int}),
                hide_index=True,
            )
            first, last = body["gap_years"]
            st.caption(
                f"The Export Gap above is the {averaged} of the yearly Export Gaps of {first}–{last}, the Gap Period; "
                "each year's Export Potential is scored from that year's conditions. Earlier years show UK Actual "
                f"Exports only. {baseline_year} figures are provisional. A year with no recorded UK exports shows 0."
            )
