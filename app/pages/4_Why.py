"""Why?: which factors raised or lowered one UK Product–Market pair's Export Potential.

Ranking rows link here with `?hs6=...&iso3=...` (`app.ui.why_link`); without them the
Advisor picks a Product and a Market.
"""

import pandas as pd
import streamlit as st

from app import client
from app.ui import banner, fetch, product_picker
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
            st.markdown(
                f"Baseline year {body['baseline_year']}: "
                f"Export Potential **{format_usd(body['export_potential_usd'])}**, "
                f"Actual Exports **{format_usd(body['actual_exports_usd'])}**, "
                f"Export Gap **{format_usd(body['export_gap_usd'])}**."
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

            st.subheader("UK Actual Exports, recent years")
            baseline_year = body["baseline_year"]
            exports = {point["year"]: point["actual_exports_usd"] for point in body["history"]}
            years = range(baseline_year - 5, baseline_year + 1)
            st.dataframe(
                pd.DataFrame({"Year": list(years), f"Actual Exports ({USD_UNIT})": [exports.get(y, 0.0) for y in years]}),
                hide_index=True,
            )
            st.caption(f"{baseline_year} figures are provisional. A year with no recorded UK exports shows 0.")
