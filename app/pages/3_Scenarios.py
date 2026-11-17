"""Scenarios: re-predict UK Export Potential under a Trading Bloc preset or custom Trade Agreements."""

import json

import pandas as pd
import streamlit as st

from app import client
from app.ui import banner, fetch, run_scenario

st.title("Scenarios")

PRESET = "Trading Bloc preset"
CUSTOM = "Custom Trade Agreements"
PAIR_COLUMNS = {
    "hs6": "HS6",
    "description": "Product",
    "market": "Market",
    "baseline_potential_usd": "Baseline Export Potential (USD)",
    "scenario_potential_usd": "Scenario Export Potential (USD)",
    "effect_usd": "Scenario Effect (USD)",
    "effect_share": "Scenario Effect (share of Baseline)",
}
TOTAL_COLUMNS = {
    "baseline_potential_usd": "Baseline Export Potential (USD)",
    "scenario_potential_usd": "Scenario Export Potential (USD)",
    "effect_usd": "Scenario Effect (USD)",
    "effect_share": "Scenario Effect (share of Baseline)",
    "pairs": "Product–Market pairs changed",
}


def table(rows: list[dict], columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns)).rename(columns=columns)


if banner() is not None:
    try:
        presets = fetch("/scenarios/presets", client.api_url())
        markets = fetch("/markets", client.api_url())
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        st.stop()
    try:
        backtest = fetch("/evaluation", client.api_url()).get("event_backtest")
    except client.NotFound:
        backtest = None
    if backtest:
        st.info(f"**How the model's Scenario Effects compared with a real Trade Agreement.** {backtest['text']}")
        st.caption(f"{backtest['caveat']} The Model Evaluation page has the details.")
    preset_info = {preset["name"]: preset for preset in presets}
    market_names = {market["iso3"]: market["name"] for market in markets}

    kind = st.radio("Scenario", [PRESET, CUSTOM], key="kind", horizontal=True)
    if kind == PRESET:
        preset = st.selectbox(
            "Trading Bloc preset",
            list(preset_info),
            format_func=lambda name: preset_info[name]["title"],
            key="preset",
        )
        st.caption(preset_info[preset]["description"])
        body = {"preset": preset}
    else:
        add = st.multiselect(
            "Add a Trade Agreement with", list(market_names), format_func=market_names.get, key="add"
        )
        remove = st.multiselect(
            "Remove the Trade Agreement with", list(market_names), format_func=market_names.get, key="remove"
        )
        body = {"add_agreements": add, "remove_agreements": remove}

    st.markdown("**Blocked Opportunity thresholds**: a pair counts when its Scenario Effect clears both.")
    left, right = st.columns(2)
    min_share = left.number_input(
        "Minimum Scenario Effect, % of Baseline Export Potential", min_value=0.0, value=20.0, step=5.0, key="min_share"
    )
    min_usd = right.number_input(
        "Minimum Scenario Effect, USD million", min_value=0.0, value=1.0, step=0.5, key="min_usd"
    )
    body |= {"min_effect_share": min_share / 100, "min_effect_usd": min_usd * 1e6, "limit": 500}

    if st.button("Run Scenario", key="run", type="primary"):
        st.session_state["scenario_body"] = json.dumps(body, sort_keys=True)

    if "scenario_body" in st.session_state:
        try:
            result = run_scenario(st.session_state["scenario_body"], client.api_url())
        except client.InvalidRequest as error:
            st.warning(f"This Scenario is not valid: {error}")
        except client.ApiUnavailable as error:
            st.error(f"The API is unavailable: {error}")
        else:
            if result["changes_nothing"]:
                st.warning(result["summary"])
            else:
                st.markdown(result["summary"])
                st.caption(
                    "Scenario Effects are associations the model has learned from past trade, not causal "
                    "forecasts: the model associates these policy conditions with this change in Export "
                    f"Potential, holding everything else at the {result['baseline_year']} Baseline."
                )
                total = result["total"]
                a, b, c = st.columns(3)
                a.metric("Scenario Effect (USD)", f"{total['effect_usd']:,.0f}")
                b.metric("Share of Baseline", f"{total['effect_share']:+.1%}")
                c.metric("Blocked Opportunities", f"{result['blocked_count']:,}")

                st.subheader("Change per Market")
                st.dataframe(table(result["markets"], {"market": "Market", **TOTAL_COLUMNS}), hide_index=True)

                st.subheader("Top Products gaining")
                st.dataframe(
                    table(result["products"], {"hs6": "HS6", "description": "Product", **TOTAL_COLUMNS}),
                    hide_index=True,
                )

                st.subheader("Blocked Opportunities")
                st.caption(
                    f"{result['blocked_count']:,} pairs have a Scenario Effect of at least "
                    f"{result['thresholds']['min_effect_share']:.0%} of Baseline Export Potential and at least "
                    f"${result['thresholds']['min_effect_usd']:,.0f}. The largest are shown."
                )
                st.dataframe(table(result["blocked_opportunities"], PAIR_COLUMNS), hide_index=True)

                st.subheader("Drill down: the pairs most affected in one Market")
                changed = [market["iso3"] for market in result["markets"]]
                market = st.selectbox(
                    "Market", changed, format_func=lambda iso3: market_names.get(iso3, iso3), key="drill_market"
                )
                pairs = [pair for pair in result["top_pairs"] if pair["iso3"] == market]
                st.dataframe(table(pairs, PAIR_COLUMNS), hide_index=True)
