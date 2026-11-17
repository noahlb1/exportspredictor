"""Model Evaluation: how accurate the model is, where it is weak, and what it cannot tell you."""

import pandas as pd
import streamlit as st

from app import client
from app.ui import banner, fetch

SCOPES = {
    "uk_out_of_fold": "UK Product–Market pairs (each scored by the fold model that never saw it)",
    "all_exporters": "All Exporters (every Flow of the test year, plus sampled zero cells)",
}
WEAKEST_SHOWN = 5
TARIFF_GROUPS = {"observed": "Observed", "imputed": "Carried forward from 2021"}


def _metric_table(overall: dict, metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": info["label"],
                "Model": overall[f"model_{name}"],
                "Naive baseline": overall[f"naive_{name}"],
                "Better is": info["better"],
            }
            for name, info in metrics.items()
        ]
    )


def _breakdown_table(rows: list[dict], label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                label: row["group"],
                "Products ranked": row["ranked_products"],
                "Rank correlation (model)": row["model_spearman_within_product"],
                "Rank correlation (naive baseline)": row["naive_spearman_within_product"],
                "RMSE of log value (model)": row["model_rmse_log"],
                "RMSE of log value (naive baseline)": row["naive_rmse_log"],
                "Tweedie deviance (model)": row["model_tweedie_deviance"],
                "Tweedie deviance (naive baseline)": row["naive_tweedie_deviance"],
                "Rows": row["rows"],
            }
            for row in rows
        ]
    )


def _weakest(rows: list[dict]) -> str:
    ranked = [row for row in rows if row["model_spearman_within_product"] is not None]
    return ", ".join(f"{row['group']} ({row['model_spearman_within_product']:.2f})" for row in ranked[:WEAKEST_SHOWN])


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0%}"


def _interval(ci: list) -> str:
    low, high = ci
    return "n/a" if low is None else f"{_pct(low)} to {_pct(high)}"


def _years(years: list[int]) -> str:
    return str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"


def _show_lift(lift: dict) -> None:
    st.subheader("Do large Export Gaps point at growth? (Lift)")
    st.markdown(lift["text"])
    held = len(lift["held_out_exporters"])
    st.caption(
        f"Lift is measured on {held} Exporters ({lift['held_out_share']:.0%}, chosen at random) that a model trained "
        f"on {_years(lift['training_years'])} never saw. Each of their Product–Market pairs is ranked by its "
        f"{lift['gap_year']} Export Gap, and a pair counts as grown when its Flow rose by at least "
        f"{lift['min_growth_share']:.0%} and ${lift['min_growth_usd']:,.0f} by {lift['outcome_year']}. The interval "
        f"comes from resampling whole Exporters. It describes a tendency across many pairs, not a forecast."
    )
    deciles = pd.DataFrame(
        {
            "Export Gap tenth (1 = largest gaps)": [str(row["decile"]) for row in lift["by_decile"]],
            "Share that grew materially": [row["rate"] for row in lift["by_decile"]],
        }
    )
    st.bar_chart(deciles, x="Export Gap tenth (1 = largest gaps)", y="Share that grew materially")


def _effects_table(direction: dict, post_years: list[int]) -> pd.DataFrame:
    pooled = [{**direction["pooled"], "year": _years(post_years)}] if len(post_years) > 1 else []
    rows = [*direction["by_year"], *pooled]
    return pd.DataFrame(
        [
            {
                "Year": str(row["year"]),
                "Predicted by the model": _pct(row["predicted"]),
                "Observed (difference-in-differences)": _pct(row["observed"]),
                "Observed 95% interval": _interval(row["observed_ci"]),
            }
            for row in rows
        ]
    )


def _tariff_table(rows: list[dict]) -> pd.DataFrame:
    table = _breakdown_table(rows, "Applied Tariff")
    table["Applied Tariff"] = [TARIFF_GROUPS[row["group"]] for row in rows]
    return table


def _show_tariff_split(report: dict) -> None:
    st.subheader("Observed against carried-forward tariffs")
    st.markdown(report["tariff_split_text"])
    for scope, label in SCOPES.items():
        split = report["accuracy"][scope].get("by_tariff")
        if split:
            st.markdown(f"**{label}**")
            st.dataframe(_tariff_table(split), hide_index=True)
    st.caption(
        "Tariff data end in 2021, so later years reuse each pair's 2021 Applied Tariff, or 0 where a Trade "
        "Agreement entered force after 2021. Splitting the test rows this way shows what carrying 2021 forward costs."
    )


def _show_potential_range(coverage: dict) -> None:
    st.subheader("How often Actual Exports fall inside the Potential Range")
    st.markdown(coverage["text"])
    by_exporter = pd.DataFrame(
        [
            {"Exporter": row["exporter"], "Coverage": row["coverage"], "Rows": row["rows"]}
            for row in coverage["by_exporter"]
        ]
    )
    st.dataframe(by_exporter, hide_index=True)
    st.caption(
        f"The Potential Range is calibrated on other Exporters' rows, never the UK's, and checked here on a held-out "
        f"half of them, per Exporter. Coverage near {coverage['target']:.0%} for every Exporter supports the "
        f"assumption that UK pairs vary as other Exporters' pairs do; it does not prove it."
    )


def _show_event_backtest(backtest: dict, names: dict[str, str]) -> None:
    st.subheader("Scenario Effects against a real Trade Agreement (Event Backtest)")
    for event in backtest["events"]:
        role = "Secondary check: " if event["role"] == "secondary" else ""
        st.markdown(f"**{role}{event['title']}**, in force {event['in_force']}")
        st.markdown(event["text"])
        for direction in event["directions"]:
            controls = "; ".join(names.get(code, code) for code in direction["controls"])
            st.markdown(f"*{direction['label']}*. Control Markets: {controls}.")
            st.dataframe(_effects_table(direction, event["post_years"]), hide_index=True)
        st.caption(
            f"The model was trained on {_years(event['training_years'])} only, before the agreement. \"Predicted\" "
            f"is its Scenario Effect: Export Potential with the agreement on against off. \"Observed\" is the change "
            f"from {_years(event['pre_years'])} relative to the change in the control Markets. "
            f"{event.get('states', '')} {event['control_rule']} {event['caveat']}"
        )


st.title("Model Evaluation")

if banner() is not None:
    try:
        report = fetch("/evaluation", client.api_url())
    except client.NotFound as error:
        report = None
        st.warning(f"{error} Rebuild the artefacts to evaluate the model.")
    except client.ApiUnavailable as error:
        report = None
        st.error(f"The API is unavailable: {error}")

    if report is not None:
        test_year = report["test_year"]
        training_years = report["test_training_years"]
        rate = report["non_uk_sampling_rate"]
        sample = (
            "every non-UK row"
            if rate >= 1
            else f"a {rate:.0%} sample of non-UK rows (reweighted so totals stay unbiased)"
        )

        st.subheader("How accurate is the model?")
        headline = report["headline"]
        (st.success if headline["beats_naive"] else st.warning)(headline["text"])
        st.caption(
            f"Measured on {test_year}, a year the model never saw: the models were trained on "
            f"{training_years[0]}–{training_years[-1]} with settings tuned on {report['validation_year']}, so "
            f"neither tuning nor training ever read {test_year}. They were trained on every UK row and {sample}, "
            f"as the Baseline models are. The naive baseline is {report['naive_baseline'][0].lower()}"
            f"{report['naive_baseline'][1:]}"
        )

        st.subheader(f"Accuracy in {test_year}")
        for scope, label in SCOPES.items():
            st.markdown(f"**{label}**")
            st.dataframe(_metric_table(report["accuracy"][scope]["overall"], report["metrics"]), hide_index=True)
        st.caption(
            "Rank correlation is averaged over Products: 1 means Markets are ranked exactly as the UK's actual "
            "exports rank them, 0 means no better than chance. RMSE of log value is the typical error in orders of "
            f"magnitude on Flows that happened. Tweedie deviance (power {report['tweedie_power']}) is the "
            "training loss over every row, zeros included."
        )

        st.subheader("Where the model is weakest")
        scope = st.radio(
            "Rows evaluated", list(SCOPES), format_func=SCOPES.get, key="breakdown_scope", horizontal=True
        )
        accuracy = report["accuracy"][scope]
        sector_tab, region_tab = st.tabs(["By Sector", "By region"])
        with sector_tab:
            st.markdown(f"**Weakest Sectors** (HS2 chapter, rank correlation): {_weakest(accuracy['by_sector'])}")
            st.dataframe(_breakdown_table(accuracy["by_sector"], "Sector"), hide_index=True)
        with region_tab:
            st.markdown(f"**Weakest regions** (Market region, rank correlation): {_weakest(accuracy['by_region'])}")
            st.dataframe(_breakdown_table(accuracy["by_region"], "Region"), hide_index=True)
        st.caption("Sorted weakest first, by the model's rank correlation within each Product.")

        if report.get("tariff_split_text"):
            _show_tariff_split(report)

        st.subheader("Stability of the rankings")
        st.markdown(report["stability"]["text"])

        if report.get("potential_range"):
            _show_potential_range(report["potential_range"])

        if report.get("lift"):
            _show_lift(report["lift"])
        if report.get("event_backtest"):
            names = {market["iso3"]: market["name"] for market in fetch("/markets", client.api_url())}
            _show_event_backtest(report["event_backtest"], names)

        st.subheader("Known limitations")
        with st.container(border=True):
            for limitation in report["limitations"]:
                st.markdown(f"**{limitation['title']}.** {limitation['text']}")
