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

        st.subheader("Stability of the rankings")
        st.markdown(report["stability"]["text"])

        # Lift and the Event Backtest (#21) go here.

        st.subheader("Known limitations")
        with st.container(border=True):
            for limitation in report["limitations"]:
                st.markdown(f"**{limitation['title']}.** {limitation['text']}")
