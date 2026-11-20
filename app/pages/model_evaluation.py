"""Model Evaluation: how accurate the model is, where it is weak, and what it cannot tell you."""

import pandas as pd
import streamlit as st

from app import client
from app.format import number_text, years_text
from app.frame import page
from app.ui import fetch, labels

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


LARGEST_MARKETS = (30, 60)


def _largest_markets_table(headline: dict, large: dict | None) -> pd.DataFrame:
    """The headline beside the same measure on the 30 and 60 largest importing Markets; "n/a" for a version
    evaluated before they were reported."""
    found = {row["markets"]: row for row in (large or {}).get("rows", [])}
    rows = [("Every Market (the headline)", headline)] + [
        (f"{n} largest importers", found.get(n, {})) for n in LARGEST_MARKETS
    ]
    return pd.DataFrame(
        {
            "Markets measured": [name for name, _ in rows],
            "Model": [number_text(row.get("model"), ".2f") for _, row in rows],
            "Naive Baseline": [number_text(row.get("naive"), ".2f") for _, row in rows],
        }
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


SIZE_EXPLANATION = (
    "Raw Lift compares the pairs with the largest Export Gaps with all pairs, so it mostly reflects pair size: "
    "large gaps sit on large pairs, which grow by ${usd:,.0f} more easily. Size-matched Lift compares them with "
    "pairs of similar Export Potential instead, so it shows what the Export Gap adds beyond size. Above 1, large "
    "gaps tended to grow more often than pairs of their size; below 1, less often. Both describe a tendency across "
    "many pairs, not a forecast for any one Opportunity."
)


def _lift_figures(lift: dict) -> pd.DataFrame:
    matched = lift["size_matched"]
    return pd.DataFrame(
        [
            {
                "Measure": "Raw Lift",
                "Lift": lift["lift"],
                "95% interval": f"{number_text(lift['ci'][0], '.1f')} to {number_text(lift['ci'][1], '.1f')}",
                "Compared with": "all pairs",
            },
            {
                "Measure": "Size-matched Lift",
                "Lift": matched["lift"],
                "95% interval": f"{number_text(matched['ci'][0], '.2f')} to {number_text(matched['ci'][1], '.2f')}",
                "Compared with": "pairs of similar Export Potential",
            },
        ]
    )


def _show_lift(lift: dict) -> None:
    st.markdown(lift["text"])
    st.dataframe(_lift_figures(lift), hide_index=True)
    st.markdown(SIZE_EXPLANATION.format(usd=lift["min_growth_usd"]))
    gap = years_text(lift["gap_years"])
    st.caption(
        f"Lift is measured on {len(lift['held_out_exporters'])} Exporters that a model trained on "
        f"{years_text(lift['training_years'])} never saw: the Exporters v1 held out, within v2's Exporter scope, so both "
        f"models are scored on the same countries. Each of their Product–Market pairs is ranked by its Export Gap "
        f"averaged over {gap}, and a pair counts as grown when its {lift['outcome_year']} Flow was at least "
        f"{lift['min_growth_share']:.0%} and ${lift['min_growth_usd']:,.0f} above its {gap} mean Actual Exports. "
        f"The intervals come from resampling whole Exporters."
    )
    deciles = pd.DataFrame(
        {
            "Export Gap tenth (1 = largest gaps)": [str(row["decile"]) for row in lift["by_decile"]],
            "Share that grew materially": [row["rate"] for row in lift["by_decile"]],
        }
    )
    st.bar_chart(deciles, x="Export Gap tenth (1 = largest gaps)", y="Share that grew materially")


def _effects_table(direction: dict, post_years: list[int]) -> pd.DataFrame:
    pooled = [{**direction["pooled"], "year": years_text(post_years)}] if len(post_years) > 1 else []
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


def _tariff_table(rows: list[dict], last_tariff_year: int) -> pd.DataFrame:
    groups = {"observed": "Observed", "imputed": f"Carried forward from {last_tariff_year}"}
    table = _breakdown_table(rows, "Applied Tariff")
    table["Applied Tariff"] = [groups[row["group"]] for row in rows]
    return table


def _show_tariff_split(report: dict) -> None:
    st.markdown(report["tariff_split_text"])
    year = labels()["last_tariff_year"]
    for scope, label in SCOPES.items():
        split = report["accuracy"][scope].get("by_tariff")
        if split:
            st.markdown(f"**{label}**")
            st.dataframe(_tariff_table(split, year), hide_index=True)
    st.caption(
        f"Tariff data end in {year}, so later years reuse each pair's {year} Applied Tariff, or 0 where a Trade "
        f"Agreement entered force after {year}. Splitting the test rows this way shows what carrying {year} forward costs."
    )


def _show_potential_range(coverage: dict) -> None:
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
    if coverage.get("by_size"):
        by_size = pd.DataFrame(
            [
                {"Size of Export Potential": row["size"], "Coverage": row["coverage"], "Rows": row["rows"]}
                for row in coverage["by_size"]
            ]
        )
        st.dataframe(by_size, hide_index=True)
        st.caption(
            "The same held-out check by size of Export Potential: a large pair's range is calibrated on pairs of "
            "similar size, so coverage near the target at every size means the range fits large pairs as well as small."
        )


def _show_model_choice(choice: dict) -> None:
    """How the served model was chosen (ADR 0008): every figure comes from the API, read from the artefact."""
    st.markdown(choice["text"])
    if choice["recorded"]:
        st.markdown(f"**Model Form.** {choice['model_form']['text']}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Experiment": e.get("label") or e["name"],
                        "Best Arm": e["best_arm"] or "n/a",
                        "Selection Measure": e["selection_measure"],
                        "Change on the best so far": e["change"],
                        "Kept": e["kept"] or "no",
                    }
                    for e in choice["experiments"]
                ]
            ),
            hide_index=True,
        )
        st.caption(
            f"Win Threshold {choice['win_threshold']:.4f}: an Arm had to beat the best so far by more than this, twice "
            f"the Noise Floor's standard deviation, to be kept. Measured on {choice['validation_year']}, never on the "
            "test year."
        )
        confirmation = choice["confirmation"]
        if confirmation:
            folds = [{"Fold": str(f["fold"]), "Chosen Model Form": f["winner"], "v2 (the model alone)": f["v2"]}
                     for f in confirmation["folds"]]
            folds.append({"Fold": "All folds", "Chosen Model Form": confirmation["winner"],
                          "v2 (the model alone)": confirmation["v2"]})
            verdict = "confirmed" if confirmation["beats_v2"] else "not confirmed"
            st.markdown(
                f"**Confirmation on all {confirmation['n_folds']} folds of {choice['validation_year']}**: the chosen "
                f"Model Form against v2's, difference {confirmation['difference']:+.4f} against the Win Threshold "
                f"{confirmation['win_threshold']:.4f} ({verdict}). The Naive Baseline scores {confirmation['naive']:.4f}."
            )
            st.dataframe(pd.DataFrame(folds), hide_index=True)
    ranking = choice["test_year_ranking"]
    if ranking:
        st.markdown(f"**The UK ranking in {ranking['test_year']}, beside the Naive Baseline and earlier versions**")
        (st.success if ranking["beats_naive"] else st.warning)(ranking["text"])
        st.dataframe(
            pd.DataFrame(
                [{"Version": row["name"], "UK ranking, old measure": row["uk_ranking"],
                  "UK ranking, tie-proof": row.get("uk_ranking_tie_proof")}
                 for row in [*ranking["rows"], *ranking["naive"]]]
            ),
            hide_index=True,
        )
    if choice["traded_away"]:
        st.markdown(f"**What the Blend traded away.** {choice['traded_away']['text']}")


def _show_event_backtest(backtest: dict, names: dict[str, str], note: dict | None) -> None:
    if note:
        st.info(note["text"])
    for event in backtest["events"]:
        role = "Secondary check: " if event["role"] == "secondary" else ""
        st.markdown(f"**{role}{event['title']}**, in force {event['in_force']}")
        st.markdown(event["text"])
        for direction in event["directions"]:
            controls = "; ".join(names.get(code, code) for code in direction["controls"])
            st.markdown(f"*{direction['label']}*. Control Markets: {controls}.")
            st.dataframe(_effects_table(direction, event["post_years"]), hide_index=True)
        st.caption(
            f"The model was trained on {years_text(event['training_years'])} only, before the agreement. \"Predicted\" "
            f"is its Scenario Effect: Export Potential with the agreement on against off. \"Observed\" is the change "
            f"from {years_text(event['pre_years'])} relative to the change in the control Markets. "
            f"{event.get('states', '')} {event['control_rule']} {event['caveat']}"
        )


def _show_scorecard(report: dict) -> None:
    """The four figures above the sections: ranking against the Naive Baseline, coverage, Lift and stability."""
    headline, coverage, lift, stability = (
        report["headline"], report.get("potential_range"), report.get("lift"), report["stability"],
    )
    rank, inside, matched, steady = st.columns(4)
    rank.metric(
        "Rank correlation, UK pairs",
        number_text(headline["model"], ".2f"),
        f"Naive Baseline {number_text(headline['naive'], '.2f')}",
        delta_color="off",
        delta_arrow="off",
        help=f"How well each Product's Markets are ranked in {report['test_year']}, a year the model never saw; "
        "1 means ranked exactly as the UK's Actual Exports rank them.",
        border=True,
    )
    inside.metric(
        "Inside the Potential Range",
        "n/a" if coverage is None else f"{coverage['coverage']:.0%}",
        None if coverage is None else f"target {coverage['target']:.0%}",
        delta_color="off",
        delta_arrow="off",
        help="How often Actual Exports fall inside the Potential Range, on held-out rows of other Exporters.",
        border=True,
    )
    matched.metric(
        "Size-matched Lift",
        "n/a" if lift is None else number_text(lift["size_matched"]["lift"], ".2f"),
        "above 1: large Export Gaps grew more often than pairs of their size",
        delta_color="off",
        delta_arrow="off",
        help="A tendency across many pairs, not a forecast for any one Opportunity.",
        border=True,
    )
    steady.metric(
        "Ranking stability",
        number_text(stability["spearman_potential"], ".2f"),
        f"{stability['top_overlap']:.0%} of the top {stability['top_n']:,} the same",
        delta_color="off",
        delta_arrow="off",
        help=f"Rank correlation of Export Potential between {years_text(stability['years'])}, scored by the same "
        "models, and the share of the top Opportunities by Export Gap found in both years.",
        border=True,
    )


def _show_excluded_products(excluded: list[dict], reasons: list[dict]) -> None:
    """Why some Products are never ranked: the reasons with their counts (counted by the API), and the full list
    folded below."""
    st.markdown(
        f"{len(excluded):,} Products are left out of training, scoring and every ranking, because their recorded "
        "trade follows bullion-market, pipeline or grid geography the model cannot see, or is unreliable; their "
        "trade still counts in every country and world total."
    )
    full = pd.DataFrame(
        {
            "HS6": [row["hs6"] for row in excluded],
            "Product": [row["description"] for row in excluded],
            "Sector": [row["sector_name"] for row in excluded],
            "Reason": [row["reason"] for row in excluded],
        }
    )
    st.dataframe(
        pd.DataFrame(
            {"Reason": [row["reason"] for row in reasons], "Products": [row["products"] for row in reasons]}
        ),
        hide_index=True,
    )
    with st.expander(f"All {len(full):,} Excluded Products"):
        st.dataframe(full, hide_index=True)


OPEN_AT_START = {"How accurate is the model?", "Known limitations"}


def _section(heading: str):
    """One folded section of the page; accuracy and the known limitations start open."""
    return st.expander(heading, expanded=heading in OPEN_AT_START)


with page("Model Evaluation", "How accurate the model is, where it is weak, and what it cannot tell you."):
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

        _show_scorecard(report)

        with _section("How accurate is the model?"):
            headline = report["headline"]
            (st.success if headline["beats_naive"] else st.warning)(headline["text"])
            large = report.get("large_markets")
            st.dataframe(_largest_markets_table(headline, large), hide_index=True)
            st.caption(
                large["text"] if large else "The measure on the largest importing Markets was not computed for this "
                "version: it is reported from the next evaluation of the test year."
            )
            st.caption(
                f"Measured on {test_year}, a year the model never saw: the models were trained on "
                f"{years_text(training_years)} with settings tuned on {report['validation_year']}, so "
                f"neither tuning nor training ever read {test_year}. They were trained on every UK row and {sample}, "
                f"as the Baseline models are. The naive baseline is {report['naive_baseline'][0].lower()}"
                f"{report['naive_baseline'][1:]}"
            )

        with _section("How the model was chosen"):
            _show_model_choice(report["model_choice"])

        with _section(f"Accuracy in {test_year}"):
            for scope, label in SCOPES.items():
                st.markdown(f"**{label}**")
                st.dataframe(_metric_table(report["accuracy"][scope]["overall"], report["metrics"]), hide_index=True)
            st.caption(
                "Rank correlation is averaged over Products: 1 means Markets are ranked exactly as the UK's actual "
                "exports rank them, 0 means no better than chance. RMSE of log value is the typical error in orders "
                f"of magnitude on Flows that happened. Tweedie deviance (power {report['tweedie_power']}) is the "
                "training loss over every row, zeros included."
            )

        with _section("Where the model is weakest"):
            scope = st.radio(
                "Rows evaluated", list(SCOPES), format_func=SCOPES.get, key="breakdown_scope", horizontal=True
            )
            accuracy = report["accuracy"][scope]
            sector_tab, region_tab = st.tabs(["By Sector", "By region"])
            with sector_tab:
                st.markdown(f"**Weakest Sectors** (HS2 chapter, rank correlation): {_weakest(accuracy['by_sector'])}")
                st.dataframe(_breakdown_table(accuracy["by_sector"], "Sector"), hide_index=True)
            with region_tab:
                st.markdown(
                    f"**Weakest regions** (Market region, rank correlation): {_weakest(accuracy['by_region'])}"
                )
                st.dataframe(_breakdown_table(accuracy["by_region"], "Region"), hide_index=True)
            st.caption("Sorted weakest first, by the model's rank correlation within each Product.")

        if report.get("tariff_split_text"):
            with _section("Observed against carried-forward tariffs"):
                _show_tariff_split(report)

        with _section("Stability of the rankings"):
            st.markdown(report["stability"]["text"])

        if report.get("potential_range"):
            with _section("How often Actual Exports fall inside the Potential Range"):
                _show_potential_range(report["potential_range"])

        if report.get("lift"):
            with _section("Do large Export Gaps point at growth? (Lift)"):
                _show_lift(report["lift"])
        if report.get("event_backtest"):
            with _section("Scenario Effects against a real Trade Agreement (Event Backtest)"):
                names = {market["iso3"]: market["name"] for market in fetch("/markets", client.api_url())}
                _show_event_backtest(report["event_backtest"], names, report.get("event_backtest_note"))
        if report.get("not_run"):
            with _section("Parts not run"):
                for part in report["not_run"]:
                    st.markdown(part["text"])

        with _section("Excluded Products"):
            _show_excluded_products(
                fetch("/products/excluded", client.api_url()), fetch("/products/excluded/reasons", client.api_url())
            )

        with _section("Known limitations"):
            for limitation in report["limitations"]:
                st.markdown(f"**{limitation['title']}.** {limitation['text']}")
