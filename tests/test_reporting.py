"""The reporting module: one chart or table per main result, read from a loaded artefact (spec #66, #88).

Tested on the fixture artefact for what each function returns (the rows behind a chart, the
table's rows, "not run" where a part is absent), not for how a chart looks.
"""

from dataclasses import replace
from html import escape

import polars as pl
import pytest

from exportspredictor import reporting
from exportspredictor.caveats import CaveatContext, with_caveat_columns
from exportspredictor.exclusions import SECTOR_71_REASON
from exportspredictor.volatility import ONE_OFF_REASON, VOLATILE_REASON


def test_the_ranking_by_version_holds_this_version_the_naive_baseline_and_earlier_versions(fixture_artefacts):
    block = fixture_artefacts.evaluation["earlier_versions"]

    chart = reporting.ranking_by_version(fixture_artefacts)

    rows = {row["name"]: row["uk_ranking"] for row in chart.data.iter_rows(named=True)}
    expected = {row["name"]: row["uk_ranking"] for row in block["rows"] + block["naive"]}
    assert rows == pytest.approx(expected)
    assert str(block["test_year"]) in chart.title


def test_the_ranking_by_version_draws_the_tie_proof_measure_once_earlier_versions_are_re_scored(fixture_artefacts):
    block = fixture_artefacts.evaluation["earlier_versions"]

    chart = reporting.ranking_by_version(fixture_artefacts)

    rows = {row["name"]: row["uk_ranking_tie_proof"] for row in chart.data.iter_rows(named=True)}
    expected = {row["name"]: row["uk_ranking_tie_proof"] for row in block["rows"] + block["naive"]}
    assert rows == pytest.approx(expected)
    assert "tie-proof" in chart.title


def test_a_chart_is_a_static_matplotlib_svg_image_naming_each_row(fixture_artefacts):
    chart = reporting.ranking_by_version(fixture_artefacts)

    svg = chart._repr_svg_()
    assert svg.lstrip().startswith(("<?xml", "<svg"))
    assert "Matplotlib" in svg  # drawn by matplotlib, not by hand
    assert "<script" not in svg  # static: nothing GitHub would refuse to render
    for name in chart.data["name"]:
        assert escape(name, quote=False) in svg  # text stays text, so labels are searchable


def test_a_chart_is_small_enough_for_a_committed_notebook_output(fixture_artefacts):
    chart = reporting.event_backtests(fixture_artefacts)

    assert len(chart._repr_svg_().encode()) < reporting.CHART_LIMIT_BYTES


def test_the_experiments_table_has_every_arm_and_whether_it_was_kept(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]

    table = reporting.experiments(fixture_artefacts)

    arms = [row["arm"] for section in record["experiments"].values() for row in section["arms"]]
    assert list(table["arm"]) == arms
    # The fixture's loop ran the blend alone: the research note's experiment 1, in the Glossary's words.
    assert set(table["experiment"]) == {"1. Naive Baseline"}
    kept = [section["kept"] for section in record["experiments"].values()]
    for row in table.iter_rows(named=True):
        assert row["kept"] == (row["arm"] in kept)
        assert row["stopped_early"] == (row["rounds"] < reporting.EARLY_STOP_ROUNDS)


def test_the_experiments_chart_draws_each_arm_against_the_win_threshold(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]

    chart = reporting.experiments_chart(fixture_artefacts)

    assert chart.data["change"].to_list() == [
        row["change"] for section in record["experiments"].values() for row in section["arms"]
    ]
    assert set(chart.data["win_threshold"]) == {record["noise_floor"]["win_threshold"]}


def test_without_experiment_results_the_experiments_are_not_run(fixture_artefacts):
    bare = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": None})

    assert isinstance(reporting.experiments(bare), reporting.NotRun)
    assert isinstance(reporting.experiments_chart(bare), reporting.NotRun)
    assert "not run" in repr(reporting.experiments(bare))


CONFIRMATION = {
    "n_folds": 2,
    "difference": 0.04,
    "win_threshold": 0.009,
    "beats_v2": True,
    "winner": {"selection_measure": 0.56},
    "v2": {"selection_measure": 0.52},
    "naive": {"selection_measure": 0.50},
    "folds": [
        {"fold": 0, "winner": {"selection_measure": 0.57}, "v2": {"selection_measure": 0.53}},
        {"fold": 1, "winner": {"selection_measure": 0.55}, "v2": {"selection_measure": 0.52}},
    ],
}


def test_the_confirmation_is_shown_fold_by_fold(fixture_artefacts):
    record = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": CONFIRMATION}
    bundle = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": record})

    table = reporting.confirmation(bundle)

    assert table["fold"].to_list() == ["0", "1", "all"]
    assert table["winner"].to_list() == [0.57, 0.55, 0.56]
    assert table["v2"].to_list() == [0.53, 0.52, 0.52]
    assert table["difference"].to_list() == pytest.approx([0.04, 0.03, 0.04])


def test_a_version_without_the_confirmation_says_it_was_not_run(fixture_artefacts):
    assert isinstance(reporting.confirmation(fixture_artefacts), reporting.NotRun)


def test_lift_by_tenth_reads_each_tenths_growth_rate_beside_the_base_rate(fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]

    chart = reporting.lift_by_tenth(fixture_artefacts)

    assert chart.data["tenth"].to_list() == [row["decile"] for row in lift["by_decile"]]
    assert chart.data["rate"].to_list() == [row["rate"] for row in lift["by_decile"]]
    assert set(chart.data["base_rate"]) == {lift["base_rate"]}


def test_the_event_backtests_set_each_predicted_change_beside_the_observed_change(fixture_artefacts):
    events = fixture_artefacts.evaluation["event_backtest"]["events"]

    chart = reporting.event_backtests(fixture_artefacts)

    pooled = chart.data.filter(year="pooled")
    expected = [direction["pooled"] for event in events for direction in event["directions"]]
    assert pooled["predicted"].to_list() == [row["predicted"] for row in expected]
    assert pooled["observed"].to_list() == [row["observed"] for row in expected]
    assert pooled["observed_low"].to_list() == [row["observed_ci"][0] for row in expected]
    by_year = sum(len(direction["by_year"]) for event in events for direction in event["directions"])
    assert chart.data.height == by_year + len(expected)


def test_a_report_without_lift_or_event_backtests_says_they_were_not_run(fixture_artefacts):
    report = {k: v for k, v in fixture_artefacts.evaluation.items() if k not in ("lift", "event_backtest")}
    bare = replace(fixture_artefacts, evaluation=report)

    assert isinstance(reporting.lift_by_tenth(bare), reporting.NotRun)
    assert isinstance(reporting.event_backtests(bare), reporting.NotRun)


def test_the_ppml_table_reads_the_saved_benchmark(fixture_artefacts):
    saved = fixture_artefacts.ppml_benchmark

    table = reporting.ppml_benchmark(fixture_artefacts)

    assert table["effect_pct"][0] == pytest.approx(saved["ppml"]["effect_pct"])
    assert table["low_pct"][0] == pytest.approx(saved["ppml"]["effect_low_pct"])
    assert table["effect_pct"][1] == pytest.approx(saved["model"]["effect_pct"])
    assert isinstance(reporting.ppml_benchmark(replace(fixture_artefacts, ppml_benchmark=None)), reporting.NotRun)


def test_the_gain_ladder_runs_from_v1_through_each_step_to_v2(fixture_artefacts):
    ladder = fixture_artefacts.evaluation["v1_comparison"]["ladder"]

    chart = reporting.gain_ladder(fixture_artefacts)

    values = {rung["name"]: rung["value"] for rung in ladder["rungs"]}
    assert chart.data["end"].to_list() == pytest.approx(
        [values["v1"], *(values[step["to"]] for step in ladder["steps"]), values["C"]]
    )
    assert chart.data["change"][1:-1].to_list() == pytest.approx([step["change"] for step in ladder["steps"]])


def test_the_provenance_names_the_version_and_its_source(fixture_artefacts):
    text = reporting.provenance(fixture_artefacts)

    assert fixture_artefacts.version in text
    assert "source: fixture" in text


def test_the_round_count_of_a_version_built_by_the_loop_was_chosen_on_the_selection_measure(fixture_artefacts):
    scaling = fixture_artefacts.metadata["round_scaling"]

    table = reporting.round_count(fixture_artefacts)

    served = table.row(0, named=True)
    assert served["model"] == "served"
    assert served["chosen_on"] == "Selection Measure"
    assert (served["rounds_chosen"], served["rounds_trained"]) == (scaling["selected_rounds"], scaling["rounds"])
    assert table.filter(model="next rebuild")["rounds_chosen"].to_list() == [scaling["selected_rounds"]]


def test_a_version_trained_at_the_searchs_round_count_discloses_that_deviance_chose_it(fixture_artefacts):
    # The served artefact's shape (20261118T185910Z): the search's 1,198 rounds, scaled to 1,453, and a
    # five-fold confirmation whose per-fold rounds the next rebuild takes the median of.
    record = fixture_artefacts.metadata["ranking_experiments"]
    confirmed = {"settings": {"winner": record["best"]["settings"]}, "winner": {"rounds": [951, 1075, 1097, 1303, 1120]}}
    metadata = {
        **fixture_artefacts.metadata,
        "rounds": 1453,
        "round_scaling": {"tuned_rounds": 1198, "ratio": 1.2127, "rounds": 1453},
        "ranking_experiments": {**record, "confirmation": confirmed},
    }

    table = reporting.round_count(replace(fixture_artefacts, metadata=metadata))

    assert table.to_dicts() == [
        {"model": "served", "chosen_on": "Tweedie deviance (the hyperparameter search)", "rounds_chosen": 1198,
         "rounds_trained": 1453},
        {"model": "next rebuild", "chosen_on": "Selection Measure", "rounds_chosen": 1097, "rounds_trained": None},
    ]


def test_provenance_lines_are_read_back_from_printed_outputs(fixture_artefacts):
    printed = f"other output\n{reporting.provenance(fixture_artefacts)}\nmore output"

    assert reporting.provenance_versions(printed) == [(fixture_artefacts.version, fixture_artefacts.metadata["source"])]
    assert reporting.provenance_versions("no provenance here") == []


def test_the_model_card_excerpt_is_its_headline_and_its_contents_with_a_link_to_the_whole(fixture_artefacts, tmp_path):
    from exportspredictor import governance

    path = governance.write_model_card(fixture_artefacts, tmp_path)
    card = path.read_text(encoding="utf-8")

    excerpt = reporting.model_card_excerpt(path)

    headings = [line for line in card.splitlines() if line.startswith("## ")]
    assert len(excerpt) < 4_000 < len(card)  # "Outputs stay small" (spec #66)
    assert "## Headline" in excerpt and card.split("## Headline", 1)[1].split("\n## ", 1)[0].strip() in excerpt
    for heading in headings:
        assert heading[3:] in excerpt, heading
    assert str(path.name) in excerpt
    assert "## Limitations" not in excerpt  # later sections are listed, not shown


def test_each_caveat_tier_s_reach_counts_its_opportunities_overall_and_in_the_top_of_the_ranking(fixture_artefacts):
    baseline = pl.DataFrame({
        "hs6": ["220830", "220830", "870323", "711319", "300490"],
        "iso3": ["SAU", "RUS", "RUS", "HKG", "JPN"],
        "export_gap_usd": [500.0, 400.0, 300.0, -50.0, 100.0],
    })
    bundle = replace(fixture_artefacts, baseline=with_caveat_columns(baseline, CaveatContext(baseline_year=2024)))

    reach = reporting.caveat_tier_reach(bundle, tops=(1, 3))

    assert reach.columns == ["Caveat Tier", "Opportunities", "in top 1", "in top 3"]
    assert reach.rows() == [
        ("Flagged Product", 0, 0, 0),  # the jewellery to Hong Kong is not an Opportunity
        ("Re-export Hub", 0, 0, 0),
        ("Sanctioned Market", 2, 0, 2),
        ("Restricted Pair", 1, 1, 1),
    ]


def test_flagged_products_are_counted_by_reason(fixture_artefacts):
    baseline = pl.DataFrame({
        "hs6": ["711319", "711319", "300490", "300490", "880230", "870323"],
        "iso3": ["HKG", "JPN", "HKG", "JPN", "QAT", "JPN"],
        "export_gap_usd": [-50.0, 10.0, 100.0, 200.0, 300.0, 400.0],
    })
    reasons = {"300490": VOLATILE_REASON, "880230": ONE_OFF_REASON}
    marked = with_caveat_columns(baseline, CaveatContext(baseline_year=2024, flagged_product_reasons=reasons))

    counts = reporting.flagged_products_by_reason(replace(fixture_artefacts, baseline=marked))

    assert counts.rows() == [
        (VOLATILE_REASON, 1, 2),
        (SECTOR_71_REASON, 1, 1),
        (ONE_OFF_REASON, 1, 1),
    ]
    assert counts.columns == ["Reason", "Products", "Opportunities"]


def test_the_large_market_measures_by_version_list_this_version_earlier_versions_and_the_naive_baseline(
    fixture_artefacts,
):
    report = fixture_artefacts.evaluation
    v1 = {row["markets"]: row["model"] for row in report["rescored_earlier_versions"]["v1"]["large_markets"]}

    table = reporting.large_markets_by_version(fixture_artefacts)

    rows = {row["version"]: row for row in table.iter_rows(named=True)}
    assert rows["v1"]["30 largest Markets"] == pytest.approx(v1[30])
    assert rows["v1"]["60 largest Markets"] == pytest.approx(v1[60])
    assert table["version"][0].startswith("This version")


def test_a_version_without_large_market_measures_shows_not_run(fixture_artefacts):
    older = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "large_markets"}
    assert isinstance(reporting.large_markets_by_version(replace(fixture_artefacts, evaluation=older)), reporting.NotRun)


def test_what_market_reach_changes_is_the_saved_sentence_or_a_not_run_line(fixture_artefacts):
    saved = fixture_artefacts.evaluation["market_reach_effect"]["text"]
    without = {k: v for k, v in fixture_artefacts.evaluation.items() if k != "market_reach_effect"}

    assert reporting.market_reach_effect(fixture_artefacts) == saved
    assert reporting.market_reach_effect(replace(fixture_artefacts, evaluation=without)) == (
        "Market Reach: not run for this version."
    )
