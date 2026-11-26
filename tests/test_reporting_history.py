"""The Model History's charts and tables (`reporting.history`), read from a loaded artefact, the loop's saved results
and the Test-year Ledger. Product reporting no longer defines them (ADR 0013)."""

from dataclasses import replace
from html import escape

import pytest

from exportspredictor import ledger, reporting
from exportspredictor.reporting import history

HISTORY_ONLY = (
    "ranking_by_version", "large_markets_by_version", "experiments_chart", "zero_cells_against_naive",
    "confirmation", "final_round", "final_round_probe", "round_count", "test_year_ledger", "gain_ladder",
    "traded_away", "FINAL_ROUND_IDEAS", "FINAL_ROUND_PROBE_IDEA", "FINAL_ROUND_PROBE_FILE",
)


@pytest.mark.parametrize("name", HISTORY_ONLY)
def test_each_history_only_part_lives_in_the_history_module_not_in_product_reporting(name):
    assert hasattr(history, name)
    assert name not in vars(reporting)


# --- Every version on 2023 ---------------------------------------------------------------------------------------


def test_the_ranking_by_version_holds_this_version_the_naive_baseline_and_earlier_versions(fixture_artefacts):
    block = fixture_artefacts.evaluation["earlier_versions"]

    chart = history.ranking_by_version(fixture_artefacts)

    rows = {row["name"]: row["uk_ranking"] for row in chart.data.iter_rows(named=True)}
    expected = {row["name"]: row["uk_ranking"] for row in block["rows"] + block["naive"]}
    assert rows == pytest.approx(expected)
    assert str(block["test_year"]) in chart.title


def test_the_ranking_by_version_draws_the_tie_proof_measure_once_earlier_versions_are_re_scored(fixture_artefacts):
    block = fixture_artefacts.evaluation["earlier_versions"]

    chart = history.ranking_by_version(fixture_artefacts)

    rows = {row["name"]: row["uk_ranking_tie_proof"] for row in chart.data.iter_rows(named=True)}
    expected = {row["name"]: row["uk_ranking_tie_proof"] for row in block["rows"] + block["naive"]}
    assert rows == pytest.approx(expected)
    assert "tie-proof" in chart.title


def test_a_chart_is_a_static_matplotlib_svg_image_naming_each_row(fixture_artefacts):
    chart = history.ranking_by_version(fixture_artefacts)

    svg = chart._repr_svg_()
    assert svg.lstrip().startswith(("<?xml", "<svg"))
    assert "Matplotlib" in svg  # drawn by matplotlib, not by hand
    assert "<script" not in svg  # static: nothing GitHub would refuse to render
    for name in chart.data["name"]:
        assert escape(name, quote=False) in svg  # text stays text, so labels are searchable


def test_without_earlier_versions_the_ranking_by_version_is_not_run(fixture_artefacts):
    bare = {k: v for k, v in fixture_artefacts.evaluation.items() if k != "earlier_versions"}

    assert isinstance(history.ranking_by_version(replace(fixture_artefacts, evaluation=bare)), reporting.NotRun)


def test_the_large_market_measures_by_version_list_this_version_earlier_versions_and_the_naive_baseline(
    fixture_artefacts,
):
    report = fixture_artefacts.evaluation
    v1 = {row["markets"]: row["model"] for row in report["rescored_earlier_versions"]["v1"]["large_markets"]}

    table = history.large_markets_by_version(fixture_artefacts)

    rows = {row["version"]: row for row in table.iter_rows(named=True)}
    assert rows["v1"]["30 largest Markets"] == pytest.approx(v1[30])
    assert rows["v1"]["60 largest Markets"] == pytest.approx(v1[60])
    assert table["version"][0].startswith("This version")


def test_a_version_without_large_market_measures_shows_not_run(fixture_artefacts):
    older = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "large_markets"}
    assert isinstance(history.large_markets_by_version(replace(fixture_artefacts, evaluation=older)), reporting.NotRun)


def _with_earlier(bundle, rows):
    block = {**bundle.evaluation["earlier_versions"], "rows": rows}
    return replace(bundle, evaluation={**bundle.evaluation, "earlier_versions": block})


def test_what_choosing_on_the_ranking_cost_names_the_earlier_version_with_the_lower_deviance(fixture_artefacts):
    bundle = _with_earlier(fixture_artefacts, [
        {"name": "This version", "uk_ranking": 0.60, "tweedie_deviance_all_exporters": 468.0},
        {"name": "Blend", "uk_ranking": 0.55, "tweedie_deviance_all_exporters": 434.0},
    ])

    text = history.traded_away(bundle)

    assert "468.0 for this version against 434.0 for Blend" in text


def test_nothing_is_said_to_be_traded_away_where_no_earlier_version_has_a_lower_deviance(fixture_artefacts):
    bundle = _with_earlier(fixture_artefacts, [
        {"name": "This version", "uk_ranking": 0.60, "tweedie_deviance_all_exporters": 400.0},
        {"name": "Blend", "uk_ranking": 0.55, "tweedie_deviance_all_exporters": 434.0},
    ])

    assert history.traded_away(bundle) == "No earlier version of this artefact records a lower Tweedie deviance."


# --- v2 and the gain ladder --------------------------------------------------------------------------------------


def test_v2_s_configuration_is_set_beside_the_naive_baseline_on_the_zero_cell_measures(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]

    table = history.zero_cells_against_naive(fixture_artefacts)

    v2, naive = table.iter_rows(named=True)
    assert v2["model"] == f"v2's configuration ({record['noise_floor']['arms'][0]['arm']})"
    assert naive["model"] == "Naive Baseline"
    assert naive["zero_vs_non_zero_spearman"] == pytest.approx(record["naive"]["zero_vs_non_zero_spearman"])
    assert table.columns == [
        "model", "selection_measure", "non_zero_spearman", "zero_vs_non_zero_spearman", "tweedie_deviance",
    ]


def test_without_experiment_results_the_zero_cell_comparison_is_not_run(fixture_artefacts):
    bare = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": None})

    assert isinstance(history.zero_cells_against_naive(bare), reporting.NotRun)


def test_the_gain_ladder_runs_from_v1_through_each_step_to_v2(fixture_artefacts):
    ladder = fixture_artefacts.evaluation["v1_comparison"]["ladder"]

    chart = history.gain_ladder(fixture_artefacts)

    values = {rung["name"]: rung["value"] for rung in ladder["rungs"]}
    assert chart.data["end"].to_list() == pytest.approx(
        [values["v1"], *(values[step["to"]] for step in ladder["steps"]), values["C"]]
    )
    assert chart.data["change"][1:-1].to_list() == pytest.approx([step["change"] for step in ladder["steps"]])


def test_a_version_without_the_gain_ladder_says_it_was_not_run(fixture_default_artefacts):
    assert isinstance(history.gain_ladder(fixture_default_artefacts), reporting.NotRun)


def test_the_saved_gain_ladder_draws_v2_s_ladder_from_its_committed_record():
    # Copied once from the one artefact that ran it, so the record never opens that folder.
    record = history.saved_gain_ladder_record()
    chart = history.saved_gain_ladder()

    assert record["source"].startswith("Artefact 20261118T100524Z")
    assert chart.data["end"].to_list() == pytest.approx([0.5397, 0.5193, 0.5154, 0.5115, 0.5115], abs=1e-4)
    assert chart.data["change"][1:-1].to_list() == pytest.approx([-0.0204, -0.0039, -0.0039], abs=1e-4)


# --- The experiment loop -----------------------------------------------------------------------------------------


def test_the_experiments_chart_draws_each_arm_against_the_win_threshold(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]

    chart = history.experiments_chart(fixture_artefacts)

    assert chart.data["change"].to_list() == [
        row["change"] for section in record["experiments"].values() for row in section["arms"]
    ]
    assert set(chart.data["win_threshold"]) == {record["noise_floor"]["win_threshold"]}


def test_the_experiments_chart_leaves_the_final_round_to_its_own_table(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]
    cap = {"arms": [{"arm": "cap Market Reach", "selection_measure": 0.62, "change": -0.0006, "rounds": 824}],
           "kept": None}
    recorded = {**record, "experiments": {**record["experiments"], "market reach cap": cap}}
    bundle = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": recorded})

    chart = history.experiments_chart(bundle)

    assert "cap Market Reach" not in chart.data["arm"].to_list()
    assert chart.data["arm"].to_list() == [row["arm"] for row in record["experiments"]["naive baseline"]["arms"]]


def test_without_experiment_results_the_experiments_chart_is_not_run(fixture_artefacts):
    bare = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": None})

    assert isinstance(history.experiments_chart(bare), reporting.NotRun)


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


def test_the_confirmation_against_v2_is_shown_fold_by_fold(fixture_artefacts):
    record = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": CONFIRMATION}
    bundle = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": record})

    table = history.confirmation(bundle)

    assert table["fold"].to_list() == ["0", "1", "all"]
    assert table["winner"].to_list() == [0.57, 0.55, 0.56]
    assert table["v2"].to_list() == [0.53, 0.52, 0.52]
    assert table["difference"].to_list() == pytest.approx([0.04, 0.03, 0.04])
    assert set(table["win_threshold"]) == {0.009}


def test_a_version_without_the_confirmation_says_it_was_not_run(fixture_artefacts):
    assert isinstance(history.confirmation(fixture_artefacts), reporting.NotRun)


def test_the_round_count_of_a_version_built_by_the_loop_was_chosen_on_the_selection_measure(fixture_artefacts):
    scaling = fixture_artefacts.metadata["round_scaling"]

    table = history.round_count(fixture_artefacts)

    served = table.row(0, named=True)
    assert served["model"] == "served"
    assert served["chosen_on"] == "Selection Measure"
    assert (served["rounds_chosen"], served["rounds_trained"]) == (scaling["selected_rounds"], scaling["rounds"])
    assert table.filter(model="next rebuild")["rounds_chosen"].to_list() == [scaling["selected_rounds"]]


def test_a_version_trained_at_the_searchs_round_count_discloses_that_deviance_chose_it(fixture_artefacts):
    # The Blend's artefact (20261118T185910Z): the search's 1,198 rounds, scaled to 1,453, and a five-fold
    # confirmation whose per-fold rounds the next rebuild takes the median of.
    record = fixture_artefacts.metadata["ranking_experiments"]
    confirmed = {"settings": {"winner": record["best"]["settings"]}, "winner": {"rounds": [951, 1075, 1097, 1303, 1120]}}
    metadata = {
        **fixture_artefacts.metadata,
        "rounds": 1453,
        "round_scaling": {"tuned_rounds": 1198, "ratio": 1.2127, "rounds": 1453},
        "ranking_experiments": {**record, "confirmation": confirmed},
    }

    table = history.round_count(replace(fixture_artefacts, metadata=metadata))

    assert table.to_dicts() == [
        {"model": "served", "chosen_on": "Tweedie deviance (the hyperparameter search)", "rounds_chosen": 1198,
         "rounds_trained": 1453},
        {"model": "next rebuild", "chosen_on": "Selection Measure", "rounds_chosen": 1097, "rounds_trained": None},
    ]


# --- Every read of 2023 ------------------------------------------------------------------------------------------


def test_the_ledger_table_is_every_row_of_the_committed_test_year_ledger():
    shown = history.test_year_ledger()

    assert shown.height == ledger.read_ledger().height
    assert "record" not in shown.columns


# --- The final round: one row per idea, read from the loop's saved results and the unrecorded-preferences probe --


def _section(base: float, arms: dict[str, float], kept: str | None = None) -> dict:
    return {
        "base": {"selection_measure": base},
        "base_arm": {"arm": "best so far as served"},
        "arms": [{"arm": arm, "selection_measure": base + change, "change": change} for arm, change in arms.items()],
        "kept": kept,
        "win_threshold": 0.0089,
        "decided": True,
    }


def _results() -> dict:
    return {"experiments": {
        "naive baseline": _section(0.53, {"blend": 0.04}, kept="blend"),  # before the round: not shown
        "market reach cap": _section(0.62, {"cap Market Reach": -0.0006}),
        "hyperparameters: depth": _section(0.62, {"max_depth 8": -0.0002, "max_depth 14": 0.0004}),
        "hyperparameters: subsampling": _section(0.62, {"colsample_bytree 0.6": 0.0005}),
        "model family: lightgbm": _section(0.62, {"LightGBM": -0.0005}),
        "model family: catboost": _section(0.62, {"CatBoost": -0.0019}),
    }}


def test_each_idea_of_the_round_is_one_row_with_its_best_arm_against_the_win_threshold():
    table = history.final_round(_results())

    rows = {row["idea"]: row for row in table.iter_rows(named=True)}
    assert list(rows) == [
        "Cap on Market Reach's upside (#163)", "A fresh hyperparameter search (#166)", "Other model families (#167)",
    ]
    search = rows["A fresh hyperparameter search (#166)"]
    assert search["experiments"] == 2
    assert search["arms"] == 3
    assert search["best_arm"] == "colsample_bytree 0.6"
    assert search["gain"] == pytest.approx(0.0005)
    assert search["best"] == pytest.approx(0.6205)
    assert search["threshold"] == pytest.approx(0.0089)
    assert search["decision"] == "dropped"
    assert rows["Other model families (#167)"]["best_arm"] == "LightGBM"


def test_a_kept_arm_is_reported_as_kept():
    results = _results()
    results["experiments"]["market reach cap"]["kept"] = "cap Market Reach"

    table = history.final_round(results)

    assert table.row(0, named=True)["decision"] == "kept"


def test_the_probe_is_a_row_decided_against_its_gate_not_the_win_threshold():
    probe = {"base": {"selection_measure": 0.62099}, "probe": {"selection_measure": 0.62089}, "gain": -0.0001,
             "gate": 0.0044, "retrained_arm_runs": False}

    table = history.final_round(_results(), probe)

    row = table.row(-1, named=True)
    assert row["idea"] == "Tariff 0 where an agreement's preference is unrecorded (#171)"
    assert row["gain"] == pytest.approx(-0.0001)
    assert row["threshold"] == pytest.approx(0.0044)
    # The row says which bar it is judged against, so its threshold is not read as the Win Threshold.
    assert row["judged_against"] == "its pre-registered gate"
    assert row["decision"] == "not retrained (missed its gate)"
    assert {r["judged_against"] for r in table.head(-1).iter_rows(named=True)} == {"Win Threshold"}


def test_without_any_idea_of_the_round_it_is_not_run():
    results = {"experiments": {"naive baseline": _section(0.53, {"blend": 0.04}, kept="blend")}}

    assert isinstance(history.final_round(results), reporting.NotRun)


def test_the_probe_is_read_from_its_saved_file_or_is_none(tmp_path):
    assert history.final_round_probe(tmp_path) is None

    saved = tmp_path / "derived" / "final_round_2022"
    saved.mkdir(parents=True)
    (saved / "unrecorded_preferences.json").write_text('{"gain": -0.0001}', encoding="utf-8")

    assert history.final_round_probe(tmp_path) == {"gain": -0.0001}


def test_the_earlier_versions_table_holds_every_row_and_both_naive_baselines(fixture_artefacts):
    block = fixture_artefacts.evaluation["earlier_versions"]

    table = history.earlier_versions(fixture_artefacts)

    assert table.height == len(block["rows"]) + len(block["naive"])
    assert history.earlier_versions(fixture_artefacts, only="v1")["name"].to_list() == ["v1"]
