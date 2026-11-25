"""The pre-registered v1/v2 comparison, on a synthetic frozen-v1 folder.

v2's scores here are a stand-in (Actual Exports with seeded noise), so these tests check the
protocol (common rows, bootstrap, verdict, manifest), not either model.
"""

import json
import shutil

import numpy as np
import polars as pl
import pytest

from exportspredictor import comparison, freeze
from exportspredictor.exclusions import is_excluded
from exportspredictor.fixture import UK
from exportspredictor.paths import artefacts_dir

TEST_YEAR = 2023
KEYS = ["exporter", "importer", "hs6"]


@pytest.fixture(scope="module")
def frozen(v1_frozen):
    return comparison.load_frozen(v1_frozen)


@pytest.fixture(scope="module")
def scope(fixture_artefacts):
    return fixture_artefacts.metadata["exporter_scope"]


@pytest.fixture(scope="module")
def flows(fixture_world):
    return fixture_world.flows.filter(pl.col("year") == TEST_YEAR)


def _v2_scores(keys: pl.DataFrame, *, noise: float, seed: int) -> pl.DataFrame:
    """A stand-in for v2's Export Potential on the common keys: Actual Exports times seeded noise."""
    draw = np.random.default_rng(seed).lognormal(0.0, noise, keys.height)
    return keys.select(*KEYS, export_potential_usd=(pl.col("value_usd") + 1_000.0) * pl.Series(draw))


@pytest.fixture(scope="module")
def v2(frozen, scope):
    keys = comparison.common_keys(frozen, scope=scope, uk=UK)
    return {name: _v2_scores(rows, noise=0.3, seed=1) for name, rows in keys.items()}


def test_v1_v2_and_the_naive_baseline_are_scored_on_identical_keys(frozen, v2, flows, scope):
    keys = comparison.common_keys(frozen, scope=scope, uk=UK)
    scored = comparison.scored_rows(frozen, v2=v2, flows=flows, scope=scope, uk=UK)

    assert set(scored) == {"uk", "all_exporters"}
    for name, rows in scored.items():
        assert rows.height == keys[name].height > 0, name
        assert rows.select(KEYS).sort(KEYS).equals(keys[name].select(KEYS).sort(KEYS)), name
        assert rows.select("v1_usd", "v2_usd", "naive_usd").null_count().sum_horizontal().item() == 0, name
        # v1 had neither Excluded Products nor an Exporter scope; the common rows have both.
        assert not rows.select(is_excluded().any()).item(), name
        assert set(rows["exporter"].unique()) <= set(scope), name
    assert set(scored["uk"]["exporter"].unique()) == {UK}
    # The all-Exporters rows keep v1's own Zero Cell weights.
    assert scored["all_exporters"].filter(pl.col("value_usd") == 0)["weight"].unique().to_list() == [2.0]


def test_the_common_rows_drop_what_v1_scored_outside_v2s_scope_and_its_excluded_products(frozen, scope):
    test_rows = frozen.tables["test_rows"]
    keys = comparison.common_keys(frozen, scope=scope, uk=UK)

    assert test_rows.filter(~pl.col("exporter").is_in(scope)).height > 0
    assert test_rows.select(is_excluded().any()).item()
    expected = test_rows.filter(pl.col("exporter").is_in(scope) & ~is_excluded()).height
    assert keys["all_exporters"].height == expected


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_v2_scores_on_other_keys_are_refused(frozen, v2, flows, scope, change):
    uk = v2["uk"]
    altered = {
        "missing": uk.slice(1),
        "extra": pl.concat([uk, uk.head(1).with_columns(importer=pl.lit("ZZZ"))]),
        "duplicate": pl.concat([uk, uk.head(1)]),
    }[change]

    with pytest.raises(ValueError, match="common"):
        comparison.scored_rows(frozen, v2={**v2, "uk": altered}, flows=flows, scope=scope, uk=UK)


def test_a_frozen_table_that_no_longer_matches_its_manifest_is_refused(v1_frozen, tmp_path):
    copy = tmp_path / freeze.FOLDER
    shutil.copytree(v1_frozen, copy)
    path = copy / freeze.TABLES["uk_grid"]
    table = pl.read_parquet(path)
    table.with_columns(pl.col("export_potential_usd") * 2).write_parquet(path)

    with pytest.raises(ValueError, match="manifest"):
        comparison.load_frozen(copy)


def _compare(frozen, v2, fixture_world, fixture_artefacts, *, seed=0):
    from exportspredictor import pipeline

    report = fixture_artefacts.evaluation
    return comparison.compare(
        frozen,
        v2=v2,
        flows=fixture_world.flows,
        lift_v2=report["lift"],
        event_v2=report["event_backtest"]["events"][0],
        lift_settings=pipeline.FIXTURE_BACKTESTS.lift,
        scope=fixture_artefacts.metadata["exporter_scope"],
        uk=UK,
        test_year=TEST_YEAR,
        seed=seed,
        resamples=200,
    )


def test_the_paired_bootstrap_is_deterministic_under_a_seed(frozen, v2, fixture_world, fixture_artefacts):
    first = _compare(frozen, v2, fixture_world, fixture_artefacts, seed=7)
    again = _compare(frozen, v2, fixture_world, fixture_artefacts, seed=7)
    other = _compare(frozen, v2, fixture_world, fixture_artefacts, seed=8)

    assert first == again
    assert first["primary"]["ci"][0] < first["primary"]["difference"] < first["primary"]["ci"][1]
    assert other["primary"]["difference"] == first["primary"]["difference"]
    assert other != first  # another seed draws other resamples (the fixture has few Products, so some intervals coincide)


def test_a_v2_that_ranks_like_actual_exports_beats_noisy_v1_on_the_primary_metric(
    frozen, fixture_world, fixture_artefacts, scope
):
    keys = comparison.common_keys(frozen, scope=scope, uk=UK)
    exact = {name: _v2_scores(rows, noise=0.0, seed=0) for name, rows in keys.items()}

    result = _compare(frozen, exact, fixture_world, fixture_artefacts)

    primary = result["primary"]
    assert primary["v2"] > primary["v1"]
    assert primary["ci"][0] > 0
    assert result["verdict"]["conditions"]["primary_interval_above_zero"]
    assert result["verdict"]["conditions"]["larger_margin_over_naive"]
    assert result["rows"] == {name: rows.height for name, rows in keys.items()}


def test_v1_is_also_reported_on_all_products(frozen, v2, fixture_world, fixture_artefacts):
    result = _compare(frozen, v2, fixture_world, fixture_artefacts)

    uk = result["v1_all_products"]["uk"]
    assert uk["excluded_rows"] > 0
    assert uk["all_products"]["rows"] == uk["common_rows"]["rows"] + uk["excluded_rows"]
    assert uk["all_products"]["tweedie_deviance"] != uk["common_rows"]["tweedie_deviance"]


# --- The verdict: a constructed case for each failing condition, and one passing case -----------------


def _primary(*, v1=0.50, v2=0.60, naive=0.40, ci=(0.02, 0.15)):
    return {**comparison.PRIMARY, "v1": v1, "v2": v2, "naive": naive, "difference": v2 - v1, "ci": list(ci)}


def _secondary(*, worse=False, counts=True, label="Tweedie deviance, all Exporters"):
    return {"metric": "tweedie_deviance", "label": label, "v1": 300.0, "v2": 290.0, "worse": worse, "counts": counts}


def test_v2_beats_v1_when_all_three_conditions_hold():
    result = comparison.verdict(_primary(), [_secondary(), _secondary(worse=True, counts=False, label="Raw Lift")])

    assert result["v2_beats_v1"]
    assert all(result["conditions"].values())
    assert result["text"].startswith("v2 beats v1")


def test_an_interval_that_reaches_zero_is_a_mixed_result():
    result = comparison.verdict(_primary(ci=(-0.01, 0.15)), [_secondary()])

    assert not result["v2_beats_v1"]
    assert result["conditions"] == {
        "primary_interval_above_zero": False, "larger_margin_over_naive": True, "no_secondary_worse": True
    }
    assert result["text"].startswith("The result is mixed")


def test_a_margin_over_the_naive_baseline_no_larger_than_v1s_is_a_mixed_result():
    result = comparison.verdict(_primary(v1=0.60, v2=0.60, ci=(0.01, 0.05)), [_secondary()])

    assert not result["v2_beats_v1"]
    assert result["conditions"] == {
        "primary_interval_above_zero": True, "larger_margin_over_naive": False, "no_secondary_worse": True
    }
    assert result["margins_over_naive"] == {"v1": pytest.approx(0.2), "v2": pytest.approx(0.2)}


def test_a_secondary_metric_worse_beyond_its_interval_is_a_mixed_result_named_in_words():
    result = comparison.verdict(_primary(), [_secondary(worse=True)])

    assert not result["v2_beats_v1"]
    assert result["conditions"] == {
        "primary_interval_above_zero": True, "larger_margin_over_naive": True, "no_secondary_worse": False
    }
    assert "Tweedie deviance, all Exporters: v1 300.000, v2 290.000 (worse beyond its interval)" in result["text"]


def test_a_v2_far_off_on_all_exporters_is_worse_beyond_the_deviance_interval(frozen, v2, fixture_world, fixture_artefacts):
    off = v2["all_exporters"].with_columns(export_potential_usd=pl.lit(1e12))

    result = _compare(frozen, {**v2, "all_exporters": off}, fixture_world, fixture_artefacts)

    deviance = next(entry for entry in result["secondary"] if entry["metric"] == "tweedie_deviance")
    assert deviance["worse"] and deviance["ci"][0] > 0
    assert not result["verdict"]["conditions"]["no_secondary_worse"]
    assert not result["verdict"]["v2_beats_v1"]


def test_the_fixture_evaluation_holds_the_comparison_on_the_fixtures_common_rows(fixture_artefacts, fixture_world):
    from exportspredictor import pipeline

    block = fixture_artefacts.evaluation["v1_comparison"]
    keys = comparison.common_keys(
        pipeline.fixture_v1_frozen(fixture_world), scope=fixture_artefacts.metadata["exporter_scope"], uk=UK
    )

    assert block["rows"] == {name: rows.height for name, rows in keys.items()}
    assert block["test_year"] == TEST_YEAR
    assert block["lift"]["held_out_exporters"] == fixture_artefacts.evaluation["lift"]["held_out_exporters"]
    counted_events = [e for e in block["secondary"] if e["metric"] == "event_distance" and e["counts"]]
    assert counted_events and {e["year"] for e in counted_events} == {pipeline.FIXTURE_BACKTESTS.event.event_year}
    assert set(block["verdict"]["conditions"]) == set(comparison.CONDITIONS)


# --- The gain ladder ---------------------------------------------------------------------------


def test_v1s_features_are_v2s_without_relatedness_market_trends_and_tariffs():
    from exportspredictor.features import FEATURES

    assert comparison.V1_FEATURES == [
        "log_gdp_o", "log_gdp_d", "log_population_o", "log_population_d", "log_distance", "contiguous",
        "common_language", "colonial_tie", "common_colonizer", "log_supply_loo", "rca_loo", "log_demand_loo",
        "high_income_share_loo", "sector", "trade_agreement", "customs_union", "eu_o", "eu_d", "cptpp_o", "cptpp_d",
        "efta_o", "efta_d",
    ]
    assert set(comparison.V1_FEATURES) < set(FEATURES)


def test_v1s_features_are_those_of_the_real_v1_artefact():
    import json

    metadata = freeze.v1_folder().parent / "metadata.json"
    if not metadata.exists():
        pytest.skip("The real v1 artefact is not on disk")
    v1 = json.loads(metadata.read_text(encoding="utf-8"))
    assert comparison.V1_FEATURES == v1["features"]
    assert comparison.REAL_LADDER.params == v1["params"]
    assert comparison.REAL_LADDER.rounds == v1["rounds"]


def test_a_rung_trained_on_v1s_features_scores_rows_that_hold_every_feature(fixture_world, scope, tmp_path):
    from exportspredictor import evaluation, pipeline

    training = pipeline.fixture_training(fixture_world, exporters=scope)
    rung = evaluation.train_test_models(
        training, test_year=TEST_YEAR, uk=UK, params={"max_depth": 2}, rounds=5, n_folds=2, device="cpu",
        features=comparison.V1_FEATURES,
    )
    assert rung.features == comparison.V1_FEATURES
    assert all(model.feature_names == comparison.V1_FEATURES for model in rung.models)
    again = evaluation.TestModels.load(rung.save(tmp_path / "rung"))
    assert again.features == comparison.V1_FEATURES

    rows = training[TEST_YEAR - 1].with_columns(year=pl.lit(TEST_YEAR, training[TEST_YEAR - 1]["year"].dtype))
    scored = evaluation.score_rows(rows, again.models, again.transform, UK, features=again.features)
    assert scored.height == rows.height
    assert scored["export_potential_usd"].null_count() == 0


def _ladder_compare(frozen, v2, fixture_world, fixture_artefacts, rungs, *, seed=0):
    from exportspredictor import pipeline

    report = fixture_artefacts.evaluation
    return comparison.compare(
        frozen, v2=v2, flows=fixture_world.flows, lift_v2=report["lift"], event_v2=report["event_backtest"]["events"][0],
        lift_settings=pipeline.FIXTURE_BACKTESTS.lift, scope=fixture_artefacts.metadata["exporter_scope"], uk=UK,
        test_year=TEST_YEAR, seed=seed, resamples=200, ladder=rungs,
    )


@pytest.fixture(scope="module")
def rungs(frozen, scope):
    uk = comparison.common_keys(frozen, scope=scope, uk=UK)["uk"]
    return {"A": _v2_scores(uk, noise=1.5, seed=3), "B": _v2_scores(uk, noise=0.1, seed=4)}


def test_the_ladder_runs_from_v1_through_a_and_b_to_v2_and_its_steps_add_up(
    frozen, v2, rungs, fixture_world, fixture_artefacts
):
    result = _ladder_compare(frozen, v2, fixture_world, fixture_artefacts, rungs)
    ladder, primary = result["ladder"], result["primary"]

    assert [r["name"] for r in ladder["rungs"]] == ["v1", "A", "B", "C"]
    values = {r["name"]: r["value"] for r in ladder["rungs"]}
    assert values["v1"] == pytest.approx(primary["v1"])
    assert values["C"] == pytest.approx(primary["v2"])
    # B's scores are close to Actual Exports, A's far: B ranks Markets better than A.
    assert values["B"] > values["A"]
    assert [(s["from"], s["to"]) for s in ladder["steps"]] == [("v1", "A"), ("A", "B"), ("B", "C")]
    assert sum(s["change"] for s in ladder["steps"]) == pytest.approx(primary["difference"])
    for step in ladder["steps"]:
        low, high = step["ci"]
        assert low <= step["change"] <= high
    assert "decides nothing" in ladder["note"] and "order" in ladder["note"]
    # The ladder decides nothing: the verdict is the same with or without it.
    assert result["verdict"] == _compare(frozen, v2, fixture_world, fixture_artefacts)["verdict"]


def test_ladder_scores_on_other_keys_are_refused(frozen, v2, rungs, fixture_world, fixture_artefacts):
    short = {**rungs, "A": rungs["A"].slice(1)}
    with pytest.raises(ValueError, match="A"):
        _ladder_compare(frozen, v2, fixture_world, fixture_artefacts, short)


def test_the_fixture_evaluation_holds_the_gain_ladder_of_two_extra_test_year_models(fixture_artefacts):
    block = fixture_artefacts.evaluation["v1_comparison"]
    ladder = block["ladder"]

    assert [r["name"] for r in ladder["rungs"]] == ["v1", "A", "B", "C"]
    assert ladder["rungs"][-1]["value"] == pytest.approx(block["primary"]["v2"])
    assert ladder["v1_configuration"]["rounds"] == comparison.FIXTURE_LADDER.rounds
    assert ladder["v1_configuration"]["features"]["A"] == comparison.V1_FEATURES


# --- Naming the new version: the comparison was written when v2 was the new version ----------------


def test_the_verdict_names_the_new_version_it_is_given():
    result = comparison.verdict(_primary(), [_secondary(worse=True)], name="the Blend")

    assert "the Blend's Export Potential ranks Markets" in result["text"]
    assert "v1 300.000, the Blend 290.000" in result["text"]
    assert "v2" not in result["text"]
    assert "Naive Baseline" in result["text"] and "naive baseline" not in result["text"]


def test_a_saved_comparison_is_relabelled_from_its_saved_figures():
    block = {"primary": _primary(), "secondary": [_secondary()], "secondaries_not_run": ["Lift"],
             "verdict": {"text": "v2 beats v1 under the pre-registered rule ..."}}

    relabelled = comparison.relabelled({"model_form": {"form": "blend", "naive_weight": 0.3}, "v1_comparison": block})

    text = relabelled["v1_comparison"]["verdict"]["text"]
    assert text.startswith("The Blend beats v1")
    assert "The Lift secondaries were not run" in text
    assert block["verdict"]["text"].startswith("v2")  # the saved report is not changed


# The served real artefact, whose comparison was saved while v2 was the name of the new version.
REAL_REPORT = artefacts_dir() / "20261119T001736Z" / "evaluation.json"


@pytest.mark.skipif(not REAL_REPORT.exists(), reason="the real artefact 20261119T001736Z is absent")
def test_relabelling_the_real_comparison_as_v2_reproduces_its_saved_text_bar_the_naive_baseline_s_casing():
    report = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    block = report["v1_comparison"]

    again = comparison.saved_verdict_text(block, name="v2")

    assert again == block["verdict"]["text"].replace("naive baseline", "Naive Baseline")


def test_the_v1_comparison_says_it_scores_the_blend_without_market_reach(fixture_artefacts):
    from exportspredictor import governance

    block = fixture_artefacts.evaluation["v1_comparison"]
    assert fixture_artefacts.model_form.market_reach is not None

    assert block["model_form"] == fixture_artefacts.model_form.without_market_reach().to_dict()
    assert block["verdict"]["text"].startswith(comparison.WITHOUT_MARKET_REACH_NOTE)
    reworded = comparison.relabelled(fixture_artefacts.evaluation)["v1_comparison"]["verdict"]["text"]
    assert reworded.startswith(comparison.WITHOUT_MARKET_REACH_NOTE)
    card = governance.render_model_card(governance.model_card(fixture_artefacts))
    assert comparison.WITHOUT_MARKET_REACH_NOTE in card
