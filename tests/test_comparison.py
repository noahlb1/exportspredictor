"""The pre-registered v1/v2 comparison (spec #39, decision #38), on a synthetic frozen-v1 folder.

v2's scores here are a stand-in (Actual Exports with seeded noise), so these tests check the
protocol (common rows, bootstrap, verdict, manifest), not either model.
"""

import shutil

import numpy as np
import polars as pl
import pytest

from exportspredictor import comparison, freeze
from exportspredictor.exclusions import is_excluded
from exportspredictor.fixture import UK

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
