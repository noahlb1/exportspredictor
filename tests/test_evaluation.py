"""Accuracy evaluation on the test year: metrics, the naive baseline, breakdowns and stability."""

import math
from dataclasses import replace

import polars as pl
import pytest

from exportspredictor import evaluation
from exportspredictor.pipeline import build_artefacts


@pytest.fixture
def report(fixture_artefacts):
    return fixture_artefacts.evaluation


# --- Metrics ---------------------------------------------------------------------------


def test_tweedie_deviance_is_zero_for_perfect_predictions():
    frame = pl.DataFrame({"value_usd": [5.0, 100.0, 3.0], "weight": [1.0, 1.0, 1.0], "p": [5.0, 100.0, 3.0]})

    assert frame.select(evaluation.tweedie_deviance("p")).item() == pytest.approx(0.0, abs=1e-9)


def test_tweedie_deviance_matches_the_unit_deviance_formula_and_weights():
    # p = 1.5: d(y, mu) = 2 * (y^0.5 / -0.25 - y * mu^-0.5 / -0.5 + mu^0.5 / 0.5)
    def unit(y, mu):
        return 2 * (y**0.5 / -0.25 + y * mu**-0.5 / 0.5 + mu**0.5 / 0.5)

    frame = pl.DataFrame({"value_usd": [0.0, 400.0], "weight": [3.0, 1.0], "p": [4.0, 100.0]})

    expected = (3 * unit(0.0, 4.0) + 1 * unit(400.0, 100.0)) / 4
    assert frame.select(evaluation.tweedie_deviance("p")).item() == pytest.approx(expected)


def test_tweedie_deviance_floors_predictions_so_a_zero_prediction_stays_finite():
    frame = pl.DataFrame({"value_usd": [1000.0], "weight": [1.0], "p": [0.0]})

    assert math.isfinite(frame.select(evaluation.tweedie_deviance("p")).item())


def test_naive_potential_is_the_exporters_world_share_times_the_importers_imports_leaving_the_own_flow_out():
    flows = pl.DataFrame(
        {
            "exporter": ["GBR", "GBR", "FRA", "FRA"],
            "importer": ["JPN", "CAN", "JPN", "CAN"],
            "hs6": ["220830"] * 4,
            "value_usd": [10.0, 30.0, 20.0, 40.0],
        }
    )
    rows = flows.vstack(
        pl.DataFrame({"exporter": ["DEU"], "importer": ["JPN"], "hs6": ["220830"], "value_usd": [0.0]})
    )

    naive = evaluation.naive_potential(rows, flows)

    by_row = {(r["exporter"], r["importer"]): r["naive_usd"] for r in naive.iter_rows(named=True)}
    # GBR->JPN: GBR's other exports 30, JPN's other imports 20, world other 90.
    assert by_row[("GBR", "JPN")] == pytest.approx(30 / 90 * 20)
    # DEU exports nothing, so its naive potential is zero.
    assert by_row[("DEU", "JPN")] == 0.0
    assert naive.height == rows.height


def test_headline_says_plainly_when_the_model_does_not_beat_the_naive_baseline():
    worse = evaluation.headline({"model_spearman_within_product": 0.2, "naive_spearman_within_product": 0.5}, 2023)
    better = evaluation.headline({"model_spearman_within_product": 0.6, "naive_spearman_within_product": 0.5}, 2023)

    assert worse["beats_naive"] is False
    assert "does not beat the naive baseline" in worse["text"]
    assert better["beats_naive"] is True
    assert "better than the naive baseline" in better["text"]


def test_headline_says_when_the_win_over_the_naive_baseline_is_narrow():
    narrow = evaluation.headline({"model_spearman_within_product": 0.54, "naive_spearman_within_product": 0.52}, 2023)
    clear = evaluation.headline({"model_spearman_within_product": 0.70, "naive_spearman_within_product": 0.52}, 2023)

    assert narrow["beats_naive"] is True
    assert "only slightly better than the naive baseline" in narrow["text"]
    assert "only slightly" not in clear["text"]


def test_stability_of_identical_rankings_is_perfect():
    table = pl.DataFrame(
        {
            "hs6": ["1"] * 3 + ["2"] * 3,
            "iso3": ["A", "B", "C"] * 2,
            "export_potential_usd": [1.0, 2.0, 3.0, 6.0, 5.0, 4.0],
            "export_gap_usd": [1.0, -2.0, 3.0, 0.5, 5.0, -4.0],
        }
    )

    result = evaluation.stability(table, table, top_n=2)

    assert result["spearman_potential"] == pytest.approx(1.0)
    assert result["spearman_gap"] == pytest.approx(1.0)
    assert result["top_overlap"] == pytest.approx(1.0)
    assert result["pairs"] == 6


def test_limitations_cover_every_required_topic():
    text = " ".join(f"{item['title']} {item['text']}" for item in evaluation.LIMITATIONS).lower()

    for topic in ["goods only", "tariff", "associations", "provisional", "2012", "reporting", "sanctions"]:
        assert topic in text, topic


# --- The evaluation built with the artefacts ------------------------------------------


def test_the_test_year_follows_the_validation_year_and_its_models_train_only_on_earlier_years(report, fixture_artefacts):
    assert report["test_year"] == 2023
    assert report["validation_year"] == fixture_artefacts.metadata["validation_year"] == 2022
    assert report["test_training_years"] == [2021, 2022]
    assert 2023 not in fixture_artefacts.metadata["tuning_years"]
    assert report["params"] == fixture_artefacts.metadata["params"]
    assert report["rounds"] == fixture_artefacts.metadata["rounds"]


def test_the_test_year_is_untouched_during_tuning(fixture_world, fixture_artefacts):
    test_year = pl.col("year") == 2023
    distorted = replace(
        fixture_world,
        flows=fixture_world.flows.with_columns(
            value_usd=pl.when(test_year).then(pl.col("value_usd") * 1000).otherwise("value_usd")
        ),
    )

    rebuilt = build_artefacts(distorted, version="distorted-test-year")

    def search(artefacts):
        return [{k: row[k] for k in ("config", "rounds", "valid_metric")} for row in artefacts.metadata["tuning"]]

    assert search(rebuilt) == search(fixture_artefacts)
    # ...while the test-year evaluation does see the change.
    assert rebuilt.evaluation["accuracy"]["all_exporters"]["overall"] != (
        fixture_artefacts.evaluation["accuracy"]["all_exporters"]["overall"]
    )


@pytest.mark.parametrize("scope", ["all_exporters", "uk_out_of_fold"])
def test_every_metric_is_reported_for_the_model_and_the_naive_baseline(report, scope):
    overall = report["accuracy"][scope]["overall"]

    for metric in evaluation.METRICS:
        assert math.isfinite(overall[f"model_{metric}"]), metric
        assert math.isfinite(overall[f"naive_{metric}"]), metric
    assert overall["rows"] > 0 and overall["non_zero_flows"] > 0


def test_uk_out_of_fold_covers_the_full_test_year_grid(report):
    assert report["accuracy"]["uk_out_of_fold"]["overall"]["rows"] == 5 * 4


def test_breakdowns_cover_every_sector_and_region(report):
    accuracy = report["accuracy"]["uk_out_of_fold"]

    assert {row["group"] for row in accuracy["by_sector"]} == {"22", "30", "71", "85", "87"}
    assert {row["group"] for row in accuracy["by_region"]} == {"East Asia & Pacific", "North America", "South Asia"}
    for row in accuracy["by_sector"] + accuracy["by_region"]:
        assert set(row) >= {"group", "rows", *(f"model_{m}" for m in evaluation.METRICS)}


def test_breakdowns_list_the_weakest_groups_first(report):
    for breakdown in ("by_sector", "by_region"):
        rows = report["accuracy"]["all_exporters"][breakdown]
        scores = [row["model_spearman_within_product"] for row in rows]
        ranked = [s for s in scores if s is not None and not math.isnan(s)]
        assert ranked == sorted(ranked), breakdown


def test_headline_compares_the_model_with_the_naive_baseline(report):
    headline = report["headline"]
    overall = report["accuracy"]["uk_out_of_fold"]["overall"]

    assert headline["model"] == overall["model_spearman_within_product"]
    assert headline["naive"] == overall["naive_spearman_within_product"]
    assert headline["beats_naive"] == (headline["model"] > headline["naive"])
    assert headline["text"]


def test_stability_compares_the_baseline_with_the_test_year(report):
    stability = report["stability"]

    assert stability["years"] == [2023, 2024]
    assert stability["pairs"] == 20
    assert -1 <= stability["spearman_potential"] <= 1
    assert 0 <= stability["top_overlap"] <= 1
    assert stability["text"]


def test_the_evaluation_records_the_training_sample_rates(report):
    assert report["non_uk_sampling_rate"] == 1.0  # the fixture keeps every row
    assert report["zero_sampling_rate"] == pytest.approx(0.5)


def test_accuracy_adds_an_income_level_breakdown_when_markets_carry_it():
    # Income level stands in for reporting quality in the bias review: poorer countries report later and less fully.
    scored = pl.DataFrame(
        {
            "exporter": ["GBR"] * 6,
            "importer": ["JPN", "JPN", "JPN", "IND", "IND", "IND"],
            "hs6": ["220830", "300490", "870323"] * 2,
            "value_usd": [10.0, 20.0, 30.0, 30.0, 20.0, 10.0],
            "weight": [1.0] * 6,
            "export_potential_usd": [11.0, 19.0, 31.0, 29.0, 21.0, 9.0],
            "naive_usd": [10.0] * 6,
        }
    )
    markets = pl.DataFrame(
        {"iso3": ["JPN", "IND"], "region": ["East Asia & Pacific", "South Asia"], "income_level": ["High income", "Lower middle income"]}
    )

    result = evaluation.accuracy(scored, markets)

    assert {row["group"]: row["rows"] for row in result["by_income"]} == {"High income": 3, "Lower middle income": 3}
    assert "by_income" not in evaluation.accuracy(scored, markets.drop("income_level"))
