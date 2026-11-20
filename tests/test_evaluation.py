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
    assert "does not beat the Naive Baseline" in worse["text"]
    assert better["beats_naive"] is True
    assert "better than the Naive Baseline" in better["text"]


def test_headline_says_when_the_win_over_the_naive_baseline_is_narrow():
    narrow = evaluation.headline({"model_spearman_within_product": 0.54, "naive_spearman_within_product": 0.52}, 2023)
    clear = evaluation.headline({"model_spearman_within_product": 0.70, "naive_spearman_within_product": 0.52}, 2023)

    assert narrow["beats_naive"] is True
    assert "only slightly better than the Naive Baseline" in narrow["text"]
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
        return [
            {k: row[k] for k in ("params", "rounds", "validation_deviance")}
            for row in artefacts.metadata["search"]["trials"]
        ]

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


# --- v2: Excluded Products and the imputed/observed tariff split (ADR 0005, ADR 0006) -------


def _scored(**extra) -> pl.DataFrame:
    # Two Products the UK ships to three Markets, plus crude oil (270900), an Excluded Product.
    return pl.DataFrame(
        {
            "exporter": ["GBR"] * 9,
            "importer": ["JPN", "CAN", "IND"] * 3,
            "hs6": ["220830"] * 3 + ["870323"] * 3 + ["270900"] * 3,
            "value_usd": [10.0, 20.0, 30.0, 30.0, 20.0, 10.0, 500.0, 900.0, 100.0],
            "weight": [1.0] * 9,
            "export_potential_usd": [11.0, 19.0, 31.0, 29.0, 21.0, 9.0, 1.0, 2.0, 3.0],
            "naive_usd": [10.0] * 9,
            **extra,
        }
    )


MARKETS = pl.DataFrame(
    {"iso3": ["JPN", "CAN", "IND"], "region": ["East Asia & Pacific", "North America", "South Asia"]}
)


def test_no_metric_row_includes_an_excluded_product():
    result = evaluation.accuracy(_scored(), MARKETS)

    assert result["overall"]["rows"] == 6
    assert result["overall"]["products"] == 2
    assert {row["group"] for row in result["by_sector"]} == {"22", "87"}
    assert sum(row["rows"] for row in result["by_region"]) == 6


def test_the_imputed_observed_split_partitions_the_test_rows():
    imputed = [True, False, True, True, False, True, True, True, False]

    result = evaluation.accuracy(_scored(tariff_imputed=imputed), MARKETS)

    split = {row["group"]: row for row in result["by_tariff"]}
    assert list(split) == ["observed", "imputed"]
    assert split["imputed"]["rows"] == 4 and split["observed"]["rows"] == 2  # Excluded rows left out
    assert split["imputed"]["rows"] + split["observed"]["rows"] == result["overall"]["rows"]
    for metric in evaluation.METRICS:
        assert f"model_{metric}" in split["observed"]


def test_the_split_reports_an_empty_side_with_no_rows_rather_than_dropping_it():
    # TRAINS ends in 2021, so every 2023 test row carries its tariff forward: nothing is observed.
    result = evaluation.accuracy(_scored(tariff_imputed=[True] * 9), MARKETS)

    split = {row["group"]: row for row in result["by_tariff"]}
    assert split["observed"]["rows"] == 0
    assert split["observed"]["model_spearman_within_product"] is None
    assert split["imputed"]["rows"] == result["overall"]["rows"]


def test_the_split_is_left_out_when_rows_carry_no_tariff_mark():
    assert "by_tariff" not in evaluation.accuracy(_scored(), MARKETS)


@pytest.mark.parametrize("scope", ["all_exporters", "uk_out_of_fold"])
def test_the_fixture_report_splits_every_test_row_by_tariff_imputed(report, scope):
    accuracy = report["accuracy"][scope]

    assert sum(row["rows"] for row in accuracy["by_tariff"]) == accuracy["overall"]["rows"]
    # The fixture's test year, 2023, is after TRAINS' last year: every row is imputed.
    assert {row["group"]: row["rows"] for row in accuracy["by_tariff"]}["observed"] == 0
    assert "carried forward from 2021" in report["tariff_split_text"]


def test_limitations_drop_no_tariffs_and_cover_v2s_caveats():
    titles = [item["title"] for item in evaluation.LIMITATIONS]
    text = " ".join(f"{item['title']} {item['text']}" for item in evaluation.LIMITATIONS).lower()

    assert "No tariffs" not in titles
    assert "not a model input in this version" not in text
    for topic in ["carried forward from 2021", "excluded products", "top 100", "nominal us dollars",
                  "potential range", "exchangeab"]:
        assert topic in text, topic


# --- The test year beside earlier versions --------------------------------------------------


def _final_report(model, naive):
    return {
        "test_year": 2023,
        "model_form": {"form": "blend", "naive_weight": 0.3},
        "headline": {"model": model, "naive": naive, "beats_naive": model > naive},
        "accuracy": {
            "all_exporters": {"overall": {"model_tweedie_deviance": 400.0}},
            "uk_out_of_fold": {"overall": {"model_tweedie_deviance": 700.0}},
        },
        "lift": {"lift": 4.0, "size_matched": {"lift": 0.9}},
        "v1_comparison": {"primary": {"v1": 0.5397}},
    }


V2 = {
    "version": "20261118T100524Z", "uk_ranking": 0.5115, "naive": 0.5233, "lift": 5.23, "size_matched_lift": 0.788,
    "tweedie_deviance_all_exporters": 355.5, "tweedie_deviance_uk": 568.6,
}


def test_the_test_year_is_reported_beside_v2_v1_and_both_naive_figures():
    block = evaluation.earlier_versions(_final_report(0.55, 0.52), v2=V2)

    rows = {row["name"]: row for row in block["rows"]}
    assert rows["Blend"]["uk_ranking"] == 0.55
    assert rows["Blend"]["size_matched_lift"] == 0.9
    assert rows["Blend"]["tweedie_deviance_all_exporters"] == 400.0
    assert rows["v2"]["uk_ranking"] == 0.5115 and rows["v2"]["size_matched_lift"] == 0.788
    assert rows["v1"]["uk_ranking"] == 0.5397
    naive = {row["name"]: row["uk_ranking"] for row in block["naive"]}
    assert naive == {"Naive Baseline, every Exporter's Flows (this run)": 0.52,
                     "Naive Baseline as reported for v2 (training rows' Flows)": 0.5233}
    assert block["beats_naive"] is True
    assert "beats the Naive Baseline" in block["text"] and "2023" in block["text"]


def test_a_model_that_does_not_beat_the_naive_baseline_is_said_so_plainly():
    block = evaluation.earlier_versions(_final_report(0.515, 0.52), v2=V2)

    assert block["beats_naive"] is False
    assert "does not beat the Naive Baseline" in block["text"]


def test_without_v2_figures_only_this_version_and_v1_are_listed():
    report = _final_report(0.55, 0.52)
    del report["lift"]
    block = evaluation.earlier_versions(report, v2=None)

    assert [row["name"] for row in block["rows"]] == ["Blend", "v1"]
    assert block["rows"][0]["size_matched_lift"] is None
    assert [row["name"] for row in block["naive"]] == ["Naive Baseline, every Exporter's Flows (this run)"]


def test_a_report_saved_with_the_old_names_is_read_with_the_glossary_s(fixture_artefacts):
    # Artefact 20261119T001736Z saved "This version (blend)" and "Naive baseline, ..." before review2.
    block = evaluation.earlier_versions(_final_report(0.555, 0.523), v2=V2)
    saved_names = {"Blend": "This version (blend)",
                   evaluation.NAIVE_THIS_RUN: "Naive baseline, every Exporter's Flows (this run)",
                   evaluation.NAIVE_AS_REPORTED_FOR_V2: "Naive baseline as reported for v2 (training rows' Flows)"}
    saved = {**block, "rows": [{**row, "name": saved_names.get(row["name"], row["name"])} for row in block["rows"]],
             "naive": [{**row, "name": saved_names[row["name"]]} for row in block["naive"]],
             "text": block["text"].replace("Naive Baseline", "naive baseline")}
    report = {**_final_report(0.555, 0.523), "earlier_versions": saved}

    assert evaluation.relabelled(report)["earlier_versions"] == block
    saved_headline = {"model": 0.555, "naive": 0.523, "beats_naive": True, "text": "... than the naive baseline ..."}
    headline = evaluation.relabelled({**report, "headline": saved_headline})["headline"]
    assert "only slightly better than the Naive Baseline" in headline["text"] and "0.56" in headline["text"]
    assert evaluation.served_report(report)["earlier_versions"] == block
    assert report["earlier_versions"] is saved and saved["rows"][0]["name"] == "This version (blend)"


def test_the_served_report_leaves_out_the_v1_comparison_and_words_the_potential_range_now(fixture_artefacts):
    saved = {**fixture_artefacts.evaluation, "v1_comparison": {"verdict": "frozen"}}
    saved["potential_range"] = {**saved["potential_range"], "text": "Worded when saved."}

    served = evaluation.served_report(saved)

    assert "v1_comparison" not in served
    assert served["potential_range"]["text"] != "Worded when saved."
    assert f"{saved['potential_range']['uncertain_opportunity_share']:.0%}" in served["potential_range"]["text"]
    assert saved["potential_range"]["text"] == "Worded when saved."  # the saved report is not changed


# --- Known limitations, worded for the served Model Form (spec #66 stories 41 and 48) ---------


def _headline_report(model=0.555, naive=0.523):
    return {"headline": {"model": model, "naive": naive, "beats_naive": model > naive}}


def test_a_blend_s_ranking_limitation_says_dollar_values_deserve_more_caution_than_rankings():
    limits = evaluation.result_limitations({"model_form": {"form": "blend", "naive_weight": 0.3}}, _headline_report())

    text = next(x["text"] for x in limits if x["title"] == "Only slightly better than a naive ranking for UK pairs")
    assert "0.56" in text and "0.52" in text
    assert "mostly in the size" not in text  # the Blend ranks better and sizes worse, not the other way round
    assert "dollar values with more caution than rankings" in text


def test_the_model_alone_s_ranking_limitation_still_says_its_value_is_in_the_size():
    limits = evaluation.result_limitations({}, _headline_report())

    text = next(x["text"] for x in limits if x["title"] == "Only slightly better than a naive ranking for UK pairs")
    assert "size of Export Potential" in text


def test_the_potential_range_limitation_refers_to_no_badge():
    bands = pl.DataFrame({
        "band": [0, 1], "min_export_potential_usd": [0.0, 2_000_000.0], "rows": [10, 10],
        "low_offset_usd": [-500.0, -8_200_000.0], "high_offset_usd": [800.0, 15_600_000.0],
    })

    (limit,) = evaluation.range_limitations(bands)

    assert "badge" not in limit["text"]  # the ranking pages carry none since spec #66 story 41
    assert "a small gap for its size" in limit["text"]


def test_log_scale_bands_have_no_fixed_width_limitation():
    bands = pl.DataFrame({
        "band": [0, 1], "min_export_potential_usd": [0.0, 2_000_000.0], "rows": [10, 10],
        "low_offset_log": [-1.2, -2.5], "high_offset_log": [0.3, 1.1],
    })

    assert evaluation.range_limitations(bands) == []


def test_the_markets_no_source_gives_a_gdp_are_a_known_limitation():
    from exportspredictor import gravity

    gaps = [item for item in evaluation.LIMITATIONS if item["title"] == "No GDP for some small Markets"]

    assert len(gaps) == 1
    for name in gravity.GDP_GAPS.values():
        assert name in gaps[0]["text"], name
