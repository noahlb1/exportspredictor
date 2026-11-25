"""ITC's realisation test run on our model (#156, #175): the report of the pure function, and the step that adds
it to the newest artefact's evaluation."""

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from exportspredictor import artefacts as store
from exportspredictor import base_years, realisation
from exportspredictor.ledger import LEDGER_PATH


def _intensive(report: dict) -> dict:
    [row] = [
        row for row in report["rows"]
        if (row["test"], row["level"], row["predictor"]) == ("intensive_margin", "HS6", "model")
    ]
    return row


def test_pairs_with_a_larger_unrealised_share_grow_more_so_the_coefficient_is_positive():
    row = _intensive(realisation.realisation_report(realisation.fixture_panel(effect=0.6, seed=1)))

    assert row["regressor"] == "unrealised_share"
    assert row["coefficient"] > 0
    assert row["coefficient"] > 3 * row["std_error"]
    assert "past_growth" in row["controls"]


def test_under_null_data_the_coefficient_is_near_zero():
    row = _intensive(realisation.realisation_report(realisation.fixture_panel(effect=0.0, seed=2)))

    assert abs(row["coefficient"]) < 3 * row["std_error"]
    assert abs(row["coefficient"]) < 0.1


def _panel(
    series: dict[tuple[str, str], tuple[list[float], dict[int, float]]],
    years=range(2012, 2023),
    naive: dict[tuple[str, str], float] | None = None,
) -> pl.DataFrame:
    """A panel from {(hs6, iso3): (Actual Exports per year, {base year: Export Potential})}; the Naive Baseline is
    Export Potential, or a pair's value in `naive` in its base years."""
    rows = []
    for (hs6, iso3), (actual, potential) in series.items():
        for year, value in zip(years, actual, strict=True):
            scored = potential.get(year)
            baseline = (naive or {}).get((hs6, iso3), scored) if scored is not None else None
            rows.append((hs6, iso3, year, scored, baseline, value))
    return pl.DataFrame(rows, schema=realisation.PANEL_SCHEMA, orient="row")


def test_pairs_with_no_exports_in_the_three_later_years_drop_out_and_are_counted():
    flat = [10.0] * 11
    stops = [10.0] * 7 + [0.0] * 4  # trades through 2018, nothing in 2019-2021: dropped at base year 2018 only
    series = {("010101", "FRA"): (flat, {2018: 20.0}), ("010102", "FRA"): (stops, {2018: 20.0})}

    row = _intensive(realisation.realisation_report(_panel(series)))

    assert row["trading"] == 2
    assert row["no_later_exports"] == 1


def test_the_unrealised_share_is_floored_at_zero_for_pairs_above_their_export_potential():
    # Actual Exports at t (2016) are 10. Export Potential of 5 gives a share of 0, not -1; 40 gives 0.75.
    shares = realisation.intensive_margin_sample(
        _panel({
            ("010101", "FRA"): ([10.0] * 11, {2016: 5.0}),
            ("010102", "FRA"): ([10.0] * 11, {2016: 40.0}),
        }),
        predictor="model",
    ).sort("hs6")["unrealised_share"].to_list()

    assert shares == pytest.approx([0.0, 0.75])


def test_growth_runs_from_the_base_year_to_the_mean_of_the_next_three_and_past_growth_from_the_three_before():
    # 2013-2015 average 5, 2016 is 10, 2017-2019 average 20.
    actual = [1.0, 4.0, 5.0, 6.0, 10.0, 10.0, 20.0, 30.0, 1.0, 1.0, 1.0]
    [row] = realisation.intensive_margin_sample(
        _panel({("010101", "FRA"): (actual, {2016: 40.0})}), predictor="model"
    ).iter_rows(named=True)

    assert row["growth"] == pytest.approx(np.log(2.0))
    assert row["past_growth"] == pytest.approx(np.log(2.0))


def _not_trading_panel() -> pl.DataFrame:
    """Base year 2016 only. FRA trades 010101 throughout; DEU starts it in 2018; ITA never buys it; 020202 sold
    to FRA in 2015 only, so at 2016 the UK exports it nowhere."""
    zeros = [0.0] * 11
    starts = [0.0] * 6 + [5.0, 7.0, 0.0, 0.0, 0.0]  # 2018 and 2019
    lapsed = [0.0, 0.0, 0.0, 4.0] + [0.0] * 7  # 2015 only
    return _panel({
        ("010101", "FRA"): ([10.0] * 11, {2016: 20.0}),
        ("010101", "DEU"): (starts, {2016: 8.0}),
        ("010101", "ITA"): (zeros, {2016: 2.0}),
        ("020202", "FRA"): (lapsed, {2016: 3.0}),
    })


def test_the_extensive_margin_takes_pairs_not_trading_at_t_whose_product_the_uk_exports_somewhere():
    sample = realisation.extensive_margin_sample(_not_trading_panel(), predictor="model").sort("iso3")

    assert sample["iso3"].to_list() == ["DEU", "ITA"]
    assert sample["started"].to_list() == [1.0, 0.0]
    assert sample["log_export_potential"].to_list() == pytest.approx([np.log(8.0), np.log(2.0)])


def test_new_flow_size_takes_the_pairs_that_start_with_the_log_of_their_three_year_sum():
    [row] = realisation.new_flow_size_sample(_not_trading_panel(), predictor="model").iter_rows(named=True)

    assert (row["hs6"], row["iso3"]) == ("010101", "DEU")
    assert row["size"] == pytest.approx(np.log(12.0))


def test_the_extensive_margin_row_counts_who_was_not_trading_and_who_was_left_out():
    report = realisation.realisation_report(_not_trading_panel())
    [row] = [
        row for row in report["rows"]
        if (row["test"], row["level"], row["predictor"]) == ("extensive_margin", "HS6", "model")
    ]

    assert row["not_trading"] == 3
    assert row["product_not_exported"] == 1
    assert row["started"] == 1


def test_at_hs2_the_unrealised_share_sums_floored_unrealised_potential_so_a_surplus_cannot_cancel_a_gap():
    # Sector 01 in FRA at 2016: 010101 sells 10 of 20 (10 unrealised); 010102 sells 30 of 10 (surplus 20).
    # Netted, the Sector sells 40 of 30 and shows nothing unrealised; ITC's sum of floored unrealised potential gives 10 / 30.
    [row] = realisation.intensive_margin_sample(
        _panel({
            ("010101", "FRA"): ([10.0] * 11, {2016: 20.0}),
            ("010102", "FRA"): ([30.0] * 5 + [60.0] * 6, {2016: 10.0}),
        }),
        predictor="model",
        level="HS2",
    ).iter_rows(named=True)

    assert row["sector"] == "01"
    assert row["unrealised_share"] == pytest.approx(1 / 3)
    # Actual Exports are summed too: 40 at t, 70 a year after it.
    assert row["growth"] == pytest.approx(np.log(70 / 40))


def test_the_naive_baseline_column_runs_through_the_same_code_as_the_model_s():
    panel = realisation.fixture_panel().with_columns(naive_baseline_usd=pl.col("export_potential_usd"))

    rows = realisation.realisation_report(panel)["rows"]

    by_predictor = {
        predictor: {(row["test"], row["level"]): row for row in rows if row["predictor"] == predictor}
        for predictor in ("model", "naive_baseline")
    }
    assert set(by_predictor["model"]) == {
        (test, level)
        for test in ("intensive_margin", "extensive_margin", "new_flow_size")
        for level in ("HS2", "HS6")
    }
    for key, model in by_predictor["model"].items():
        naive = by_predictor["naive_baseline"][key]
        assert {k: v for k, v in naive.items() if k != "predictor"} == {
            k: v for k, v in model.items() if k != "predictor"
        }, key


def test_both_predictors_run_on_the_units_every_predictor_scores_above_zero_and_the_rest_are_counted():
    # Three trading pairs at 2016; the Naive Baseline gives 010103 nothing, so it leaves both columns' samples.
    series = {
        ("010101", "FRA"): ([10.0] * 11, {2016: 20.0}),
        ("010102", "FRA"): ([10.0] * 11, {2016: 40.0}),
        ("010103", "FRA"): ([10.0] * 11, {2016: 30.0}),
    }
    panel = _panel(series, naive={("010103", "FRA"): 0.0})

    samples = {
        predictor: realisation.intensive_margin_sample(panel, predictor=predictor)["hs6"].to_list()
        for predictor in ("model", "naive_baseline")
    }
    assert samples == {"model": ["010101", "010102"], "naive_baseline": ["010101", "010102"]}
    rows = [
        row for row in realisation.realisation_report(panel)["rows"]
        if (row["test"], row["level"]) == ("intensive_margin", "HS6")
    ]
    assert [(row["predictor"], row["trading"], row["zero_score"]) for row in rows] == [
        ("model", 3, 1), ("naive_baseline", 3, 1),
    ]


def test_a_training_cut_off_after_its_base_year_is_refused():
    late = {t: t for t in realisation.FIXTURE_BASE_YEARS} | {2017: 2018}

    with pytest.raises(ValueError, match="after their base year"):
        realisation.realisation_report(realisation.fixture_panel(), training_cut_offs=late)


def test_on_the_fixture_panel_every_test_is_fitted_at_both_levels_for_both_predictors():
    report = realisation.realisation_report(realisation.fixture_panel())

    assert len(report["rows"]) == 12
    for row in report["rows"]:
        assert row["coefficient"] is not None and row["std_error"] is not None, row


def test_itc_s_three_static_coefficients_are_stored_with_their_tables():
    published = {row["test"]: row for row in realisation.PUBLISHED}

    assert (published["intensive_margin"]["coefficient"], published["intensive_margin"]["controls"]) == (
        0.615, {"past_growth": -0.386}
    )
    assert published["extensive_margin"]["coefficient"] == 0.023
    assert published["new_flow_size"]["coefficient"] == 0.243
    assert "Table 1" in published["intensive_margin"]["source"]
    assert "Table 2" in published["extensive_margin"]["source"]
    assert "Table 4" in published["new_flow_size"]["source"]


def test_the_report_names_its_base_years_and_holds_itc_s_published_coefficient():
    report = realisation.realisation_report(realisation.fixture_panel())

    assert report["base_years"] == [2015, 2016, 2017, 2018, 2019]
    [published] = [row for row in report["published"] if row["test"] == "intensive_margin"]
    assert published["coefficient"] == 0.615
    assert published["controls"] == {"past_growth": -0.386}
    assert "Table 1" in published["source"]
    assert report["direct_comparison"]["run"] is False


def test_a_panel_reaching_the_test_year_is_refused():
    panel = realisation.fixture_panel()
    late = panel.filter(pl.col("year") == 2022).with_columns(year=pl.lit(2023, dtype=pl.Int16))

    with pytest.raises(ValueError, match="test year"):
        realisation.realisation_report(pl.concat([panel, late]))


def test_the_report_is_json_ready():
    report = realisation.realisation_report(realisation.fixture_panel())

    assert json.loads(json.dumps(report, allow_nan=False)) == report


# --- Adding the report to the newest artefact -----------------------------------------------------


def test_the_report_is_added_to_a_new_version_whose_evaluation_is_otherwise_unchanged(fixture_artefacts, tmp_path):
    source = store.save(fixture_artefacts, tmp_path)
    ledger_before = hashlib.sha256(LEDGER_PATH.read_bytes()).hexdigest() if LEDGER_PATH.exists() else None
    panel = realisation.fixture_panel()

    folder = realisation.add_to_latest(tmp_path, panel=panel, version="fixture-itc")

    assert folder.name == "fixture-itc" and folder != source
    bundle = store.load(folder)
    saved = bundle.evaluation["itc_comparison"]
    assert _intensive(saved)["coefficient"] == pytest.approx(_intensive(realisation.realisation_report(panel))["coefficient"])
    rest = {key: value for key, value in bundle.evaluation.items() if key != "itc_comparison"}
    assert rest == {key: value for key, value in fixture_artefacts.evaluation.items() if key != "itc_comparison"}
    assert bundle.metadata["copied_from"] == fixture_artefacts.version
    assert bundle.baseline.equals(fixture_artefacts.baseline)
    assert bundle.ppml_benchmark == fixture_artefacts.ppml_benchmark
    # 2023 is never read, so the Test-year Ledger is untouched (ADR 0009).
    assert (hashlib.sha256(LEDGER_PATH.read_bytes()).hexdigest() if LEDGER_PATH.exists() else None) == ledger_before


def _saved_panel(root, *, artefact_version: str):
    """The fixture panel saved under the data `root` with the record of the artefact it was built from."""
    records = [{"base_year": t, "training_years": list(range(2012, t + 1))} for t in realisation.FIXTURE_BASE_YEARS]
    base_years.save_panel(
        realisation.fixture_panel(), root, info={"artefact_version": artefact_version, "base_year_models": records}
    )


def test_a_saved_panel_built_from_another_artefact_is_refused(fixture_artefacts, tmp_path):
    store.save(fixture_artefacts, tmp_path / "artefacts")
    _saved_panel(tmp_path / "data", artefact_version="another-version")

    with pytest.raises(ValueError, match="another-version"):
        realisation.add_to_latest(tmp_path / "artefacts", root=tmp_path / "data", version="fixture-itc")

    assert not (tmp_path / "artefacts" / "fixture-itc").exists()


def test_a_saved_panel_of_the_artefact_the_newest_was_copied_from_is_added_with_its_training_cut_offs(
    fixture_artefacts, tmp_path
):
    artefacts = tmp_path / "artefacts"
    store.save(fixture_artefacts, artefacts)
    # The newest version is a copy of the one the panel was built from.
    realisation.add_to_latest(artefacts, panel=realisation.fixture_panel(), version="zz-copy")
    _saved_panel(tmp_path / "data", artefact_version=fixture_artefacts.version)

    folder = realisation.add_to_latest(artefacts, root=tmp_path / "data", version="zz-copy-itc")

    saved = store.load(folder).evaluation["itc_comparison"]
    assert saved["training_cut_offs"] == [
        {"base_year": t, "training_cut_off": t} for t in (2015, 2016, 2017, 2018, 2019)
    ]
