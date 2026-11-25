"""ITC's realisation test run on our model (#156, #175): the report of the pure function, and the step that adds
it to the newest artefact's evaluation."""

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from exportspredictor import artefacts as store
from exportspredictor import realisation
from exportspredictor.ledger import LEDGER_PATH

PANEL_SCHEMA = {
    "hs6": pl.String, "iso3": pl.String, "year": pl.Int16,
    "export_potential": pl.Float64, "naive_baseline": pl.Float64, "actual_exports": pl.Float64,
}


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


def _panel(series: dict[tuple[str, str], tuple[list[float], dict[int, float]]], years=range(2012, 2023)) -> pl.DataFrame:
    """A panel from {(hs6, iso3): (Actual Exports per year, {base year: Export Potential})}."""
    rows = []
    for (hs6, iso3), (actual, potential) in series.items():
        for year, value in zip(years, actual, strict=True):
            scored = potential.get(year)
            rows.append((hs6, iso3, year, scored, scored, value))
    return pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")


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
        predictor="export_potential",
    ).sort("hs6")["unrealised_share"].to_list()

    assert shares == pytest.approx([0.0, 0.75])


def test_growth_runs_from_the_base_year_to_the_mean_of_the_next_three_and_past_growth_from_the_three_before():
    # 2013-2015 average 5, 2016 is 10, 2017-2019 average 20.
    actual = [1.0, 4.0, 5.0, 6.0, 10.0, 10.0, 20.0, 30.0, 1.0, 1.0, 1.0]
    [row] = realisation.intensive_margin_sample(
        _panel({("010101", "FRA"): (actual, {2016: 40.0})}), predictor="export_potential"
    ).iter_rows(named=True)

    assert row["growth"] == pytest.approx(np.log(2.0))
    assert row["past_growth"] == pytest.approx(np.log(2.0))


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
