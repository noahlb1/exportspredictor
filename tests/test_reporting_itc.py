"""The notebook's ITC comparison tables (#156 parts a and b), read from the saved evaluation report."""

from dataclasses import replace

import pytest

from exportspredictor import realisation, reporting


@pytest.fixture(scope="module")
def report():
    return realisation.realisation_report(realisation.fixture_panel())


@pytest.fixture(scope="module")
def bundle(fixture_artefacts, report):
    return realisation.with_itc_comparison(fixture_artefacts, report)


def _fit(report: dict, test: str, level: str, predictor: str) -> dict:
    [row] = [
        row for row in report["rows"] if (row["test"], row["level"], row["predictor"]) == (test, level, predictor)
    ]
    return row


def test_the_coefficient_table_has_a_row_per_test_level_and_term_and_a_column_per_predictor(bundle, report):
    table = reporting.itc_coefficients(bundle)

    assert list(table.columns) == ["Test", "Level", "Term", "ITC (76 sub-sectors)", "Model", "Naive Baseline"]
    assert [(row["Test"], row["Level"], row["Term"]) for row in table.rows(named=True)] == [
        ("Intensive margin", "HS2", "Unrealised Share"),
        ("Intensive margin", "HS2", "past growth"),
        ("Intensive margin", "HS6", "Unrealised Share"),
        ("Intensive margin", "HS6", "past growth"),
        ("Extensive margin", "HS2", "log Export Potential"),
        ("Extensive margin", "HS6", "log Export Potential"),
        ("New-flow size", "HS2", "log Export Potential"),
        ("New-flow size", "HS6", "log Export Potential"),
    ]
    share = table.row(0, named=True)
    model, naive = (_fit(report, "intensive_margin", "HS2", p) for p in ("model", "naive_baseline"))
    assert share["Model"] == f"{model['coefficient']:.3f} ({model['std_error']:.3f})"
    assert share["Naive Baseline"] == f"{naive['coefficient']:.3f} ({naive['std_error']:.3f})"
    past = table.row(1, named=True)
    past_fit = model["controls"]["past_growth"]
    assert past["Model"] == f"{past_fit['coefficient']:.3f} ({past_fit['std_error']:.3f})"


def test_itc_s_figures_sit_on_the_hs2_rows_only(bundle):
    table = reporting.itc_coefficients(bundle)

    itc = {(row["Test"], row["Level"], row["Term"]): row["ITC (76 sub-sectors)"] for row in table.rows(named=True)}
    assert itc[("Intensive margin", "HS2", "Unrealised Share")] == "0.615"
    assert itc[("Intensive margin", "HS2", "past growth")] == "-0.386"
    assert itc[("Extensive margin", "HS2", "log Export Potential")] == "0.023"
    assert itc[("New-flow size", "HS2", "log Export Potential")] == "0.243"
    assert {value for (_, level, _), value in itc.items() if level == "HS6"} == {"not compared"}


def test_the_method_table_sets_the_two_methods_side_by_side(bundle):
    table = reporting.itc_method(bundle)

    assert list(table.columns) == ["Aspect", "ITC's Export Potential Map", "This model"]
    assert table["Aspect"].to_list() == [
        "Estimator", "Data and years", "Target year", "Product level", "Gap definition", "How it is tested",
    ]
    assert "2015–2019" in table.filter(table["Aspect"] == "How it is tested")["This model"].item()


def test_the_direct_comparison_says_it_was_dropped_and_why(bundle):
    text = reporting.itc_direct_comparison(bundle)

    assert text.startswith("Direct comparison with the EPM's UK figures: dropped")
    assert "terms" in text


def test_without_the_report_each_table_is_not_run(fixture_artefacts):
    evaluation = {k: v for k, v in fixture_artefacts.evaluation.items() if k != realisation.REPORT_KEY}
    bare = replace(fixture_artefacts, evaluation=evaluation)

    for table in (reporting.itc_coefficients(bare), reporting.itc_method(bare), reporting.itc_direct_comparison(bare)):
        assert isinstance(table, reporting.NotRun)
