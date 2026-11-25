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


def test_the_coefficient_table_sets_our_coefficient_beside_itc_s(bundle, report):
    table = reporting.itc_coefficients(bundle)
    [row] = [row for row in report["rows"] if row["test"] == "intensive_margin"]

    assert list(table.columns) == ["Test", "Term", "ITC (76 sub-sectors)", "Model, HS6"]
    share, past = table.rows(named=True)
    assert (share["Test"], share["Term"]) == ("Intensive margin", "Unrealised Share")
    assert share["ITC (76 sub-sectors)"] == "0.615"
    assert share["Model, HS6"] == f"{row['coefficient']:.3f} ({row['std_error']:.3f})"
    assert past["Term"] == "past growth" and past["ITC (76 sub-sectors)"] == "-0.386"
    past_fit = row["controls"]["past_growth"]
    assert past["Model, HS6"] == f"{past_fit['coefficient']:.3f} ({past_fit['std_error']:.3f})"


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
