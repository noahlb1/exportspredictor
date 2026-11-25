"""`/evaluation`'s `itc_comparison`: ITC's realisation test on our model, worded for Advisors by the API (#175)."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import realisation


def _evaluation(bundle, root) -> dict:
    store.save(bundle, root)
    with TestClient(create_app(root)) as client:
        response = client.get("/evaluation")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def report():
    return realisation.realisation_report(realisation.fixture_panel())


def test_the_evaluation_carries_the_rows_the_years_and_the_api_s_own_text(fixture_artefacts, report, tmp_path):
    served = _evaluation(realisation.with_itc_comparison(fixture_artefacts, report), tmp_path)["itc_comparison"]

    [row] = served["rows"]
    [saved] = report["rows"]
    assert (row["test"], row["level"], row["predictor"]) == ("intensive_margin", "HS6", "model")
    assert row["coefficient"] == pytest.approx(saved["coefficient"])
    assert row["controls"]["past_growth"]["coefficient"] == pytest.approx(saved["controls"]["past_growth"]["coefficient"])
    assert served["base_years"] == [2015, 2016, 2017, 2018, 2019]
    assert {row["test"]: row["coefficient"] for row in served["published"]}["intensive_margin"] == 0.615
    assert f"{saved['coefficient']:.2f}" in served["text"] and "0.62" in served["text"]
    assert "2015–2019" in served["text"]
    # Agreement, never validation (CLAUDE.md: associational language).
    assert "rough benchmark" in served["text"]
    assert "validat" not in served["text"].replace("not a validation", "")
    assert "UK only" in served["caveat"]
    assert served["direct_comparison"]["run"] is False
    assert "ITC comparison" not in [part["title"] for part in _evaluation(
        realisation.with_itc_comparison(fixture_artefacts, report), tmp_path / "again"
    )["not_run"]]


def test_without_the_result_the_field_is_absent_and_the_part_is_listed_as_not_run(fixture_artefacts, tmp_path):
    evaluation = {k: v for k, v in fixture_artefacts.evaluation.items() if k != realisation.REPORT_KEY}
    served = _evaluation(replace(fixture_artefacts, evaluation=evaluation), tmp_path)

    assert served.get("itc_comparison") is None
    [part] = [part for part in served["not_run"] if part["title"] == "ITC comparison"]
    assert "not run for this version" in part["text"]
    assert "python -m exportspredictor.realisation" in part["text"]
