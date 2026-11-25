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

    # Every row, at both levels and for both predictors, as saved.
    keys = [(row["test"], row["level"], row["predictor"]) for row in served["rows"]]
    assert keys == [(row["test"], row["level"], row["predictor"]) for row in report["rows"]]
    assert len(keys) == 12
    for row, saved in zip(served["rows"], report["rows"], strict=True):
        assert row["coefficient"] == pytest.approx(saved["coefficient"])
    [saved] = [
        row for row in report["rows"]
        if (row["test"], row["level"], row["predictor"]) == ("intensive_margin", "HS2", "model")
    ]
    assert served["base_years"] == [2015, 2016, 2017, 2018, 2019]
    assert {row["test"]: row["coefficient"] for row in served["published"]} == {
        "intensive_margin": 0.615, "extensive_margin": 0.023, "new_flow_size": 0.243,
    }
    # The text reads the HS2 level, the one compared with ITC.
    assert f"{saved['coefficient']:.3f}" in served["text"] and "0.615" in served["text"]
    assert "(HS2)" in served["text"] and "of 3" in served["text"] and "Naive Baseline" in served["text"]
    assert "2015–2019" in served["text"]
    # Agreement, never validation (CLAUDE.md: associational language).
    assert "rough benchmark" in served["text"]
    assert "validat" not in served["text"].replace("not a validation", "")
    # The caveats name each difference from ITC.
    for difference in ("UK only", "no exporter fixed effect", "76 sub-sectors", "HS2", "2006–2018", "2015–2019",
                       "learned model", "no forward projection"):
        assert difference in served["caveat"], difference
    # Advisors see only the level compared with ITC.
    assert {row["Level"] for row in served["table"]} == {"HS2"}
    assert "not compared" not in {row["ITC (76 sub-sectors)"] for row in served["table"]}
    assert [row["Test"] for row in served["table"]] == [
        "Intensive margin", "Intensive margin", "Extensive margin", "New-flow size",
    ]
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
