"""The Model Evaluation page's "How the model was chosen" section, asserted against the API's values."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store
from exportspredictor import evaluation

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "6_Model_Evaluation.py"
SECTION = "How the model was chosen"


def _run(monkeypatch, url: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    assert not page.exception
    return page


def _text(page: AppTest) -> str:
    elements = [*page.markdown, *page.caption, *page.success, *page.warning, *page.subheader, *page.info]
    return " ".join(element.value for element in elements)


def _tables(page: AppTest) -> list:
    return [element.value for element in page.dataframe]


@pytest.fixture
def served(monkeypatch, live_api, fixture_artefacts, tmp_path):
    """The page and the API's evaluation for the fixture artefact, with v2's figures beside it and a
    v2 whose Tweedie deviance is lower, so the Blend has traded something away."""
    report = {**fixture_artefacts.evaluation}
    deviance = report["accuracy"]["all_exporters"]["overall"]["model_tweedie_deviance"]
    v2 = {"version": "v2-version", "uk_ranking": 0.5115, "naive": 0.5233, "lift": 5.23, "size_matched_lift": 0.788,
          "tweedie_deviance_all_exporters": deviance / 2, "tweedie_deviance_uk": 568.6}
    report["earlier_versions"] = evaluation.earlier_versions(report, v2=v2)
    store.save(replace(fixture_artefacts, evaluation=report), tmp_path)
    with live_api(tmp_path) as url:
        yield _run(monkeypatch, url), httpx.get(f"{url}/evaluation").json()


def test_page_has_a_how_the_model_was_chosen_section(served):
    page, _ = served

    assert SECTION in [element.value for element in page.subheader]


def test_page_shows_the_selection_measure_the_model_form_and_the_win_threshold(served):
    page, body = served
    choice = body["model_choice"]
    text = _text(page)

    assert choice["text"] in text
    assert choice["model_form"]["text"] in text
    assert f"{choice['win_threshold']:.4f}" in text


def test_page_shows_every_experiment_and_whether_it_was_kept(served):
    page, body = served
    experiments = body["model_choice"]["experiments"]
    table = next(t for t in _tables(page) if "Experiment" in t.columns)

    assert list(table["Experiment"]) == [e["label"] for e in experiments]
    assert list(table["Selection Measure"]) == pytest.approx([e["selection_measure"] for e in experiments])
    assert list(table["Kept"]) == [e["kept"] or "no" for e in experiments]


def test_page_shows_the_test_year_ranking_beside_the_naive_baseline_and_earlier_versions(served):
    page, body = served
    block = body["model_choice"]["test_year_ranking"]
    table = next(t for t in _tables(page) if "UK ranking" in t.columns and "Version" in t.columns)

    expected = [*block["rows"], *block["naive"]]
    assert list(table["Version"]) == [row["name"] for row in expected]
    assert list(table["UK ranking"]) == pytest.approx([row["uk_ranking"] for row in expected])
    assert block["text"] in _text(page)
    assert str(block["test_year"]) in _text(page)


def test_page_says_what_the_blend_traded_away(served):
    page, body = served
    traded = body["model_choice"]["traded_away"]

    assert traded["text"] in _text(page)
    assert f"{traded['deviance']:,.1f}" in _text(page)


def test_the_event_backtest_section_says_it_measured_the_model_alone(served):
    page, body = served
    text = _text(page)

    backtest = text.split("Event Backtest)")[1]
    assert body["event_backtest_note"]["text"] in backtest


def test_page_shows_the_five_fold_confirmation_fold_by_fold(monkeypatch, live_api, fixture_artefacts, tmp_path):
    confirmation = {
        "n_folds": 2, "validation_year": 2022, "difference": 0.0429, "win_threshold": 0.0089, "beats_v2": True,
        "winner": {"selection_measure": 0.5635}, "v2": {"selection_measure": 0.5206},
        "naive": {"selection_measure": 0.5332},
        "folds": [
            {"fold": 0, "v2": {"selection_measure": 0.5334}, "winner": {"selection_measure": 0.5771}},
            {"fold": 1, "v2": {"selection_measure": 0.5295}, "winner": {"selection_measure": 0.5745}},
        ],
    }
    ranking = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": confirmation}
    store.save(replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": ranking}), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)
        served = httpx.get(f"{url}/evaluation").json()["model_choice"]["confirmation"]

    table = next(t for t in _tables(page) if "Fold" in t.columns)
    assert list(table["Fold"]) == [str(fold["fold"]) for fold in served["folds"]] + ["All folds"]
    assert list(table["Chosen Model Form"]) == pytest.approx([f["winner"] for f in served["folds"]] + [served["winner"]])
    assert list(table["v2 (the model alone)"]) == pytest.approx([f["v2"] for f in served["folds"]] + [served["v2"]])


def test_an_artefact_without_experiment_results_still_renders_the_section_saying_so(
    monkeypatch, live_api, fixture_artefacts, tmp_path
):
    metadata = {key: value for key, value in fixture_artefacts.metadata.items() if key != "ranking_experiments"}
    store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)
        choice = httpx.get(f"{url}/evaluation").json()["model_choice"]

    assert SECTION in [element.value for element in page.subheader]
    assert choice["recorded"] is False and choice["text"] in _text(page)
    assert not any("Experiment" in t.columns for t in _tables(page))
