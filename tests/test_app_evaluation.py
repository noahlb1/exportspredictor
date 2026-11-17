"""The Model Evaluation page, run with AppTest against a live local API on fixture artefacts."""

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store
from exportspredictor.evaluation import LIMITATIONS

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "5_Model_Evaluation.py"


def _run(monkeypatch, url: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    assert not page.exception
    return page


def _text(page: AppTest) -> str:
    elements = [*page.markdown, *page.caption, *page.success, *page.warning, *page.subheader, *page.info]
    return " ".join(element.value for element in elements)


@pytest.fixture
def page(monkeypatch, live_api, fixture_artefacts_dir):
    with live_api(fixture_artefacts_dir) as url:
        yield _run(monkeypatch, url)


def test_page_shows_a_plain_english_headline_against_the_naive_baseline(page, fixture_artefacts):
    headline = fixture_artefacts.evaluation["headline"]
    shown = page.success if headline["beats_naive"] else page.warning

    assert any(headline["text"] in element.value for element in shown)


def test_page_shows_metrics_and_the_weakest_sectors_and_regions(page, fixture_artefacts):
    tables = [element.value for element in page.dataframe]
    sector_table = next(t for t in tables if "Sector" in t.columns)
    region_table = next(t for t in tables if "Region" in t.columns)
    weakest = fixture_artefacts.evaluation["accuracy"]["uk_out_of_fold"]["by_sector"][0]["group"]

    assert any("Naive baseline" in t.columns for t in tables)
    assert sector_table["Sector"].iloc[0] == weakest
    assert set(region_table["Region"]) == {"East Asia & Pacific", "North America", "South Asia"}
    assert "Weakest Sectors" in _text(page) and "Weakest regions" in _text(page)


def test_page_lists_every_known_limitation(page):
    text = _text(page)

    for limitation in LIMITATIONS:
        assert limitation["title"] in text


def test_page_states_the_test_year_and_how_the_models_were_trained(page):
    text = _text(page)

    assert "2023" in text and "never" in text
    assert "every non-UK row" in text  # the fixture keeps them all; real data states its sample


def test_page_states_the_non_uk_training_sample(monkeypatch, live_api, fixture_artefacts, tmp_path):
    sampled = {**fixture_artefacts.evaluation, "non_uk_sampling_rate": 0.3}
    store.save(replace(fixture_artefacts, evaluation=sampled), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)

    assert "30% sample of non-UK rows" in _text(page)


def test_page_says_so_when_the_artefact_has_no_evaluation(monkeypatch, live_api, fixture_artefacts, tmp_path):
    store.save(replace(fixture_artefacts, evaluation=None), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)

    assert any("no evaluation" in element.value.lower() for element in page.warning)
