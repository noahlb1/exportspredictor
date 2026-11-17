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


def test_page_shows_the_imputed_observed_tariff_split(page, fixture_artefacts):
    report = fixture_artefacts.evaluation
    tables = [element.value for element in page.dataframe]
    split = next(t for t in tables if "Applied Tariff" in t.columns)

    assert list(split["Applied Tariff"]) == ["Observed", "Carried forward from 2021"]
    imputed = report["accuracy"]["uk_out_of_fold"]["by_tariff"][1]["rows"]
    assert list(split["Rows"]) == [0, imputed]
    assert report["tariff_split_text"] in _text(page)


def test_page_shows_the_potential_range_coverage(page, fixture_artefacts):
    coverage = fixture_artefacts.evaluation["potential_range"]
    tables = [element.value for element in page.dataframe]

    assert "Potential Range" in _text(page)
    assert f"{coverage['coverage']:.0%}" in _text(page) and f"{coverage['target']:.0%}" in _text(page)
    assert any("Coverage" in t.columns and "Exporter" in t.columns for t in tables)


def test_page_drops_no_tariffs_and_lists_v2s_limitations(page):
    text = _text(page)

    assert "No tariffs" not in text
    for title in ["Tariffs carried forward from 2021", "Excluded Products", "Trained on the top 100 Exporters",
                  "Values are nominal US dollars", "What the Potential Range assumes"]:
        assert title in text, title


def test_page_says_how_the_event_backtest_sets_the_tariff(page, fixture_artefacts):
    assert fixture_artefacts.evaluation["event_backtest"]["events"][0]["states"] in _text(page)


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


def test_page_shows_lift_with_its_interval_as_a_tendency(page, fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]

    assert lift["text"] in _text(page)
    assert "Lift" in _text(page)


def test_page_shows_size_matched_lift_next_to_raw_lift_and_explains_the_difference(page, fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]
    tables = [element.value for element in page.dataframe]
    figures = next(t for t in tables if "Lift" in t.columns and "95% interval" in t.columns)
    text = _text(page)

    assert list(figures["Measure"]) == ["Raw Lift", "Size-matched Lift"]
    assert list(figures["Lift"]) == [pytest.approx(lift["lift"]), pytest.approx(lift["size_matched"]["lift"])]
    assert "mostly reflects pair size" in text
    assert "tendency" in text
    assert "2021–2022" in text  # the Gap Period the Export Gaps are averaged over


def test_page_shows_the_event_backtest_predicted_against_observed_with_its_control_group(page, fixture_artefacts):
    backtest = fixture_artefacts.evaluation["event_backtest"]
    tables = [element.value for element in page.dataframe]
    effects = next(t for t in tables if "Observed (difference-in-differences)" in t.columns)

    assert "Predicted by the model" in effects.columns
    assert list(effects["Year"]) == ["2022", "2023", "2022–2023"]
    assert "India" in _text(page)  # the control Market, by name
    assert backtest["caveat"] in _text(page)
