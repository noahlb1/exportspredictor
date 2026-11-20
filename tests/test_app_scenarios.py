"""The Streamlit Scenarios page and the ranking toggle, run with AppTest against a live API on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with India.
"""

import httpx
import pytest
import streamlit
from conftest import run_page
from streamlit.testing.v1 import AppTest

from app import client

SCENARIOS = "pages/scenarios.py"


@pytest.fixture
def api_url(fixture_api_url):
    return fixture_api_url


def _scenarios_page(monkeypatch, api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(SCENARIOS)


def test_a_preset_scenario_shows_an_associational_summary_and_markets(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Trading Bloc preset").run()
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()

    assert not page.exception
    text = " ".join(element.value for element in page.markdown)
    assert "associates" in text
    markets = page.dataframe[0].value
    assert set(markets["Market"]) == {"Japan", "Canada"}


def test_a_custom_scenario_shows_blocked_opportunities_and_a_drill_down(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["IND"]).run()
    page.number_input(key="min_share").set_value(0.0).run()
    page.number_input(key="min_usd").set_value(0.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert any("Blocked Opportunities" in element.value for element in page.subheader)
    page.selectbox(key="drill_market").set_value("IND").run()
    pairs = page.dataframe[-1].value
    assert len(pairs) == 5 and set(pairs["Market"]) == {"India"}
    effects = list(pairs["Scenario Effect, illustrative (USD, nominal)"])
    assert effects == sorted(effects, reverse=True)


def test_a_scenario_that_changes_nothing_says_so(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["JPN"]).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert page.warning  # the API's summary, shown as a warning instead of a table
    assert not page.dataframe


def _run_india(monkeypatch, api_url) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["IND"]).run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _page_text(page: AppTest) -> str:
    return " ".join(element.value for element in [*page.markdown, *page.caption, *page.info, *page.warning])


def test_the_note_is_not_shown_above_the_inputs_before_a_run(monkeypatch, api_url, fixture_artefacts):
    page = _scenarios_page(monkeypatch, api_url)

    note_text = httpx.get(f"{api_url}/labels").json()["illustrative"]["text"]

    assert not page.exception
    text = _page_text(page)
    assert fixture_artefacts.evaluation["event_backtest"]["text"] not in text
    assert note_text not in text


def test_a_run_shows_the_apis_scenario_note_beside_the_scenario_effect(monkeypatch, api_url):
    note = httpx.post(f"{api_url}/scenarios", json={"add_agreements": ["IND"], "limit": 100}).json()["note"]

    text = _page_text(_run_india(monkeypatch, api_url))

    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text
    assert note["blend"]["text"] in text


def test_scenario_effect_column_headings_say_illustrative(monkeypatch, api_url):
    word = httpx.get(f"{api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _run_india(monkeypatch, api_url)

    headings = [column for frame in page.dataframe for column in frame.value.columns if "Scenario Effect" in column]
    assert headings and all(word in heading for heading in headings)
    assert all(word in metric.label for metric in page.metric if "Scenario Effect" in metric.label)


def test_the_drill_down_asks_the_api_for_the_chosen_markets_own_top_pairs(monkeypatch, api_url):
    posted = []
    original = client.post

    def record(path, body):
        posted.append(body)
        return original(path, body)

    monkeypatch.setattr(client, "post", record)
    streamlit.cache_data.clear()  # earlier tests may have cached this Scenario's responses
    page = _scenarios_page(monkeypatch, api_url)
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()
    page.selectbox(key="drill_market").set_value("CAN").run()

    assert not page.exception
    assert any(body.get("market") == "CAN" for body in posted)
    assert set(page.dataframe[-1].value["Market"]) == {"Canada"}


def test_threshold_inputs_default_to_the_apis_thresholds(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)
    defaults = client.get("/scenarios/thresholds")

    assert page.number_input(key="min_share").value == defaults["min_effect_share"] * 100
    assert page.number_input(key="min_usd").value == defaults["min_effect_usd"] / 1e6


def test_the_tariff_mode_runs_and_says_when_the_effect_flattens_beyond_the_training_range(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.number_input(key="tariff_rate").set_value(1000.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert "associates" in " ".join(element.value for element in page.markdown)
    clip = [element.value for element in page.warning if "training range" in element.value]
    assert clip and "5" in clip[0]  # the clipped pairs, counted by the API
    pairs = page.dataframe[-1].value
    assert set(pairs["Market"]) == {"Japan"}


def test_a_tariff_within_the_training_range_shows_no_clip_message(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.multiselect(key="tariff_sectors").set_value(["22"]).run()
    page.number_input(key="tariff_rate").set_value(20.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert not [element.value for element in page.warning if "training range" in element.value]
    assert set(page.dataframe[0].value["Market"]) == {"Japan"}
