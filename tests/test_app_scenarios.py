"""The Streamlit Scenarios page and the ranking toggle, run with AppTest against a live API on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with India.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
SCENARIOS = APP_DIR / "pages" / "3_Scenarios.py"
PRODUCT_TO_MARKETS = APP_DIR / "pages" / "1_Product_to_Markets.py"


@pytest.fixture
def api_url(fixture_api_url):
    return fixture_api_url


def _scenarios_page(monkeypatch, api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(SCENARIOS), default_timeout=30)
    page.run()
    return page


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
    effects = list(pairs["Scenario Effect (USD)"])
    assert effects == sorted(effects, reverse=True)


def test_a_scenario_that_changes_nothing_says_so(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["JPN"]).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert any("changes nothing" in element.value for element in page.warning)
    assert not page.dataframe


def test_product_to_markets_can_rank_by_blocked_opportunity_value(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(PRODUCT_TO_MARKETS), default_timeout=30)
    page.run()
    page.text_input(key="product_query").input("220830").run()

    page.radio(key="rank_by").set_value("blocked").run()

    assert not page.exception
    table = page.dataframe[0].value
    assert list(table["Market"])[0] == "India"
    values = list(table["Blocked Opportunity value (£)"])
    assert values == sorted(values, reverse=True)
