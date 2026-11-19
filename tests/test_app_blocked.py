"""The Blocked Opportunities page, run with AppTest against a live local API on fixture artefacts.

The fixture Baseline has UK Trade Agreements with Japan and Canada, none with India or Hong Kong.
"""

import io
import re
from pathlib import Path

import httpx
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "3_Blocked_Opportunities.py"
VALUE = "Blocked Opportunity value (USD, nominal)"
# Wording that would read a Blocked Opportunity value as a cause or a forecast.
CAUSAL = re.compile(
    r"(?<!not )\b(causes?|caused|will (raise|add|unlock|increase)|would (raise|add|unlock|increase)|unlocks?)\b",
    re.IGNORECASE,
)


@pytest.fixture
def downloads(monkeypatch):
    """Record what each download button offers, then render it as usual."""
    offered = []
    original = streamlit.download_button

    def record(label, data, *args, **kwargs):
        offered.append(data)
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", record)
    return offered


def _page(monkeypatch, api_url: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    return page


def _metrics(page: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in page.metric}


def test_the_page_ranks_every_uk_pair_by_blocked_opportunity_value(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    assert not page.exception
    assert [title.value for title in page.title] == ["Blocked Opportunities"]
    table = page.dataframe[0].value
    assert len(table) == 20
    values = list(table[VALUE])
    assert values == sorted(values, reverse=True) and values[0] > 0
    assert set(table.loc[table["Market"].isin(["Japan", "Canada"]), VALUE]) == {0}
    # The page has one order, so there is no Rank by toggle.
    assert "rank_by" not in {radio.key for radio in page.radio}
    metrics = _metrics(page)
    assert metrics["Product–Market pairs"] == "20"
    # The total covers the Blocked Opportunities only; the fixture's values are under the $1M threshold.
    blocked_total = table.loc[table["Blocked Opportunity"], VALUE].sum()
    assert blocked_total == 0 and metrics["Total Blocked Opportunity value"] == "$0"
    assert metrics["Blocked Opportunities"] == str(int(table["Blocked Opportunity"].sum()))


def test_filters_change_the_rows_and_the_summary(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    page.selectbox(key="market").set_value("IND").run()
    assert set(page.dataframe[0].value["Market"]) == {"India"}
    assert _metrics(page)["Product–Market pairs"] == "5"

    page.selectbox(key="market").set_value(None).run()
    page.selectbox(key="trade_agreement").set_value(False).run()
    page.checkbox(key="hide_hubs").check().run()
    assert set(page.dataframe[0].value["Market"]) == {"India"}

    page.selectbox(key="opportunity_type").set_value("New Product").run()
    assert not page.dataframe
    assert any("No Opportunities" in element.value for element in page.info)


def test_csv_export_matches_the_table_on_screen(monkeypatch, fixture_api_url, downloads):
    page = _page(monkeypatch, fixture_api_url)

    on_screen = page.dataframe[0].value.drop(columns="Why?")
    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str, "Sector": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(exported, on_screen.reset_index(drop=True), check_dtype=False)


def test_every_caption_uses_associational_language(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    texts = [element.value for element in [*page.caption, *page.markdown, *page.info]]
    assert not [text for text in texts if CAUSAL.search(text)]
    about_value = [text for text in texts if "Blocked Opportunity value" in text]
    assert about_value and all("associat" in text for text in about_value)  # associates, associations


def test_the_leaderboard_no_longer_offers_a_blocked_ranking(monkeypatch, fixture_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", fixture_api_url)
    home = AppTest.from_file(str(PAGE.parents[1] / "Home.py"), default_timeout=30)
    home.run()

    assert home.radio(key="rank_by").options == ["Export Gap", "Relative Export Gap"]



def _page_text(page: AppTest) -> str:
    return " ".join(element.value for element in [*page.markdown, *page.caption, *page.info])


def test_the_page_shows_the_apis_scenario_note_beside_its_totals(monkeypatch, fixture_api_url):
    note = httpx.get(f"{fixture_api_url}/opportunities/blocked").json()["note"]

    page = _page(monkeypatch, fixture_api_url)

    assert not page.exception
    text = _page_text(page)
    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text
    assert note["blend"]["text"] in text
