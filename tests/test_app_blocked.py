"""The Blocked Opportunities page, run with AppTest against a live local API on fixture artefacts.

The fixture Baseline has UK Trade Agreements with Japan and Canada, none with India or Hong Kong.
"""

import io
import re
from dataclasses import replace

import httpx
import pandas as pd
import pytest
import streamlit
from conftest import run_page
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store

PAGE = "pages/blocked_opportunities.py"
VALUE = "Blocked Opportunity value, illustrative (USD, nominal)"
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
    return run_page(PAGE)


def _metrics(page: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in page.metric}


def test_the_page_ranks_every_uk_pair_by_blocked_opportunity_value(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    assert not page.exception
    assert page.header[0].value == "Blocked Opportunities"
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
    assert blocked_total == 0 and metrics["Total Blocked Opportunity value, illustrative"] == "$0"
    assert metrics["Blocked Opportunities"] == str(int(table["Blocked Opportunity"].sum()))


def test_filters_change_the_rows_and_the_summary(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    page.selectbox(key="blocked_market").set_value("IND").run()
    assert set(page.dataframe[0].value["Market"]) == {"India"}
    assert _metrics(page)["Product–Market pairs"] == "5"

    page.selectbox(key="blocked_market").set_value(None).run()
    page.selectbox(key="blocked_trade_agreement").set_value(False).run()
    page.checkbox(key="blocked_hide_hubs").check().run()
    assert set(page.dataframe[0].value["Market"]) == {"India"}

    page.selectbox(key="blocked_opportunity_type").set_value("New Product").run()
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
    home = run_page()

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


def test_the_value_heading_metric_and_csv_export_say_illustrative(monkeypatch, fixture_api_url, downloads):
    word = httpx.get(f"{fixture_api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _page(monkeypatch, fixture_api_url)

    headings = [column for column in page.dataframe[0].value.columns if "Blocked Opportunity value" in column]
    exported = pd.read_csv(io.StringIO(downloads[-1]), nrows=0).columns
    exported = [column for column in exported if "Blocked Opportunity value" in column]
    assert headings and all(word in heading for heading in headings)
    assert exported == headings
    assert all(word in metric.label for metric in page.metric if "Blocked Opportunity value" in metric.label)


def test_with_no_benchmark_or_event_backtests_the_page_keeps_the_label_and_says_they_were_not_run(
    monkeypatch, fixture_artefacts, tmp_path, live_api
):
    report = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "event_backtest"}
    store.save(replace(fixture_artefacts, ppml_benchmark=None, evaluation=report), tmp_path)

    with live_api(tmp_path) as api_url:
        note = httpx.get(f"{api_url}/opportunities/blocked").json()["note"]
        page = _page(monkeypatch, api_url)

    assert not page.exception
    text = _page_text(page)
    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["run"] is False and note[part]["text"] in text
    assert page.dataframe[0].value[VALUE].max() > 0
