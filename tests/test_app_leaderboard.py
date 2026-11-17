"""The Leaderboard on Home, run with AppTest against a live local API on fixture artefacts."""

import io
import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

HOME = Path(__file__).resolve().parents[1] / "app" / "Home.py"
GAP = "Export Gap, 2022–2024 average (USD, nominal)"
RELATIVE = "Relative Export Gap, 2022–2024 average"
RANGE = "Potential Range, 2022–2024 average (USD, nominal)"


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


def _home(monkeypatch, api_url: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(HOME), default_timeout=30)
    page.run()
    return page


def _metrics(page: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in page.metric}


def test_home_opens_on_the_leaderboard_of_every_uk_pair(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    assert not page.exception
    table = page.dataframe[0].value
    assert len(table) == 20  # every fixture UK Product–Market pair
    gaps = list(table[GAP])
    assert gaps == sorted(gaps, reverse=True)
    assert {"HS6", "Product", "Market", "Opportunity Type", RELATIVE, "Caveats"} <= set(table.columns)
    assert set(table["Opportunity Type"]) == {"Expand"}
    assert "Re-export Hub" in set(table["Caveats"]) and "Flagged Product" in set(table["Caveats"])
    metrics = _metrics(page)
    assert metrics["Product–Market pairs"] == "20"
    positive = sum(gap for gap in gaps if gap > 0)
    assert metrics["Total positive Export Gap, 2022–2024 average"] == f"${positive / 1e3:,.1f}k"


def test_relative_toggle_shows_the_minimum_selector_and_reranks(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    assert "min_potential_usd" not in {box.key for box in page.selectbox}

    page.radio(key="rank_by").set_value("relative").run()

    minimum = page.selectbox(key="min_potential_usd")
    assert minimum.value == 1e6 and minimum.options == ["$0", "$100.0k", "$1.0m", "$10.0m", "$100.0m"]
    # Every fixture pair's Export Potential is below $1M.
    assert not page.dataframe and _metrics(page)["Product–Market pairs"] == "0"

    minimum.set_value(0.0).run()

    table = page.dataframe[0].value
    relative = list(table[RELATIVE])
    assert len(table) == 20 and relative == sorted(relative, reverse=True)


def test_filters_change_the_rows_and_the_summary(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    page.selectbox(key="market").set_value("JPN").run()
    assert set(page.dataframe[0].value["Market"]) == {"Japan"}
    assert _metrics(page)["Product–Market pairs"] == "5"

    page.selectbox(key="market").set_value(None).run()
    page.selectbox(key="trade_agreement").set_value(False).run()
    page.checkbox(key="hide_hubs").check().run()
    assert set(page.dataframe[0].value["Market"]) == {"India"}

    page.selectbox(key="opportunity_type").set_value("New Product").run()
    assert not page.dataframe
    assert any("No Opportunities" in element.value for element in page.info)


def test_csv_export_matches_the_leaderboard_on_screen(monkeypatch, fixture_api_url, downloads):
    page = _home(monkeypatch, fixture_api_url)

    on_screen = page.dataframe[0].value.drop(columns="Why?")
    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str, "Sector": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(exported, on_screen.reset_index(drop=True), check_dtype=False)


def test_the_excluded_products_list_is_reachable_from_the_leaderboard(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    assert any("Excluded Products" in expander.label for expander in page.expander)


def test_the_leaderboard_shows_each_potential_range_badges_uncertain_opportunities_and_can_hide_them(
    monkeypatch, fixture_api_url
):
    page = _home(monkeypatch, fixture_api_url)

    table = page.dataframe[0].value
    ranges = list(table[RANGE])
    assert all(re.fullmatch(r"\$[0-9.,]+(bn|m|k)?–\$[0-9.,]+(bn|m|k)?", text) for text in ranges)
    uncertain = table["Caveats"].str.contains("Uncertain Opportunity")
    assert 0 < uncertain.sum() < len(table)
    assert any("Uncertain Opportunit" in caption.value for caption in page.caption)

    page.checkbox(key="hide_uncertain").check().run()
    shown = page.dataframe[0].value
    assert list(shown["HS6"] + shown["Market"]) == list((table["HS6"] + table["Market"])[~uncertain])
    assert _metrics(page)["Product–Market pairs"] == str(len(table) - uncertain.sum())
