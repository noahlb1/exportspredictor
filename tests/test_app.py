"""The Streamlit ranking pages, run with AppTest against a live local API on fixture artefacts."""

import io
import socket
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
import streamlit
import uvicorn
from streamlit.testing.v1 import AppTest

from api.main import create_app

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGE = APP_DIR / "pages" / "1_Product_to_Markets.py"
MARKET_PAGE = APP_DIR / "pages" / "2_Market_to_Products.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def api_url(fixture_artefacts_dir):
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(fixture_artefacts_dir), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.05)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def _run_page(monkeypatch, api_url: str, query: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    page.text_input(key="product_query").input(query).run()
    return page


def test_page_shows_markets_ranked_by_export_gap(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    assert not page.exception
    table = page.dataframe[0].value
    assert list(table["Market"]) and set(table["Market"]) == {"Japan", "Canada", "India", "Hong Kong"}
    gaps = list(table["Export Gap, 2022–2024 average (USD, nominal)"])
    assert gaps == sorted(gaps, reverse=True)


def test_page_shows_baseline_year_banner(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    banner = " ".join(element.value for element in page.info)
    assert "2024" in banner and "provisional" in banner
    assert "Export Gaps are 2022–2024 averages" in banner


@pytest.mark.parametrize("market_page", [False, True], ids=["product", "market"])
def test_ranking_values_are_labelled_as_gap_period_averages(monkeypatch, api_url, market_page):
    page = _market_page(monkeypatch, api_url, "JPN") if market_page else _run_page(monkeypatch, api_url, "220830")

    columns = list(page.dataframe[0].value.columns)
    for value in ("Export Potential", "Actual Exports", "Export Gap"):
        assert f"{value}, 2022–2024 average (USD, nominal)" in columns
    # The Blocked Opportunity value is a 2024 Scenario Effect, not an average.
    assert "Blocked Opportunity value (USD, nominal)" in columns


def test_searching_by_everyday_words_finds_the_product(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "smartphone")

    assert not page.exception
    assert page.selectbox(key="hs6").value == "851712"
    assert any("851712" in element.value for element in page.subheader)


def test_choosing_among_several_matches_shows_that_product(monkeypatch, api_url):
    # Sector-level prefix "8": 851712 and 870323.
    page = _run_page(monkeypatch, api_url, "8")
    options = page.selectbox(key="hs6").options

    assert len(options) == 2
    page.selectbox(key="hs6").select_index(1).run()
    assert any("870323" in element.value for element in page.subheader)


def test_unknown_search_text_shows_a_message(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "zzz-not-a-product")

    assert not page.exception
    assert any("zzz-not-a-product" in element.value for element in page.warning)
    assert not page.dataframe


def test_api_down_shows_a_clear_message(monkeypatch):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", f"http://127.0.0.1:{_free_port()}")
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()

    assert not page.exception
    assert any("API" in element.value for element in page.error)


def _market_page(monkeypatch, api_url: str, iso3: str | None = "JPN") -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(MARKET_PAGE), default_timeout=30)
    page.run()
    if iso3 is not None:
        page.selectbox(key="iso3").set_value(iso3).run()
    return page


def _captions(page: AppTest) -> str:
    return " ".join(element.value for element in page.caption)


def test_market_page_ranks_products_by_export_gap(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    assert not page.exception
    assert any("Japan" in element.value for element in page.subheader)
    table = page.dataframe[0].value
    assert set(table["HS6"]) == {"711319", "220830", "300490", "851712", "870323"}
    gaps = list(table["Export Gap, 2022–2024 average (USD, nominal)"])
    assert gaps == sorted(gaps, reverse=True)


def test_market_page_shows_the_uk_trade_agreement(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "IND")

    assert set(page.dataframe[0].value["UK Trade Agreement in force"]) == {"No"}
    assert any("no UK Trade Agreement" in element.value for element in page.markdown)


def test_product_page_shows_which_markets_have_a_uk_trade_agreement(monkeypatch, api_url):
    table = _run_page(monkeypatch, api_url, "220830").dataframe[0].value

    assert dict(zip(table["Market"], table["UK Trade Agreement in force"])) == {
        "Japan": "Yes",
        "Canada": "Yes",
        "India": "No",
        "Hong Kong": "No",
    }


def test_sector_filter_narrows_the_market_ranking(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    page.selectbox(key="sector").set_value("85").run()

    assert list(page.dataframe[0].value["HS6"]) == ["851712"]


def test_region_filter_narrows_the_product_ranking(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    page.selectbox(key="region").set_value("South Asia").run()

    assert list(page.dataframe[0].value["Market"]) == ["India"]


def test_region_filter_narrows_the_market_choice(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, iso3=None)

    page.selectbox(key="region").set_value("South Asia").run()

    assert page.selectbox(key="iso3").options == ["India"]


def test_sector_filter_narrows_the_product_search(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "8")  # 851712 and 870323

    page.selectbox(key="sector").set_value("87").run()

    assert len(page.selectbox(key="hs6").options) == 1
    assert any("870323" in element.value for element in page.subheader)


def test_min_gap_filter_drops_smaller_gaps(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")
    everything = page.dataframe[0].value

    page.number_input(key="min_gap_usd").set_value(0).run()

    table = page.dataframe[0].value
    assert 0 < len(table) < len(everything)
    assert (table["Export Gap, 2022–2024 average (USD, nominal)"] >= 0).all()


def test_filters_that_match_nothing_say_so(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    page.number_input(key="min_gap_usd").set_value(10**12).run()

    assert not page.exception
    assert not page.dataframe
    assert any("No Opportunities" in element.value for element in page.info)


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


@pytest.mark.parametrize("make_page", [_market_page, lambda m, url: _run_page(m, url, "220830")], ids=["market", "product"])
def test_csv_export_matches_the_ranking_on_screen(monkeypatch, api_url, downloads, make_page):
    page = make_page(monkeypatch, api_url)

    on_screen = page.dataframe[0].value.drop(columns="Why?")  # links are navigation, not data
    # keep_default_na: a row without caveats exports an empty Caveats cell, not a missing value
    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str, "Sector": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(exported, on_screen.reset_index(drop=True), check_dtype=False)
    assert len(page.get("download_button")) == 1


def test_market_page_api_down_shows_a_clear_message(monkeypatch):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", f"http://127.0.0.1:{_free_port()}")
    page = AppTest.from_file(str(MARKET_PAGE), default_timeout=30)
    page.run()

    assert not page.exception
    assert any("API" in element.value for element in page.error)


def test_market_page_can_rank_by_blocked_opportunity_value(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "IND")

    page.radio(key="rank_by").set_value("blocked").run()

    assert not page.exception
    values = list(page.dataframe[0].value["Blocked Opportunity value (USD, nominal)"])
    assert values == sorted(values, reverse=True) and values[0] > 0
    assert "Blocked Opportunity value" in _captions(page)


# --- Flagged Products and Re-export Hubs: caveat badges and hide filters (ADR 0005) -------

JEWELLERY = "711319"  # the fixture's Flagged Product; Hong Kong is its Re-export Hub


def _caveats(table: pd.DataFrame, key: str) -> dict[str, str]:
    """The Flagged Product and Re-export Hub badges per row (the Uncertain Opportunity badge left out)."""
    adr_0005 = ("Flagged Product", "Re-export Hub")
    return {
        name: ", ".join(badge for badge in caveats.split(", ") if badge in adr_0005)
        for name, caveats in zip(table[key], table["Caveats"], strict=True)
    }


def test_ranking_rows_show_caveat_badges_for_flagged_products_and_re_export_hubs(monkeypatch, api_url):
    whisky = _caveats(_run_page(monkeypatch, api_url, "220830").dataframe[0].value, "Market")
    jewellery = _caveats(_run_page(monkeypatch, api_url, JEWELLERY).dataframe[0].value, "Market")

    assert whisky == {"Japan": "", "Canada": "", "India": "", "Hong Kong": "Re-export Hub"}
    assert jewellery["Japan"] == "Flagged Product"
    assert jewellery["Hong Kong"] == "Flagged Product, Re-export Hub"


def test_hide_filters_are_off_by_default_and_hide_the_flagged_rows(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")
    assert page.checkbox(key="hide_flagged").value is False
    assert page.checkbox(key="hide_hubs").value is False

    page.checkbox(key="hide_hubs").check().run()

    assert set(page.dataframe[0].value["Market"]) == {"Japan", "Canada", "India"}

    market_page = _market_page(monkeypatch, api_url, "JPN")
    assert JEWELLERY in set(market_page.dataframe[0].value["HS6"])
    market_page.checkbox(key="hide_flagged").check().run()

    assert set(market_page.dataframe[0].value["HS6"]) == {"220830", "300490", "851712", "870323"}
